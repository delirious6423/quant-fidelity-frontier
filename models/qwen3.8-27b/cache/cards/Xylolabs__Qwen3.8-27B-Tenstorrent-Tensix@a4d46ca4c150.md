---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- tenstorrent
- tensix
- tt-metal
- ttnn
- tt-forge
- block-floating-point
- bfloat4_b
- bfloat8_b
- quantized
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Qwen3.8-27B for Tenstorrent Tensix</h1>

[![Base model](https://img.shields.io/badge/base-Qwen%2FQwen3.8--27B-blue)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Formats](https://img.shields.io/badge/formats-BFP4%20%2F%20BFP8-5B2C83)](#folders)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://huggingface.co/Xylolabs/Qwen3.8-27B-Tenstorrent-Tensix/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-tensix--bfp--gptq-black)](https://github.com/XylolabsInc/tensix-bfp-gptq)
[![Tenstorrent Tensix](https://img.shields.io/badge/Tenstorrent-Tensix-5B2C83)](https://docs.tenstorrent.com)

</div>

These weights from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) are quantized for Tenstorrent Tensix NPUs (Wormhole, Blackhole), with each format in its own folder.

## Folders

| Folder | Formats | Size (GB, packed) | Mean KL to BF16 | Answers |
| --- | --- | --- | --- | --- |
| `bfp4/` | Attention, DeltaNet and MLP bfloat4_b; LM head bfloat8_b; the rest bf16 | 17.596 | 0.02894 | 31/33 + 7/7 |
| `bfp8/` | Attention, DeltaNet and MLP bfloat8_b; the rest bf16 | 30.963 | 0.00074 | - |
| BF16 ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)) for `bfp4/` | - | 53.79 | 0 | - |
| Q4_K_M (llama.cpp) for `bfp4/` | - | 16.811 | 0.02081 | - |
| BF16 ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)) for `bfp8/` | - | 53.79 | 0 | - |
| Q4_K_M (llama.cpp) for `bfp8/` | - | 16.811 | 0.02084 | - |

For `bfp4/`, KL covers 16,240 positions of four texts, four sequences of up to 1,024 tokens per text, using the start of an English Wikipedia article and of a Korean one, and the model's own reply to "Write an encyclopedia article." in its chat template for each language (scored on the reply tokens only), with a mean-KL limit of 1.5 times the Q4_K_M reference. Answers: the 33 + 7 answer check.

For `bfp8/`, KL covers 4,094 positions of English and Korean text (WikiText-2 test, Korean Wikipedia), as plain text, with a mean-KL limit of 1.5 times the Q4_K_M reference.

## How to run

```bash
hf download Xylolabs/Qwen3.8-27B-Tenstorrent-Tensix --include "bfp4/*" --local-dir Qwen3.8-27B-Tenstorrent-Tensix
export WEIGHTS_DIR="$PWD/Qwen3.8-27B-Tenstorrent-Tensix/bfp4"
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen3.8-27B-Tenstorrent-Tensix/bfp4")
model = AutoModelForCausalLM.from_pretrained("Qwen3.8-27B-Tenstorrent-Tensix/bfp4", dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Run on Tensix

Pack one matrix:

```python
import json
import torch
import ttnn
from safetensors import safe_open

d = "Qwen3.8-27B-Tenstorrent-Tensix/bfp4"
name = "model.language_model.layers.0.mlp.down_proj.weight"
fmt = json.load(open(f"{d}/quant4.json"))["layers"]["0"]["mlp.down_proj"]["format"]
dtype = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[fmt]
shard = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"][name]
with safe_open(f"{d}/{shard}", "pt") as f:
    w = f.get_tensor(name).t().contiguous()

device = ttnn.open_device(device_id=0)
w_tt = ttnn.as_tensor(w, dtype=dtype, layout=ttnn.TILE_LAYOUT,
                      device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
assert torch.equal(ttnn.to_torch(w_tt).float(), w.float())
ttnn.close_device(device)
```

```bash
python Qwen3.8-27B-Tenstorrent-Tensix/bfp4/pack_tensix.py Qwen3.8-27B-Tenstorrent-Tensix/bfp4
python Qwen3.8-27B-Tenstorrent-Tensix/bfp4/verify_grid.py Qwen3.8-27B-Tenstorrent-Tensix/bfp4
```

Add `--device` to round-trip the packed values.

```
# build tt-metal v0.81.1 as the demo's README describes, then
export MODEL_WEIGHTS_DIR="$WEIGHTS_DIR"
export QWEN_PRECISION_CONFIG=baseline
python models/demos/qwen38_27b_qb2/demo/text_demo.py --full --length 128 --generate 128 --output qwen38.json
```

To use another format, download its folder the same way and substitute its local path.

## Updates

- 2026-11-27: BFP4 and BFP8 now in one repo, formerly Qwen3.8-27B-Tenstorrent-Tensix-BFP4 and Qwen3.8-27B-Tenstorrent-Tensix-BFP8.
- 2026-11-24: bfp4/: Evaluation redone (16,240 positions): KL 0.0289, 1.39 times Q4_K_M, answers 31/33 and 7/7.

## License

Apache 2.0, as the base model (`LICENSE`). The changes are listed in `NOTICE`. `pack_tensix.py` and
`verify_grid.py` are by Xylolabs under the same license.

Each folder carries its own `LICENSE` and `NOTICE`.

## Citation

For Qwen family background, cite the Qwen3 technical report:

```bibtex
@misc{qwen3technicalreport,
    title={Qwen3 Technical Report},
    author={Qwen Team},
    year={2025},
    eprint={2505.09388},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.09388}
}
```
