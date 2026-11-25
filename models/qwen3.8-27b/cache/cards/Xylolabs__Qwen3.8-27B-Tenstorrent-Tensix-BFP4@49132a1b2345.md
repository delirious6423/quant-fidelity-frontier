---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
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

<h1>Qwen3.8-27B for Tenstorrent Tensix (BFP4)</h1>

[![Base model](https://img.shields.io/badge/base-Qwen%2FQwen3.8--27B-blue)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Formats](https://img.shields.io/badge/formats-BFP4%20%2F%20BFP8%20head-5B2C83)](#formats)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://huggingface.co/Xylolabs/Qwen3.8-27B-Tenstorrent-Tensix-BFP4/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-tensix--bfp--gptq-black)](https://github.com/XylolabsInc/tensix-bfp-gptq)
[![Tenstorrent Tensix](https://img.shields.io/badge/Tenstorrent-Tensix-5B2C83)](https://docs.tenstorrent.com)

</div>

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized for Tenstorrent Tensix NPUs (Wormhole, Blackhole).

- The Gated DeltaNet, MLP and attention projections quantized to `bfloat4_b` (BFP4); the LM head in `bfloat8_b` (BFP8).
- **Size:** 17.596 GB packed for Tensix, against 53.79 GB in bf16.
- **Accuracy against llama.cpp Q4_K_M:** mean KL to bf16 0.0289 against Q4_K_M's 0.0208 (1.39 times). The answer check (33 + 7 items) passed on these weights with the bfloat4_b top-16 head and bf16 rescore: 31/33 and 7/7, no regressions.
- **Code:** [tensix-bfp-gptq at 5c52065](https://github.com/XylolabsInc/tensix-bfp-gptq/tree/5c52065fffec88021a0703a1bfe5266e4c674711) (its `tools/pack_tensix.py` and `tools/verify_grid.py` ship here unchanged); [tensix-bfp-gptq v0.1.0](https://github.com/XylolabsInc/tensix-bfp-gptq/releases/tag/v0.1.0) (quantization and format checks), [tensix-kernels v0.1.0](https://github.com/XylolabsInc/tensix-kernels/releases/tag/v0.1.0) and [tensix-runner v0.1.0](https://github.com/XylolabsInc/tensix-runner/releases/tag/v0.1.0) (kernels and runner for Tensix).

## Formats

| Tensors | Format |
|---|---|
| attention q, k, v, o (16 layers) | bfloat4_b |
| Gated DeltaNet q, k, v, z, a, b, output (48 layers) | bfloat4_b |
| MLP gate, up, down (64 layers) | bfloat4_b |
| LM head | bfloat8_b |
| embedding, norms and every other tensor | bf16, as in the base model |
| `mtp`, `visual` | bf16, unchanged |

`quant4.json` lists each projection's format. The bfloat4_b tensors were quantized with GPTQ and a per-block exponent search; the bfloat8_b tensors with round to nearest.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Xylolabs/Qwen3.8-27B-Tenstorrent-Tensix-BFP4"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Run on Tensix

Checked with ttnn 0.81.1 on ttsim, Tenstorrent's Blackhole simulator. Pack one matrix:

```python
import json
import torch
import ttnn
from safetensors import safe_open

d = "Qwen3.8-27B-Tenstorrent-Tensix-BFP4"  # hf download Xylolabs/Qwen3.8-27B-Tenstorrent-Tensix-BFP4 --local-dir Qwen3.8-27B-Tenstorrent-Tensix-BFP4
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

`python pack_tensix.py Qwen3.8-27B-Tenstorrent-Tensix-BFP4` checks every matrix this way (`--device` also round-trips it
through device DRAM). `python verify_grid.py Qwen3.8-27B-Tenstorrent-Tensix-BFP4` checks the grid with torch alone. The
same values also pack exactly into `ttnn.bfloat8_b`.

```
# build tt-metal v0.81.1 as the demo's README describes, then
export MODEL_WEIGHTS_DIR=/path/to/Qwen3.8-27B-Tenstorrent-Tensix-BFP4
export QWEN_PRECISION_CONFIG=baseline
python models/demos/qwen38_27b_qb2/demo/text_demo.py --full --length 128 --generate 128 --output qwen38.json
```

## Evaluation

| | Size (GB) | Mean KL to BF16 | Top-1 (%) |
|---|---:|---:|---:|
| BF16 ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B/tree/37561816e66b5b9153b1b64c10968c18c1e6f4a0)) | 53.79 | 0 | 100 |
| this model | 17.596 | 0.02894 | 92.62 |
| Q4_K_M (llama.cpp) | 16.811 | 0.02081 | 92.94 |

KL and top-1 cover 16,240 positions of four texts, four sequences of up to 1,024 tokens from each, against bf16 log-probabilities over the full vocabulary: the start of an English Wikipedia article and of a Korean one, and the model's own reply to "Write an encyclopedia article." in its chat template for each language (scored on the reply tokens only). The accuracy rule is a mean KL of at most 1.5 times Q4_K_M's, here 0.03121. Q4_K_M is a plain Q4_K_M with no importance matrix, made from the same checkpoint with llama.cpp `cd959ba`; its sha256 is `76cdd73b285057201c3ecb0c54c83486a50b325607844b5381384ff8e190e5c1`.

## Updates

- 2026-11-24: Evaluation redone on four texts (16,240 positions): KL 0.0289, 1.39 times Q4_K_M. Answer check 31/33 and 7/7.
- 2026-11-22: Corrected the bf16 size in the summary to match the evaluation table.
- 2026-11-11: First release.

## License

Apache 2.0, as the base model (`LICENSE`). The changes are listed in `NOTICE`. `pack_tensix.py` and
`verify_grid.py` are by Xylolabs under the same license. Not affiliated with Tenstorrent.

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
