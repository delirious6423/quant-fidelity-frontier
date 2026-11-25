---
license: apache-2.0
base_model:
- google/gemma-4-26B-A4B-it
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
- bfloat8_b
- quantized
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>gemma-4-26B-A4B-it for Tenstorrent Tensix (BFP8)</h1>

[![Base model](https://img.shields.io/badge/base-google%2Fgemma--4--26B--A4B--it-blue)](https://huggingface.co/google/gemma-4-26B-A4B-it)
[![Formats](https://img.shields.io/badge/formats-BFP8-5B2C83)](#formats)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP8/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-tensix--bfp--gptq-black)](https://github.com/XylolabsInc/tensix-bfp-gptq)
[![Tenstorrent Tensix](https://img.shields.io/badge/Tenstorrent-Tensix-5B2C83)](https://docs.tenstorrent.com)

</div>

[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), quantized for Tenstorrent Tensix NPUs (Wormhole, Blackhole).

- The attention, MLP and routed-expert projections quantized to `bfloat8_b` (BFP8); the LM head, embedding and norms stay bf16.
- **Size:** 27.513 GB packed for Tensix.
- **Accuracy against llama.cpp Q4_K_M:** mean KL to bf16 0.0080 against Q4_K_M's 0.1249 (0.06 times).
- **Code:** [tensix-bfp-gptq at 5c52065](https://github.com/XylolabsInc/tensix-bfp-gptq/tree/5c52065fffec88021a0703a1bfe5266e4c674711) (its `tools/pack_tensix.py` and `tools/verify_grid.py` ship here unchanged); [tensix-bfp-gptq v0.1.0](https://github.com/XylolabsInc/tensix-bfp-gptq/releases/tag/v0.1.0) (quantization and format checks), [tensix-kernels v0.1.0](https://github.com/XylolabsInc/tensix-kernels/releases/tag/v0.1.0) and [tensix-runner v0.1.0](https://github.com/XylolabsInc/tensix-runner/releases/tag/v0.1.0) (kernels and runner for Tensix).

Use the [BFP4 version](https://huggingface.co/Xylolabs/gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP4) on devices with less memory.

## Formats

| Tensors | Format |
|---|---|
| attention q, k, o (30 layers) | bfloat8_b |
| attention v (25 layers) | bfloat8_b |
| MLP gate, up, down and routed experts down, gate, up (30 layers) | bfloat8_b |
| router per-expert scale and router (30 layers) | bf16 |
| LM head (tied to the embedding), embedding, norms and every other tensor | bf16, as in the base model |
| `embed_vision`, `vision_tower` | bf16, unchanged |

`quant4.json` lists each projection's format. Quantized with GPTQ on the bfloat8_b grid.

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Xylolabs/gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP8"
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

d = "gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP8"  # hf download Xylolabs/gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP8 --local-dir gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP8
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

`python pack_tensix.py gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP8` checks every matrix this way (`--device` also
round-trips it through device DRAM). `python verify_grid.py gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP8` checks the grid
with torch alone.

```
"gemma-4-26B-A4B-it-Tenstorrent-Tensix-BFP8": {"default": {"attention": "bfp8", "shared_mlp": "bfp8", "experts": "bfp8"}}
```

## Evaluation

| | Size (GB) | Mean KL to BF16 | Top-1 (%) |
|---|---:|---:|---:|
| BF16 ([google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it/tree/4d7ae4984b7db7de8f8457170b3f1a419ee76d52)) | 50.47 | 0 | 100 |
| **this model** | 27.513 | 0.00804 | 96.97 |
| Q4_K_M (llama.cpp) | 16.796 | 0.1249 | 88.16 |

KL and top-1 cover 8,048 positions of English and Korean text (WikiText-2 test, Korean Wikipedia), in the chat template. Q4_K_M was made from the same checkpoint with llama.cpp v0.9.0 (`cd959bae`) and no importance matrix; its sha256 is `2289deaba79c84dfa5727922ac0e7aee2a4807cc5b438faebdb75644e4856c7c`.

## Updates

- 2026-11-12: First release.

## License

Apache 2.0, as the base model's card says; its repository has no LICENSE file, so `LICENSE` holds the standard
text. The changes are listed in `NOTICE`. `pack_tensix.py` and `verify_grid.py` are by Xylolabs under the same
license. Not affiliated with Tenstorrent.

## Citation

```bibtex
@misc{gemmateam2026gemma4,
      title={Gemma 4 Technical Report},
      author={Gemma Team},
      year={2026},
      eprint={2607.02770},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2607.02770},
}
```
