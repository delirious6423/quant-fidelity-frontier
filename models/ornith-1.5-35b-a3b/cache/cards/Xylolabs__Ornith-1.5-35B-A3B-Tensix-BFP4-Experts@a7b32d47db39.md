---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
language:
- en
- ko
tags:
- tenstorrent
- tensix
- tt-metal
- ttnn
- block-floating-point
- bfloat4_b
- bfloat8_b
- quantized
---

<div align="center">

<img class="block dark:hidden mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup.svg" alt="Xylolabs" width="256" height="40">
<img class="hidden dark:block mx-auto" src="https://raw.githubusercontent.com/XylolabsInc/tensix-bfp-gptq/main/assets/xylolabs-lockup-dark.svg" alt="Xylolabs" width="256" height="40">

<h1>Ornith 1.5 35B A3B text decoder for Tenstorrent Tensix (BFP4 Experts)</h1>

[![Base model](https://img.shields.io/badge/base-ornith--ai%2FOrnith--1.5--35B--A3B-blue)](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
[![Formats](https://img.shields.io/badge/formats-BFP4%20Experts%20%2F%20BFP8-5B2C83)](#formats)
[![License](https://img.shields.io/badge/license-MIT-green)](https://huggingface.co/Xylolabs/Ornith-1.5-35B-A3B-Tensix-BFP4-Experts/blob/main/LICENSE)
[![Code](https://img.shields.io/badge/code-tensix--bfp--gptq-black)](https://github.com/XylolabsInc/tensix-bfp-gptq)
[![Tenstorrent Tensix](https://img.shields.io/badge/Tenstorrent-Tensix-5B2C83)](https://docs.tenstorrent.com)

`tenstorrent` · `tensix` · `tt-metal` · `ttnn` · `block-floating-point` · `bfloat4_b` · `bfloat8_b` · `quantized`

</div>

The main text decoder of [Ornith 1.5 35B A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), with
routed experts in BFP4 GPTQ and other declared projections in BFP8 RTN, packaged by Xylolabs.

This export needs about 21.22 GB of device memory for its language-model weights alone, plus room for the KV
cache and buffers.

## Scope

This complete 40-layer text decoder includes its input embedding and LM head. Vision and MTP weights are
omitted, and drafting is disabled. The text configuration selects the native Transformers causal language
model and caps total prompt plus generation at 4,608 tokens. Selected main-model values are preserved through
the namespace and container conversion.

## Formats

| Tensors | Format |
|---|---|
| 80 routed-expert tensors, 20,480 expert matrices | BFP4 GPTQ with block exponent search; 4.5 bits per weight |
| 311 attention/shared-expert/head matrices | BFP8 round to nearest; 8.5 bits per weight |
| Input embedding and other stored unquantized tensors | BF16 (302 entries) |

Main text weights pack to about 21.22 GB, averaging 4.897 bits per stored weight. The embedding stays BF16;
routed experts use 4.5 bits and other declared projections use 8.5. `quant4.json` records each group. The BF16
source containers are larger than their packed representation.

## Quickstart

Use Transformers 5.23.0 and PyTorch. The files contain BF16 grid values, so the host reference uses more
memory than packed Tensix weights.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "Xylolabs/Ornith-1.5-35B-A3B-Tensix-BFP4-Experts"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype="bfloat16", device_map="auto")
messages = [{"role": "user", "content": "What is the capital of France? Reply with the city name only."}]
inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                 return_dict=True, enable_thinking=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
print(tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Run on Tensix

All 391 declared projection tensors (covering 20,791 expert and dense matrices) pass the grid and native TT-NN
packing checks, including packing as BFP8. The exact example also passes on the Blackhole simulator, checked
with TT-NN 0.81.1. After downloading the repository locally, pack one routed-expert matrix:

```python
import json
import torch
import ttnn
from safetensors import safe_open

d = "Ornith-1.5-35B-A3B-Tensix-BFP4-Experts"  # hf download Xylolabs/Ornith-1.5-35B-A3B-Tensix-BFP4-Experts --local-dir Ornith-1.5-35B-A3B-Tensix-BFP4-Experts
name = "model.layers.0.mlp.experts.down_proj"
shard = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"][name]
with safe_open(f"{d}/{shard}", "pt") as f:
    w = f.get_tensor(name)[0].t().contiguous()
device = ttnn.open_device(device_id=0)
w_tt = ttnn.as_tensor(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT,
                      device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
assert torch.equal(ttnn.to_torch(w_tt).float(), w.float())
ttnn.close_device(device)
```

`python pack_tensix.py Ornith-1.5-35B-A3B-Tensix-BFP4-Experts` checks all declared matrices; `--device` adds a
DRAM round trip. `python verify_grid.py Ornith-1.5-35B-A3B-Tensix-BFP4-Experts` checks the grid using torch.
These checks verify the saved representation.

## Evaluation

| Profile | Mean KL | Normal right / 33 | Thinking right / 7 | Counted losses |
|---|---:|---:|---:|---:|
| Original BF16 reference | 0 | 31 | 7 | — |
| This main text export | 0.058006 | 31 | 7 | 0 |
| Matched Q4_K_M | 0.079161 | 31 | 7 | 0 |

The relative rule-7 gate passes: 0.732760 times Q4_K_M's KL, within the 1.5-times limit, and zero counted
losses. Fresh host KL covers 8,188 positions from English/Korean raw texts, two chunks of 2,048 tokens each.
The original saved BF16 logprob rows and matched Q4_K_M records are retained comparisons; the llama.cpp BF16
floor is 0.008877.

Fresh host 33 + 7 answers use the unchanged original prompt IDs, stops and budgets. Every loaded tensor is
checked against its indexed source at the loaded dtype. The standard thinking budgets fit the 4,608-token cap.
Speed is not reported here.

## License

The pinned Ornith model card declares MIT; its linked license file is absent upstream. `LICENSE` contains the
full standard MIT terms naming the original model authors. Inherited Qwen3.5 components retain Apache 2.0 and
Copyright 2026 Alibaba Cloud; `UPSTREAM_QWEN_LICENSE` is the exact original license. `NOTICE` records both the
inheritance and our text-only changes. Generic Xylolabs helpers use Apache 2.0 (`CODE_LICENSE`).

## Citation

Cite the [original Ornith 1.5 release](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

