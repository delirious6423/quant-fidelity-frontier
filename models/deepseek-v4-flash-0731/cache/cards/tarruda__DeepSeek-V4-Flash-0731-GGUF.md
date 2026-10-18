---
quantized_by: tarruda
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
license: mit
license_link: https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/mit.md
tags:
- gguf
- imatrix
- conversational
- deepseek
- llama.cpp
---

# DeepSeek V4 Flash 0731 GGUF

GGUF quantizations for
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

DeepSeek published the original model weights in MXFP4, so the `MXFP4` GGUFs in
this repo are direct conversions of those original safetensors.

## Quant Recipes

| Recipe | Quant Size | Default type | Tensor-specific overrides |
| --- | --- | --- | --- |
| `Q3_K` | 120974.36 MiB (3.57 BPW) | `Q8_0` | `ffn_down_exps=q3_k`, `ffn_gate_exps=q3_k`, `ffn_up_exps=q3_k` |
| `IQ3_XXS` | 108590.36 MiB (3.20 BPW) | `Q8_0` | `ffn_down_exps=iq3_xxs`, `ffn_gate_exps=iq3_xxs`, `ffn_up_exps=iq3_xxs` |
| `Q2_K` | 94142.36 MiB (2.78 BPW) | `Q8_0` | `ffn_down_exps=q2_k`, `ffn_gate_exps=q2_k`, `ffn_up_exps=q2_k` |

## Usage

This is the script I use to run:

```bash
#!/bin/sh -e

model="./IQ3_XXS/DeepSeek-V4-Flash-0731-IQ3_XXS-00001-of-00004.gguf"

ctx=262144
parallel=2

ctx_size=$((ctx * parallel))

llama-server --no-mmap --no-warmup \
  --model $model --ctx-size $ctx_size -np $parallel \
  --repeat-penalty 1.0  --presence-penalty 0.0 \
  --chat-template-kwargs '{"reasoning_effort": "max"}' \
  --reasoning-preserve \
  -b 2048 -ub 2048 \
  --cache-ram 4096 -ctxcp 128 \
  --temp 1.0 --top-p 1.0
```
