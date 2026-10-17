---
quantized_by: tarruda
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
tags:
- gguf
- imatrix
- multimodal
- qwen
- llama.cpp
---

# Qwen3.8 Flash Next GGUF

GGUF quantizations for
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

## Quant Recipes

| Recipe | Quant Size | Default type | Tensor-specific overrides |
| --- | --- | --- | --- |
| `Q5_K` | 150694.87 MiB (7.14 BPW) | `Q8_0` | `ffn_down_exps=q8_0`, `ffn_gate_exps=q5_k`, `ffn_up_exps=q5_k`, `per_layer_token_embd=q8_0` |
| `IQ4_NL` | 97480.69 MiB (4.62 BPW) | `Q8_0` | `ffn_down_exps=iq4_nl`, `ffn_gate_exps=iq4_nl`, `ffn_up_exps=iq4_nl`, `per_layer_token_embd=iq4_nl` |

## Usage

```sh
#!/bin/sh
set -eu

model_dir=${MODEL_DIR:-.}
llama_server=${LLAMA_SERVER:-llama-server}

model=${MODEL:-"$model_dir/IQ4_NL/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf"}
mmproj=${MMPROJ_MODEL:-"$model_dir/Qwen3.8-Flash-Next-mmproj-Q8_0.gguf"}

exec "$llama_server" \
  --load-mode none --no-warmup \
  --model "$model" \
  --mmproj "$mmproj" \
  --ctx-size 262144 -np 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning-preserve \
  --repeat-penalty 1.0 --presence-penalty 0.0 \
  --port 8080
```
