---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- 2-bit
- mtp
---

# agnosticeng/Qwen3.8-27B-2bit

2-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), with the Qwen3.8 MTP head bundled at `mtp/` for speculative decoding.

## Quantization

- Affine 2-bit, group size 64 (~2.7 bits/weight, ~8.4 GB)
- Converted with mlx-vlm; vision tower removed (text-only)

## MTP head

`mtp/weights.safetensors` contains the Qwen3.8-27B multi-token-prediction sidecar (2-bit, `mtp.`-prefixed keys).
