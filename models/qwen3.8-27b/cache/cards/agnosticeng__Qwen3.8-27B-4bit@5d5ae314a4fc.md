---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- 4-bit
- mtp
---

# agnosticeng/Qwen3.8-27B-4bit

4-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), with the Qwen3.8 MTP head bundled at `mtp/` for speculative decoding.

## Quantization

- Affine 4-bit, group size 64 (~4.7 bits/weight, ~15 GB)
- Converted with mlx-vlm; vision tower included (image-text-to-text)

## MTP head

`mtp/weights.safetensors` contains the Qwen3.8-27B multi-token-prediction sidecar (4-bit, `mtp.`-prefixed keys).
