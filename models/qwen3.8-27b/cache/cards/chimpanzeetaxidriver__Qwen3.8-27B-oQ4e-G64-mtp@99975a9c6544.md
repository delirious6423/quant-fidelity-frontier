---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
tags:
- mlx
- oq
- quantized
license: apache-2.0
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-oQ4e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.0.dev1) mixed-precision quantization.

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 4
- **Group size**: **64**
- **Format**: MLX safetensors
