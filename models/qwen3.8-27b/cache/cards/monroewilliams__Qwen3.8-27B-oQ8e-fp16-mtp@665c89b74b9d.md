---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-oQ8e-fp16-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.0.dev1) mixed-precision quantization.

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 8
- **Group size**: 64
- **Format**: MLX safetensors