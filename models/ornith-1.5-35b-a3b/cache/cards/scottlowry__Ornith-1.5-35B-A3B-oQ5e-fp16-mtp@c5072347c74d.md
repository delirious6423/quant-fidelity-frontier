---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
---

# Ornith-1.5-35B-A3B-oQ5e-fp16-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1.dev1) mixed-precision quantization.

## Quantization details

- **Model type**: qwen3_5_moe
- **Bits**: 5
- **Group size**: 64
- **Format**: MLX safetensors