---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- LiquidAI/LFM2.5-24B-A2B
---

# LFM2.5-24B-A2B-oQ6e

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1.dev1) mixed-precision quantization.

## Quantization details

- **Model type**: lfm2_moe
- **Bits**: 6
- **Group size**: 64
- **Format**: MLX safetensors