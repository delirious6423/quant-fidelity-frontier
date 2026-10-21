---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- Qwen/Qwen3.8-27B
- swdjojo/Qwen3.8-27B-MLX-BF16
pipeline_tag: image-text-to-text
---

> [!IMPORTANT]
> This quantization was uploaded on **2026-08-15** and replaces a previous version.
> If you downloaded this model before this date, please re-download for the updated weights.

# Qwen3.8-27B-MLX-oQ8e

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.0.dev1) mixed-precision quantization.

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 8
- **Group size**: 64
- **Format**: MLX safetensors