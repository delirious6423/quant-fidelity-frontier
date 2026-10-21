---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
---

# gemma-4-26B-A4B-it-oQ6

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.3.6) mixed-precision quantization.

## Quantization details

- **Model type**: gemma4
- **Bits**: 6
- **Group size**: 64
- **Format**: MLX safetensors