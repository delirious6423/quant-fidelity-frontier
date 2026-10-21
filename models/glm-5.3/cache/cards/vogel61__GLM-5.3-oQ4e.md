---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- zai-org/GLM-5.3
license: other
license_name: glm-5.3
pipeline_tag: text-generation
---

# GLM-5.3-oQ4e

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.4) mixed-precision quantization.

## Quantization details

- **Model type**: glm_moe_dsa
- **Bits**: 4
- **Group size**: 64
- **Format**: MLX safetensors