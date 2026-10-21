---
library_name: mlx
tags:
- mlx
- oq
- quantized
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- froggeric/Qwen-Fixed-Chat-Templates
---

# Qwen3.8-27B-oQ4e-fp16-mtp-Fixed-Chat-Templates

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.0) mixed-precision quantization.

Original model: https://huggingface.co/Qwen/Qwen3.8-27B

After quantization, the model as added with https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates


## Quantization details

- **Model type**: qwen3_5
- **Bits**: 4
- **Group size**: 64
- **Format**: MLX safetensors