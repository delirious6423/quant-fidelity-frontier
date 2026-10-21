
---
library_name: mlx
tags:
- mlx
- oq
- quantized
license: mit
language:
- zh
- en
- ja
- ko
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
---
![grok_1786759001197.jpg](https://cdn-uploads.huggingface.co/production/uploads/6997328cf288253cb2162c0f/BlPOd7URsIVWylHlJ4PFj.webp)
# Qwen3.8-27B-oQ4-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.5.7) mixed-precision quantization.

jundot is a developer I deeply admire. He said:

oQ: oMLX Universal Dynamic Quantization

Quantization should not be exclusive to any particular inference server. oQ produces standard mlx-lm models that work everywhere — oMLX, mlx-lm, LM Studio, and any app that supports the MLX safetensors format. No custom loader required.

A tribute to him.🍺 

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 4
- **Group size**: 64
- **Format**: MLX safetensors