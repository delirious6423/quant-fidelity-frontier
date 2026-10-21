---
library_name: mlx
tags:
- mlx
- oq
- quantized
license: apache-2.0
language:
- en
- zh
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
---

![grok_1786759001197.jpg](https://cdn-uploads.huggingface.co/production/uploads/6997328cf288253cb2162c0f/-w_7nqzipzigQVyD2t5O-.webp)
# Qwen3.8-27B-oQ4

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.3) mixed-precision quantization.

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 4
- **Group size**: 64
- **Format**: MLX safetensors