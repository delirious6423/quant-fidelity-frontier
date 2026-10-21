---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
---

[![Open in MLXHub](https://mlxhub.app/assets/badge-open-in-mlxhub.svg)](https://mlxhub.app/open-model?repo=DreamFoundries/Qwen3.8-27B-4bit)

# Qwen3.8-27B 4-bit MLX

MLX conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized with `mlx-lm 0.31.3` using affine 4-bit weights and group size 64 (4.501 effective bits per weight). The safetensors weights occupy approximately 14 GB.

Comparative quality and performance benchmarks are not available for this conversion.

## Use with mlx

```python
from mlx_lm import load, generate

model, tokenizer = load("DreamFoundries/Qwen3.8-27B-4bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)
```

---

[![Download MLXHub](https://mlxhub.app/assets/badge-download-mlxhub.svg)](https://apps.apple.com/app/apple-store/id6766485144?pt=121945436&ct=HuggingFace&mt=8)
