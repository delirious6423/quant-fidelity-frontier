---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
library_name: mlx
tags:
- mlx
- gemma
- gemma4
- moe
- text-generation
---

# Gemma-4-26B-A4B-it-MLX-8bit

MLX (Apple Silicon) conversion of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (Mixture-of-Experts, ~4B active), quantized to **8-bit · near-lossless**. Text-only build.

## Quantizations

Part of the [**Gemma-4-26B-A4B-it MLX** collection](https://huggingface.co/collections/pipenetwork/gemma-4-26b-a4b-it-mlx-6a2e1fad8514bfa5f0c44576).

| Variant | Notes |
|---|---|
| [4-bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) | 4-bit · community build (mlx-community) |
| **8-bit** (this repo) | 8-bit · near-lossless |
| [6-bit](https://huggingface.co/pipenetwork/Gemma-4-26B-A4B-it-MLX-6bit) | 6-bit · high quality |
| [5-bit](https://huggingface.co/pipenetwork/Gemma-4-26B-A4B-it-MLX-5bit) | 5-bit |

## Use with mlx-lm

```bash
pip install mlx-lm
python -m mlx_lm generate --model pipenetwork/Gemma-4-26B-A4B-it-MLX-8bit --prompt "Explain Mixture-of-Experts briefly." -m 256
```

## Validation
Smoke-tested locally: loads and generates coherent text.

## License
Apache 2.0 (inherited from base). Quantization config: `{"group_size": 64, "bits": 8, "mode": "affine", "language_model.model.layers.0.router.proj": {"group_size": 64, "bits": 8}, "language_model.model.layers.1.router.proj": {"group_size": 64, "bits": 8}, `.
