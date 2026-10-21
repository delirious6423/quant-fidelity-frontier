---
license: apache-2.0
base_model: [google/gemma-4-26B-A4B-it]
library_name: mlx
pipeline_tag: image-text-to-text
language: [en, fr]
tags: [mlx, mlx-vlm, vlm, gemma4, gemma, moe, apple-silicon, quantized]
---

# Gemma 4 26B-A4B Instruct (MoE) — 4-bit MLX

4-bit [MLX](https://github.com/ml-explore/mlx) quantization of
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it), for Apple Silicon (~14 GB).
Sparse-MoE vision-language model — run it with `mlx-vlm`, not `mlx-lm`. The expert router (`router.proj`) is kept at 8-bit precision.

## Usage

```bash
pip install -U mlx-vlm
```
```bash
python -m mlx_vlm.generate \
  --model TyKaoz/gemma-4-26B-A4B-it-4bit \
  --prompt "Explique la quantization en une phrase." \
  --max-tokens 200
```

| Base | Tool | Precision | Size |
|------|------|-----------|------|
| `google/gemma-4-26B-A4B-it` | `mlx-vlm` | 4-bit · group 64 (router 8-bit) | ~14 GB |

By **[TyKaoz](https://www.tykaoz.bzh)** — privacy-first native macOS LLM chat client. Apache 2.0, inherited from the base model.
