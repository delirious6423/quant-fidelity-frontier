---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - mlx-vlm
  - quantized
  - vision-language
  - qwen3.5
  - apple-silicon
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-VL-mlx-4bit

A 4-bit MLX build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) **with the vision tower preserved**, converted with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for Apple Silicon.

## Vision vs. speculative decode — this is the vision-capable variant

Qwen3.8-27B ships with a vision encoder. The [MTPLX text-only builds](https://huggingface.co/johninthepool/Qwen3.8-27B-MTPLX-4bit) in this account are converted with `mtplx forge`, which preserves the model's multi-token-prediction (MTP) head for native speculative decoding — but drops the vision tower, since mtplx/mlx-lm's conversion path is text-only.

This repo is the inverse tradeoff: converted with `mlx-vlm` instead, which keeps the vision tower intact so the model can actually process images and video, but does **not** carry mtplx's MTP contract — no speculative decoding here.

| Variant | Vision | MTP / spec-decode |
|---|---|---|
| [MTPLX-4bit](https://huggingface.co/johninthepool/Qwen3.8-27B-MTPLX-4bit) | No | Yes |
| [MTPLX-8bit](https://huggingface.co/johninthepool/Qwen3.8-27B-MTPLX-8bit) | No | Yes |
| [MTPLX-bf16](https://huggingface.co/johninthepool/Qwen3.8-27B-MTPLX-bf16) | No | Yes |
| **VL-mlx-4bit (this repo)** | **Yes** | No |

## Quantization

| Parameter | Value |
|---|---|
| Precision | 4-bit (4.695 effective bits/weight) |
| Group size | 64 |
| Source | `Qwen/Qwen3.8-27B` (bf16 native) |
| Toolchain | mlx-vlm (not mtplx) |

## Requirements

- Apple Silicon Mac (M-series)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm): `pip install -U mlx-vlm`

## Usage

```bash
python -m mlx_vlm generate --model johninthepool/Qwen3.8-27B-VL-mlx-4bit \
  --image path/to/image.jpg --prompt "Describe this image."
```

## Provenance

Converted directly from `Qwen/Qwen3.8-27B` with no fine-tuning or distillation — a direct 4-bit affine quantization of the release weights, vision tower included.
