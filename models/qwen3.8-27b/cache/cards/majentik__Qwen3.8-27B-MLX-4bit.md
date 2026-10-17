---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
tags:
  - mlx
  - qwen3_5
  - quantized
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-MLX-4bit

MLX **4bit** (affine, group size 32) quantized
variant of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (text tower quantized; vision tower and projector retained in BF16) for Apple
silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Provenance

- Source: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) @ revision
  `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0 (upstream `LICENSE`)).
- Quantized with `mlx_lm.convert` (mlx-lm 0.31.3): affine,
  4-bit, group size 32.

## Smoke gate

Before upload this pack passed a deterministic coherence gate: greedy
48-token chat generation loaded through
`mlx_lm.load`, judged for emptiness, repetition loops, multi-script
gibberish, and special-token debris. Verdict: `ok`.

## Usage

```bash
pip install mlx-lm
mlx_lm.generate --model majentik/Qwen3.8-27B-MLX-4bit --prompt "Hello"
```

## Evaluation

| Benchmark | Score |
| --- | --- |
| arc_easy_acc | 0.5900 |
| hellaswag_acc | 0.5150 |

## License

Apache-2.0 — see the upstream `LICENSE` file in [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## Available tiers

| Repo | Quant |
| --- | --- |
| [majentik/Qwen3.8-27B-MLX-2bit](https://huggingface.co/majentik/Qwen3.8-27B-MLX-2bit) | 2bit |
| [majentik/Qwen3.8-27B-MLX-3bit](https://huggingface.co/majentik/Qwen3.8-27B-MLX-3bit) | 3bit |
| [majentik/Qwen3.8-27B-MLX-4bit](https://huggingface.co/majentik/Qwen3.8-27B-MLX-4bit) | 4bit |
| [majentik/Qwen3.8-27B-MLX-5bit](https://huggingface.co/majentik/Qwen3.8-27B-MLX-5bit) | 5bit |
| [majentik/Qwen3.8-27B-MLX-6bit](https://huggingface.co/majentik/Qwen3.8-27B-MLX-6bit) | 6bit |
| [majentik/Qwen3.8-27B-MLX-8bit](https://huggingface.co/majentik/Qwen3.8-27B-MLX-8bit) | 8bit |
| [majentik/Qwen3.8-27B-MLX-MXFP4](https://huggingface.co/majentik/Qwen3.8-27B-MLX-MXFP4) | MXFP4 |
