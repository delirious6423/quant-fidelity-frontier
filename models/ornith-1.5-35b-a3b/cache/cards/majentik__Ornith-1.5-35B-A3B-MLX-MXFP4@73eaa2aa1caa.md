---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: mlx
tags:
- mlx
- qwen3_5_moe
- quantized
pipeline_tag: image-text-to-text
---

# Ornith-1.5-35B-A3B-MLX-MXFP4

MLX **MXFP4** (mxfp4, group size 32) quantized
variant of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (text tower quantized; vision tower and projector retained in BF16) for Apple
silicon via [mlx-lm](https://github.com/ml-explore/mlx-lm).

## Provenance

- Source: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) @ revision
  `e4dfb35a93d4b6822a811a7676f3488514abe7e2` (MIT (upstream `LICENSE`)).
- Quantized with `mlx_lm.convert` (mlx-lm 0.31.3): mxfp4,
  4-bit, group size 32.

## Smoke gate

Before upload this pack passed a deterministic coherence gate: greedy
48-token chat generation loaded through
`mlx_lm.load`, judged for emptiness, repetition loops, multi-script
gibberish, and special-token debris. Verdict: `ok`.

## Usage

```bash
pip install mlx-lm
mlx_lm.generate --model majentik/Ornith-1.5-35B-A3B-MLX-MXFP4 --prompt "Hello"
```

## Evaluation

| Benchmark | Score |
| --- | --- |
| arc_easy_acc | 0.5650 |
| hellaswag_acc | 0.5350 |
## First-party tiers

ornith-ai publishes first-party 4bit/6bit/8bit MLX and GGUF packs for this
model. The tiers in this family (2bit, 3bit, 5bit, MXFP4) complement those
first-party releases rather than duplicating them.

## Available tiers

| Repo | Quant |
| --- | --- |
| [majentik/Ornith-1.5-35B-A3B-MLX-2bit](https://huggingface.co/majentik/Ornith-1.5-35B-A3B-MLX-2bit) | 2bit |
| [majentik/Ornith-1.5-35B-A3B-MLX-3bit](https://huggingface.co/majentik/Ornith-1.5-35B-A3B-MLX-3bit) | 3bit |
| [majentik/Ornith-1.5-35B-A3B-MLX-5bit](https://huggingface.co/majentik/Ornith-1.5-35B-A3B-MLX-5bit) | 5bit |
| [majentik/Ornith-1.5-35B-A3B-MLX-MXFP4](https://huggingface.co/majentik/Ornith-1.5-35B-A3B-MLX-MXFP4) | MXFP4 |
