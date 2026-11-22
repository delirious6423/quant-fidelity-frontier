---
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- oQ
- mtp
- speculative-decoding
- apple-silicon
library_name: transformers
---

# Qwen3.8-27B oQ8 (MTP)

oMLX oQ8 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the MTP (multi-token prediction) head retained, so speculative decoding works out of the box.

## Quantization

- oQ format: affine quantization, 8-bit, group size 64.
- Vision components retained: the model accepts text, image, and video input.

## Requirements

- Weights are ~28 GB. Runs on 32 GB machines with modest context; 64 GB recommended for long context.
- Context: 262,144 tokens (config `max_position_embeddings`).
- Loading requires a runtime that supports oQ checkpoints and MTP: oMLX v0.7 or later.

## Measured throughput

Measured on Mac Studio M3 Ultra (512 GB unified memory), oMLX v0.7.3, MTP enabled, server defaults, 256 generated tokens, mean of 3 runs.

| pp | Prefill tok/s | Decode tok/s |
|---|---|---|
| 1024 | 410.4 | 59.6 |
| 4096 | 435.3 | 50.1 |
| 32768 | 385.0 | 70.6 |
| 131072 | 270.5 | 42.9 |

Dense model. Decode is non-monotonic with context length; the pp4096 dip reproduced across all 3 runs. Numbers are highly repeatable (prefill CV <= 0.6%).

## Usage

Serve with oMLX or any runtime that loads oQ checkpoints, and point an OpenAI-compatible client at it.