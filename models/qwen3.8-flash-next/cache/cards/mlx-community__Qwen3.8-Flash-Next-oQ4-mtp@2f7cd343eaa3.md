---
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
- oQ
- mtp
- speculative-decoding
- apple-silicon
library_name: transformers
---

# Qwen3.8-Flash-Next oQ4 (MTP)

oMLX oQ4 quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) with the MTP (multi-token prediction) head retained and quantized, so speculative decoding works out of the box.

## Quantization

- oQ format: affine quantization, group size 64, mixed bits (4/5/6/8) assigned per tensor by oMLX's oQ4 profile.
- The MTP block is quantized alongside the base model.
- Vision components retained: the model accepts text and image input.

## Requirements

- Weights are ~99 GB. Budget ~128 GB of unified memory for weights plus KV cache at moderate context; more for long context.
- Context: 262,144 tokens (config `max_position_embeddings`).
- Loading requires a runtime that supports oQ checkpoints and MTP: oMLX v0.7 or later.

## Measured throughput

Measured on Mac Studio M3 Ultra (512 GB unified memory), oMLX v0.7.3, MTP enabled, server defaults, 256 generated tokens, mean of 3 runs.

| pp | Prefill tok/s | Decode tok/s |
|---|---|---|
| 1024 | 1125.9 | 103.6 |
| 4096 | 1339.0 | 88.1 |
| 32768 | 1358.2 | 92.1 |
| 131072 | 1317.5 | 58.2 |

Prefill stays roughly flat from pp4096 to pp131072. Decode falls as context grows.

## Usage

Serve with oMLX or any runtime that loads oQ checkpoints, and point an OpenAI-compatible client at it.