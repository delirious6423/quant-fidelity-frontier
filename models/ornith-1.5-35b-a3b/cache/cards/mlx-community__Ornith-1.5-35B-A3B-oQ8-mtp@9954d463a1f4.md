---
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
- mlx
- oQ
- mtp
- speculative-decoding
- apple-silicon
library_name: transformers
---

# Ornith-1.5-35B-A3B oQ8 (MTP)

oMLX oQ8 quantization of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) with the MTP (multi-token prediction) head retained and quantized, so speculative decoding works out of the box.

## Quantization

- oQ format: affine quantization, 8-bit, group size 64, uniform across tensors.
- The MTP block is quantized alongside the base model.
- Vision components retained: the model accepts text, image, and video input.

## Requirements

- Weights are ~36 GB. Runs on 64 GB machines; budget more for long context.
- Loading requires a runtime that supports oQ checkpoints and MTP: oMLX v0.7 or later.

## Measured throughput

Measured on Mac Studio M3 Ultra (512 GB unified memory), oMLX v0.7.3, MTP enabled, server defaults, 256 generated tokens, mean of 3 runs.

| pp | Prefill tok/s | Decode tok/s |
|---|---|---|
| 1024 | 2331.1 | 131.5 |
| 4096 | 3085.8 | 154.3 |
| 32768 | 2290.5 | 157.5 |
| 131072 | 1107.8 | 102.2 |

MoE with ~3B active parameters. Decode rises with context up to pp32768 before dropping at pp131072; prefill falls about 52% past pp32768.

## Usage

Serve with oMLX or any runtime that loads oQ checkpoints, and point an OpenAI-compatible client at it.