---
base_model: zai-org/GLM-5.3-Flash
tags:
- mlx
- oQ
- mtp
- speculative-decoding
- apple-silicon
library_name: transformers
---

# GLM-5.3-Flash oQ6 (MTP)

oMLX oQ6 quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) with the MTP (multi-token prediction) head retained and quantized, so speculative decoding works out of the box.

## Quantization

- oQ format: affine quantization, group size 64, mixed bits assigned per tensor by oMLX's oQ6 profile.
- The MTP block is quantized alongside the base model.
- Vision components retained: the model accepts text and image input.

## Requirements

- Weights are ~246 GB. Budget a 256 GB unified-memory machine minimum; 512 GB recommended for long context plus KV cache.
- Context: 1,048,576 tokens (config `max_position_embeddings`).
- Loading requires a runtime that supports oQ checkpoints and MTP: oMLX v0.7 or later.

## Measured throughput

Measured on Mac Studio M3 Ultra (512 GB unified memory), oMLX v0.7.3, MTP enabled, server defaults, 256 generated tokens, mean of 3 runs.

| pp | Prefill tok/s | Decode tok/s |
|---|---|---|
| 1024 | 486.0 | 41.4 |
| 4096 | 560.1 | 43.2 |
| 32768 | 521.6 | 50.7 |
| 131072 | 505.3 | 70.6 |

Decode rises with context: 41.4 tok/s at pp1024 to 70.6 tok/s at pp131072. Prefill stays flat across the whole range (486 to 505 tok/s, peak 560 at pp4096). Long context costs TTFT: about 259 s at pp131072.

## Usage

Serve with oMLX or any runtime that loads oQ checkpoints, and point an OpenAI-compatible client at it.