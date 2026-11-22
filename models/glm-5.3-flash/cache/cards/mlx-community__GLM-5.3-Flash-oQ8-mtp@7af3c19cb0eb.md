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

# GLM-5.3-Flash oQ8 (MTP)

oMLX oQ8 quantization of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) with the MTP (multi-token prediction) head retained and quantized, so speculative decoding works out of the box.

## Quantization

- oQ format: affine quantization, 8-bit, group size 64.
- The MTP block is quantized alongside the base model.
- Vision components retained: the model accepts text and image input.

## Requirements

- Weights are ~319 GB. Requires a 512 GB unified-memory machine (M3 Ultra class) with room left for KV cache.
- Context: 1,048,576 tokens (config `max_position_embeddings`).
- Loading requires a runtime that supports oQ checkpoints and MTP: oMLX v0.7 or later.

## Throughput

Not yet benchmarked at oQ8. Numbers for this quant will be added when measured.

## Usage

Serve with oMLX or any runtime that loads oQ checkpoints, and point an OpenAI-compatible client at it.