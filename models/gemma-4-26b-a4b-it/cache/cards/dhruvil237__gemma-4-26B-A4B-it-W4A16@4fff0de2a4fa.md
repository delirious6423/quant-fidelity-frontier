---
base_model: google/gemma-4-26B-A4B-it
tags:
  - gemma4
  - multimodal
  - image-text-to-text
  - llm-compressor
  - compressed-tensors
  - gptq
  - int4
  - vllm
---

# gemma-4-26B-A4B-it-W4A16

This repository contains a `llm-compressor` GPTQ export of `google/gemma-4-26B-A4B-it` using 4-bit weight-only quantization with bf16 activations.

## What This Is

- Base model: `google/gemma-4-26B-A4B-it`
- Export format: `compressed-tensors` / `pack-quantized`
- Quantization style: W4A16 GPTQ
- Weight quantization: 4-bit signed integer, symmetric, grouped
- Weight group size: `64`
- Modalities: text and image

This checkpoint uses `group_size=64` intentionally. Gemma 4 26B A4B contains MoE `down_proj` widths such as `704` and `2112`, which are not divisible by `128`, so a default W4A16 G128 export is not valid for these layers.

## Calibration Setup

- Text calibration baseline: `HuggingFaceH4/ultrachat_200k`
- Image calibration baseline: `lmms-lab/flickr30k`
- Calibration mode: mixed text/image

## vLLM Serving

### Serve Command (Works after "[Gemma4] Support quantized MoE" commit (3aecdf08b4a896a92e2cbd11c3d5a83d3c09abc1))

```bash
vllm serve dhruvil237/gemma-4-26B-A4B-it-W4A16 \
  --gpu-memory-utilization 0.8 \
  --reasoning-parser gemma4 \
  --dtype float16
```