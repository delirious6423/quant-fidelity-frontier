---
language:
- en
license: apache-2.0
tags:
- gemma4
- compressed-tensors
- quantized
- 4-bit
- W4A16
- moe
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
---

# gemma-4-26B-A4B-it · INT4 W4A16 (channel-wise)

4-bit weight-only quantized version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

## Quantization Details

| Property | Value |
|---|---|
| Scheme | W4A16 (INT4 weights, BF16 activations) |
| Strategy | Channel-wise (group_size = -1) |
| Tool | llm-compressor (git main) |
| Calibration | HuggingFaceH4/ultrachat_200k · 512 samples |
| Base model | google/gemma-4-26B-A4B-it |

## Why channel-wise?

Gemma 4 MoE FFN dims are 2112 and 704 — not divisible by 128.
Marlin (the standard A100 W4A16 kernel) requires all dims % 128 == 0.
Channel-wise (group_size=-1) is the only strategy compatible with A100 (SM80).

## vLLM Usage

```bash
vllm serve lokeshe09/gemma-4-26B-A4B-it-INT4-W4A16-channelwise --max-model-len 32768 --gpu-memory-utilization 0.95
```
