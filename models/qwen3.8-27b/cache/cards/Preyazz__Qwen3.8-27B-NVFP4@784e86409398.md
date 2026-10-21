---
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- nvfp4
- compressed-tensors
- vllm
- qwen3_5
- vision-language
---

# Qwen3.8-27B-NVFP4

NVFP4 (4-bit, `compressed-tensors` `nvfp4-pack-quantized`) weight quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for fast inference on
NVIDIA Blackwell (sm_120) with vLLM.

## Details

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Dense 27B, native vision-language, Gated-DeltaNet hybrid with MTP.
- **Method:** weight-only NVFP4, group size 16 (`compressed-tensors`).
- **Target runtime:** vLLM on Blackwell-class GPUs.
- **Relation to base:** quantization only. No weights were trained or fine-tuned.

## Usage

```bash
vllm serve Preyazz/Qwen3.8-27B-NVFP4 --trust-remote-code
```

## Provenance and license

This repository redistributes a quantized copy of `Qwen/Qwen3.8-27B`. All model
capabilities, credit, and the governing license belong to the Qwen team, and the base
model's license applies to this quantized derivative. See the original model card for
full model documentation, intended use, and limitations.

Provided as-is, without warranty.
