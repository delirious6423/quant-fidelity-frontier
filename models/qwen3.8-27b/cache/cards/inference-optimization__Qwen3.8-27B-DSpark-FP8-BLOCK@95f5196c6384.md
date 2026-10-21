---
library_name: speculators
base_model:
- RedHatAI/Qwen3.8-27B-speculator.dspark
- Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: text-generation
tags:
- speculative-decoding
- dspark
- qwen3
- qwen3.8
- quantized
- fp8
- data-free
---

# Qwen3.8-27B-DSpark-FP8-BLOCK

An FP8 block, data-free DSpark drafter derived from [RedHatAI/Qwen3.8-27B-speculator.dspark](https://huggingface.co/RedHatAI/Qwen3.8-27B-speculator.dspark) at revision `7f33c272e5da240978e0d55767abab8193d74b95`. Pair it with [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. This repository contains a drafter component, not a standalone chat model.

## Quantization

Data-free FP8_BLOCK export; no calibration data was used.

No calibration data was used for this data-free export.
Quantization commands, manifests, source files, patches, and the exported checkpoint checksum are in `provenance/quantization/`.

## Example serving command

```bash
vllm serve Qwen/Qwen3.8-27B \
  --spec-model inference-optimization/Qwen3.8-27B-DSpark-FP8-BLOCK \
  --spec-method dspark \
  --spec-tokens 8
```

This command is an example; this checkpoint has not completed serving validation.

## Evaluation status

Evaluation is pending. No completed acceptance, speed, or quality results are included with this publication. The checkpoint has quantization provenance only; serving/runtime validation and the planned evaluation matrix have not completed.

## Reproducibility

`provenance/quantization/` includes the training and quantization commands, quantization manifest, calibration metadata where applicable, source scripts and patches, and the SHA-256 digest of the published drafter weights. Calibration prompt data is not redistributed.
