---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen3_8
- awq
- w4a16
- compressed-tensors
- llm-compressor
- Marketing
datasets:
- yothinS/Marketing_Calibration
---

# Qwen3.8-27B Marketing AWQ W4A16

A quantized checkpoint based on `Qwen/Qwen3.8-27B`.

## Quantization

- Method: AWQ activation-aware scaling + GPTQ quantization
- Weight scheme: W4A16
- Calibration dataset: `yothinS/Marketing_Calibration`
- Calibration samples: 32
- Maximum calibration sequence length: 2048
- Vision encoder: kept at original precision

## Important

This checkpoint was quantized using marketing article text for calibration.
Calibration is a post-training compression procedure, not fine-tuning.
It does not by itself teach the model new marketing knowledge or skills.

The output uses the LLM Compressor / compressed-tensors format.
Use a compatible inference runtime and verify model loading and generation
before production deployment.