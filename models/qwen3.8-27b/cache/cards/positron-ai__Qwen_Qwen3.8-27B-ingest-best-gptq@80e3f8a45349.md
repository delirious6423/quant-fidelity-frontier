---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
license: apache-2.0
pipeline_tag: text-generation
tags:
- positron-ai
- quantized
- gptq
---

# Positron AI Quantized Build

This repository contains a Positron AI quantized build of Qwen/Qwen3.8-27B for inference.

## Recommended Use

Use this artifact when you need a GPTQ 4-bit build of Qwen/Qwen3.8-27B built by Positron AI.

For general-purpose GPU inference, compare against the original model and other quantized formats before deployment.

## Artifact Summary

| Field | Value |
|---|---|
| Base model | Qwen/Qwen3.8-27B |
| Published artifact | positron-ai/Qwen_Qwen3.8-27B-ingest-best-gptq |
| Quantization method | GPTQ |
| Quantization format | gptq |
| Source precision | n/a |
| Target runtime | n/a |
| Hardware target | n/a |
| Release date | 2026-10-27 |
| License | apache-2.0 |

## Quantization Details

| Field | Value |
|---|---|
| Weight precision | 4-bit |
| Activation precision | not quantized |
| Bits | 4 |
| Group size | 64 |
| Symmetric quantization | true |
| Activation ordering / desc_act | false |
| Damp percent | 0.05 |
| Calibration dataset | Universal mixed-domain set |
| Calibration samples | 512 |
| Calibration sequence length | 8192 |
| MoE experts per token | n/a |
| Quantization toolchain | GPTQModel 7.6.0, transformers 5.12.1, torch 2.10.0, CUDA 12.8 |

## Evaluation

This card intentionally reports no performance or quality metrics (no
KL-divergence, accuracy, or perplexity figures). Validation results are
tracked internally by Positron AI.

## Provenance

This artifact was produced by Positron AI from Qwen/Qwen3.8-27B. The original model license and usage restrictions continue to apply.
