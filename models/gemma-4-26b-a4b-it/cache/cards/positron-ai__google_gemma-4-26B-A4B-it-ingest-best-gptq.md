---
base_model: google/gemma-4-26B-A4B-it
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
tags:
- positron-ai
- quantized
- gptq
---

# Positron AI Quantized Build

This repository contains a Positron AI quantized build of google/gemma-4-26B-A4B-it for inference.

## Recommended Use

Use this artifact when you need a GPTQ 4-bit build of google/gemma-4-26B-A4B-it built by Positron AI.

For general-purpose GPU inference, compare against the original model and other quantized formats before deployment.

## Artifact Summary

| Field | Value |
|---|---|
| Base model | google/gemma-4-26B-A4B-it |
| Published artifact | positron-ai/google_gemma-4-26B-A4B-it-ingest-best-gptq |
| Quantization method | GPTQ |
| Quantization format | gptq |
| Source precision | n/a |
| Target runtime | n/a |
| Hardware target | n/a |
| Release date | 2026-09-01 |
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
| Calibration dataset | Mixed-domain calibration set |
| Calibration samples | 128 |
| Calibration sequence length | 4096 |
| MoE experts per token | n/a |
| Quantization toolchain | GPTQModel 7.2.0, transformers 5.11.0, torch 2.9.1, CUDA 12.8 |

## Evaluation

This card intentionally reports no performance or quality metrics (no
KL-divergence, accuracy, or perplexity figures). Validation results are
tracked internally by Positron AI.

## Provenance

This artifact was produced by Positron AI from google/gemma-4-26B-A4B-it. The original model license and usage restrictions continue to apply.
