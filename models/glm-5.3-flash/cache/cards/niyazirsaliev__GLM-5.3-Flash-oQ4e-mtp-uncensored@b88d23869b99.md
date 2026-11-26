---
license: mit
library_name: mlx
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- 4-bit
- abliterated
- uncensored
- mtp
---

# GLM-5.3-Flash-oQ4e-mtp-uncensored

An uncensored GLM-5.3-Flash for Apple Silicon at 4-bit quality: MLX format, oQ4e mixed precision (4.641 bits/weight), with the vision encoder and the multi-token prediction (MTP) head. It fits a Mac with 256 GB of memory.

> **Status:** the weights are not uploaded yet. This card describes the build that will be published here.

## Measured (Mac Studio M5 Ultra 256 GB, oMLX 0.7.3rc1)

| | |
|---|---|
| Size on disk | about 187 GB |
| Memory when loaded | about 196 GB |
| Generation speed | 70-95 tokens/s |
| JailbreakBench JBB-Behaviors (100 prompts) | 99 answered, 1 refused, 0 without an answer |
| Same check on the standard 4-bit build (10 prompts of our own) | standard build refused 8 of 10; this build refused 0 |
| Tool calling, vision, long context | not re-tested on this build |

Refusals are judged by a phrase heuristic on the first 300 characters of each answer (temperature 0). Prompts that ran out of tokens while the model was still reasoning were asked once more with a 12,000-token budget. This is not a judge-model evaluation.

## Requirements

About 200 GB of free unified memory and an MLX runtime that supports this model family. Tested in oMLX 0.7.3rc1.

## License

MIT, the same as the base model. See `LICENSE`.

## Responsibility

This model answers requests that the base model refuses. You are responsible for how you use it.
