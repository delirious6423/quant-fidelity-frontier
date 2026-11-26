---
license: mit
library_name: mlx
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- mlx
- oq
- 4-bit
- abliterated
- uncensored
- mtp
---

# DeepSeek-V4.1-Flash-1017-oQ4e-mtp-uncensored

An uncensored DeepSeek-V4.1-Flash-1017 for Apple Silicon at 4-bit quality: MLX format, oQ4e mixed precision (4.529 bits/weight), with the multi-token prediction (MTP) head. It is sized for a Mac Studio with 512 GB of memory.

> **Status:** the weights are not uploaded yet. This card describes the build that will be published here.

## Why this build

On 2026-11-24 the uncensored builds of this model we could find were either full size (about 500 GB and more, in GPU formats) or 2-3 bit MLX. This one keeps 4-bit quality and loads in about 403 GB.

## Measured (Mac Studio M3 Ultra 512 GB, oMLX 0.8.3)

| | |
|---|---|
| Memory when loaded | about 403 GB |
| Generation speed | 36-40 tokens/s |
| Long context | a cold 243,819-token prompt was read in 537 s and answered correctly (server limit set to 262,144) |
| JailbreakBench JBB-Behaviors (100 prompts), with a 4,000-token thinking budget set in the runtime | 99 answered, 1 refused, 0 without an answer |
| The same run without a thinking budget | 83 answered, 0 refused, 17 gave no answer within 12,000 tokens (the model kept reasoning) |
| Tool calling, vision | not re-tested on this build |
Refusals are judged by a phrase heuristic on the first 300 characters of each answer (temperature 0). Prompts that ran out of tokens while the model was still reasoning were asked once more with a 12,000-token budget. This is not a judge-model evaluation.

Tip: on prompts where the model reasons for a very long time, set a thinking budget in your runtime (in oMLX: the per-request `thinking_budget` field or the per-model setting).

## Requirements

About 410 GB of free unified memory and an MLX runtime that supports this model family. Tested in oMLX 0.8.3.

## License

MIT, the same as the base model. See `LICENSE`.

## Responsibility

This model answers requests that the base model refuses. You are responsible for how you use it.
