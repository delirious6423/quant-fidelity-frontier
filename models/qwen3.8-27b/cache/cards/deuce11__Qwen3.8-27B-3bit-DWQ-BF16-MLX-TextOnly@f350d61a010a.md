---
language: en
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- text-only
- dwq
- 3-bit
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B 3-bit DWQ, text only

MLX checkpoint. Affine 3-bit, group size 64, about 11.8 GB. The vision tower is not in these weights. `config.json` still has `language_model_only: false`. Serve it with Splash `--language-only`.

The packed integers are a round-to-nearest snap of `Qwen/Qwen3.8-27B` bf16. mlx-lm 0.32.2 then learned the scales and biases against that same bf16 model. The calibration set is `allenai/tulu-3-sft-mixture`, 1024 train rows and 32 validation rows from `train[:1056]`, max sequence 512, batch 2, learning rate 1e-6, 512 steps. The saved weights are step 511, which was the best validation step.

Validation loss on those 32 rows:

| step | loss |
|---:|---:|
| 0 | 0.3231 |
| 199 | 0.1053 |
| 399 | 0.0805 |
| 511 | 0.0757 |

On 2026-10-21 the same four chat prompts were run through Splash with thinking off, once on this file and once on `lukaskremla/Qwen3.8-27B-3bit-MLX-TextOnly`. All eight answers ended on a sentence. This file produced 418, 674, 731, and 907 tokens. lukaskremla produced 579, 982, 692, and 740.

An earlier DWQ of this model, distilled from an 8-bit teacher, cut three of those four answers off mid-sentence. That file is `deuce11/Qwen3.8-27B-3bit-DWQ-MLX-TextOnly`.

The base model is Apache 2.0.
