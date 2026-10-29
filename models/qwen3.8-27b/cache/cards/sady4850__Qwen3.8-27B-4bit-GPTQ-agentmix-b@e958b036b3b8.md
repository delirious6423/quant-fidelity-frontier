---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- gptq
- 4-bit
- agent
---

# sady4850/Qwen3.8-27B-4bit-GPTQ-agentmix-b

Experimental checkpoint from an agent-calibration study; the summary, test set and all variants are on the card of
`sady4850/Qwen3.8-27B-4bit-GPTQ-agentmix`. Calibration-noise control for `-agentmix`.

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) in exactly the layout of
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (MLX affine 4-bit, group size 64), GPTQ as in
[`sady4850/Qwen3.8-27B-4bit-GPTQ`](https://huggingface.co/sady4850/Qwen3.8-27B-4bit-GPTQ), calibrated on the same mix as `-agentmix` with other agent chunks (128-167 of the pool instead of 0-39), to measure how much the result depends on which agent chunks are drawn.

| vs `sady4850/Qwen3.8-27B-4bit-GPTQ` | value (95% CI over 49 tasks) |
|---|---|
| agent reasoning KLD | -31% [-33%, -29%] (0.0180) |
| agent tool-call KLD | -29% [-33%, -25%] (0.0042) |
| same call as bf16 | +5.5 [+1.2, +9.8] points (59.9%) |
| wiki KLD (torch) | +5% (0.0387) |
| chat KLD (torch) | -1% (0.0978) |

Raw numbers in `eval/`.
