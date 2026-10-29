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

# sady4850/Qwen3.8-27B-4bit-GPTQ-agentmix20

Experimental checkpoint from an agent-calibration study; the summary, test set and all variants are on the card of
`sady4850/Qwen3.8-27B-4bit-GPTQ-agentmix`. Smaller agent share than `-agentmix`.

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) in exactly the layout of
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (MLX affine 4-bit, group size 64), GPTQ as in
[`sady4850/Qwen3.8-27B-4bit-GPTQ`](https://huggingface.co/sady4850/Qwen3.8-27B-4bit-GPTQ), calibrated on 50 chunks of UltraChat train, 50 of calibration_v5 and 20 of agent traces (120 x 2048 tokens).

| vs `sady4850/Qwen3.8-27B-4bit-GPTQ` | value (95% CI over 49 tasks) |
|---|---|
| agent reasoning KLD | -24% [-26%, -22%] (0.0197) |
| agent tool-call KLD | -21% [-26%, -15%] (0.0047) |
| same call as bf16 | +1.8 [-2.2, +5.6] points (56.2%) |
| wiki KLD (torch) | +8% (0.0397) |
| chat KLD (torch) | +1% (0.0999) |

Raw numbers in `eval/`.
