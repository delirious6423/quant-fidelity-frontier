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

# sady4850/Qwen3.8-27B-4bit-GPTQ-agent

Experimental checkpoint from an agent-calibration study; the summary, test set and all variants are on the card of
`sady4850/Qwen3.8-27B-4bit-GPTQ-agentmix`. Agent traces only: closest to bf16 on agent text, but general text falls back to RTN level.

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) in exactly the layout of
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (MLX affine 4-bit, group size 64), GPTQ as in
[`sady4850/Qwen3.8-27B-4bit-GPTQ`](https://huggingface.co/sady4850/Qwen3.8-27B-4bit-GPTQ), calibrated on 128 chunks of agent traces alone (128 x 2048 tokens).

| vs `sady4850/Qwen3.8-27B-4bit-GPTQ` | value (95% CI over 49 tasks) |
|---|---|
| agent reasoning KLD | -41% [-44%, -39%] (0.0153) |
| agent tool-call KLD | -41% [-45%, -35%] (0.0035) |
| same call as bf16 | +7.9 [+4.0, +11.7] points (62.3%) |
| wiki KLD (torch) | +54% (0.0566) |
| chat KLD (torch) | +54% (0.1521) |

Raw numbers in `eval/`.
