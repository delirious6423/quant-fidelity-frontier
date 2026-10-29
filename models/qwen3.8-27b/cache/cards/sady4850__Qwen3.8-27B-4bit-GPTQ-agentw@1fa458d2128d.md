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

# sady4850/Qwen3.8-27B-4bit-GPTQ-agentw

Experimental checkpoint from an agent-calibration study; the summary, test set and all variants are on the card of
`sady4850/Qwen3.8-27B-4bit-GPTQ-agentmix`. Agent traces only, with token weighting.

[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) in exactly the layout of
[`mlx-community/Qwen3.8-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (MLX affine 4-bit, group size 64), GPTQ as in
[`sady4850/Qwen3.8-27B-4bit-GPTQ`](https://huggingface.co/sady4850/Qwen3.8-27B-4bit-GPTQ), calibrated on the same 128 agent chunks as `-agent`, with Hessian rows of prompt and tool-output tokens weighted 0.25 (reasoning, assistant text and tool calls 1).

| vs `sady4850/Qwen3.8-27B-4bit-GPTQ` | value (95% CI over 49 tasks) |
|---|---|
| agent reasoning KLD | -44% [-46%, -41%] (0.0146) |
| agent tool-call KLD | -42% [-47%, -36%] (0.0035) |
| same call as bf16 | +7.7 [+3.1, +12.1] points (62.1%) |
| wiki KLD (torch) | +48% (0.0541) |
| chat KLD (torch) | +66% (0.1637) |

Raw numbers in `eval/`.
