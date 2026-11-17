---
license: other
base_model:
  - LiquidAI/LFM2.5-24B-A2B
tags:
  - rco
  - mixed-precision
  - quantized
library_name: mlx
---

# LFM2.5-24B-A2B — RCO 3.0 bpw

Mixed-precision RCO quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).

| | |
|---|---|
| achieved bpw | **2.9975** |
| size | 9.8 GB |
| group size | 64, affine |
| expert allocation | per-projection (114 groups, bits {3:85, 2:29}) |
| scale dtype | bfloat16 |
| calibration | FineWeb-Edu 512x2048, checkpoint's own tokenizer (roundtrip 1.000) |
| imatrix | `hermitdave/LFM2.5-24B-A2B-imatrix` |

## Quality

Recovery = quant_acc / BF16_acc, 5 tasks x 400 examples, identical prompts:

| task | BF16 | quant | recovery |
|---|---|---|---|
| arc_challenge | 0.4400 | 0.3950 | 0.898 |
| arc_easy | 0.7400 | 0.6650 | 0.899 |
| hellaswag | 0.3975 | 0.3925 | 0.987 |
| winogrande | 0.4975 | 0.5000 | 1.005 |
| mmlu | 0.4075 | 0.3350 | 0.822 |
| **mean** | | | **0.9222** |

Mean recovery 0.9222 — usable, but **not** task-lossless. Prefer the 4.5 bpw
build when the extra 4 GB is affordable.

Built with the RCO MLX pipeline. Quantization workflow assisted by [Hermes Agent](https://hermes-agent.nousresearch.com).
