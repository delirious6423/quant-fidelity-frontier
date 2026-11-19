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

# LFM2.5-24B-A2B — RCO 4.5 bpw

Mixed-precision RCO quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B).

| | |
|---|---|
| achieved bpw | **4.4795** |
| size | 14 GB |
| group size | 64, affine |
| expert allocation | per-projection (114 groups, bits {4:101, 6:13}) |
| scale dtype | bfloat16 |
| calibration | FineWeb-Edu 512x2048, checkpoint's own tokenizer (roundtrip 1.000) |
| imatrix | `hermitdave/LFM2.5-24B-A2B-imatrix` (7 440 tensors, per-expert) |

## Quality — task-lossless

Recovery = quant_acc / BF16_acc, 5 tasks x 400 examples, identical prompts:

| task | BF16 | quant | recovery |
|---|---|---|---|
| arc_challenge | 0.4400 | 0.4400 | 1.000 |
| arc_easy | 0.7400 | 0.7050 | 0.953 |
| hellaswag | 0.3975 | 0.3875 | 0.975 |
| winogrande | 0.4975 | 0.5175 | 1.040 |
| mmlu | 0.4075 | 0.4025 | 0.988 |
| **mean** | | | **0.9911** |

**Mean recovery 0.9911 — task-lossless.**

## Teacher validation (KL / top-1 / PPL)

WikiText-2 test, 8 rows x 512 tokens, against the **resident** BF16 checkpoint —
at 45 GB it loads on this machine, so no streaming teacher was needed. Teacher
PPL on those rows is **23.17** (BOS-prefixed; LFM2.5's tokenizer defines
`bos_token_id` 124894).

| metric | value |
|---|---|
| forward KL(BF16 ‖ quant) ↓ | **0.2304** |
| top-1 agreement ↑ | **77.6 %** |
| quant PPL ↓ | **22.75** |

## Throughput

`bench_tps_local.py` — 2 warmups + 3 measured runs, 128 decode tokens, mlx-lm:

| | decode tok/s |
|---|---|
| RCO-4.5 | **120.5** (120.1–120.8) |

## Notes

- The per-expert imatrix weighting engages directly here: LFM2.5 stores its
  routed experts unfused (`feed_forward.experts.<id>.w1/w2/w3`), so the capture
  resolves per expert.
- Calibration was verified against the checkpoint's own tokenizer; reusing a
  cache built for another vocabulary silently measures off-vocabulary
  activations.

Built with the RCO MLX pipeline. Quantization workflow assisted by [Hermes Agent](https://hermes-agent.nousresearch.com).
