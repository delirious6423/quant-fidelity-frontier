---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- glm5_next
- mixture-of-experts
- 6-bit
pipeline_tag: text-generation
library_name: mlx
---

# GLM-5.3-Flash-Alis-MLX-6bit

MLX (mlx-vlm tree) 6-bit build of [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(320B-A18B, `glm5_next`), converted by streaming dequant of the official FP8 release (`84c6a6aa`).

**This replaces the withdrawn earlier build** (which quantized the MoE router and decoded at
≈5.5 tok/s). This build keeps the router (`mlp.gate`) and correction bias unquantized/fp32,
the mHC arrays and KDA `A_log`/`dt_bias` in fp32 as stored, and the vision tower in bf16;
the MTP layer (45) is dropped (a standalone MTP drafter is published separately as
`avlp12/GLM-5.3-Flash-Alis-MTP-Drafter`). Conversion receipts (`state.json`,
`finalize_receipt.json`) are in-repo.

## Recipe

6-bit g64 affine on experts, attention, dense/shared MLPs, embeddings and head; router / mHC /
KDA decay params / norms / convs as stored; vision bf16. Per-module map in `config.json`.

## Measured (M3 Ultra 512GB)

| prompt | decode tok/s (oMLX SSE) |
|---|---:|
| p512 | 26.3 |
| p2048 | 25.3 |
| p4096 | 24.6 |

Held-out paired KL vs the 8-bit teacher of this family (en/ko/code, ctx-2048, same harness as
the 4-bit builds): **0.0269 / 0.0226 / 0.0177, weighted mean ≈ 0.0226** — versus 0.0569 for the
4-bit QUASAR build and 0.0622 for 4-bit RTN. Same comparability caveat as the 4-bit card:
quantized-teacher, private panel — not comparable to BF16-teacher KLD numbers elsewhere.
Add `MLX_MAX_MB_PER_BUFFER=2048 MLX_MAX_OPS_PER_BUFFER=100000` to the serving environment for
≈+12% decode on M3 Ultra.
The fused-KDA Metal kernel and long-context gathered-prefill findings documented on the
[4bit card](https://huggingface.co/avlp12/GLM-5.3-Flash-Alis-MLX-4bit) (+20% decode, +50% prefill
at 131k, both bit-identical) are architecture-level and apply to this build too; measured on the
4-bit tree.

For the fastest 4-bit-class build of this family (QUASAR-init, held-out KL −8.5% vs 4-bit RTN),
see `avlp12/GLM-5.3-Flash-Alis-MLX-4bit`.

## Runtime

Requires mlx-vlm with `glm5_next` (merged 2026-08-26); the text-stack correctness fixes
(swiglu clamp, fp32 mHC/KDA aux, eps) live in the PR
[#2044](https://github.com/Blaizzy/mlx-vlm/pull/2044)/[#2074](https://github.com/Blaizzy/mlx-vlm/pull/2074)
branches — recommended until merged.
