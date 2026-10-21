---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- glm5_next
- mixture-of-experts
- 8-bit
pipeline_tag: text-generation
library_name: mlx
---

# GLM-5.3-Flash-Alis-MLX-8bit

MLX (mlx-vlm tree) 8-bit build of [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
(320B-A18B, `glm5_next`), converted by streaming dequant of the official FP8 release (`84c6a6aa`).

**This replaces the withdrawn earlier build** (which quantized the MoE router). This build keeps
the router (`mlp.gate`) and correction bias unquantized/fp32, the mHC arrays and KDA
`A_log`/`dt_bias` in fp32 as stored, and the vision tower in bf16; the MTP layer (45) is dropped
(standalone drafter: `avlp12/GLM-5.3-Flash-Alis-MTP-Drafter`). Conversion receipts (`state.json`,
`finalize_receipt.json`) are in-repo. 334.1 GB on disk.

## Recipe

8-bit g64 affine on experts, attention, dense/shared MLPs, embeddings and head; router / mHC /
KDA decay params / norms / convs as stored; vision bf16. Per-module map in `config.json`.

## Role in the family

This is the highest-fidelity tier of the family and the **teacher/reference class** used to score
the 4-bit builds (held-out paired KL): 4-bit QUASAR `avlp12/GLM-5.3-Flash-Alis-MLX-4bit` measures
−8.5% KL vs its 4-bit RTN baseline against an 8-bit teacher of this layout family (6-bit measures
≈0.0226 mean KL on the same panel). Expect ≈24-25 tok/s decode @ p512 on an M3 Ultra 512GB
(8-bit class), vs ≈29-33 tok/s for the 4-bit build; add `MLX_MAX_MB_PER_BUFFER=2048
MLX_MAX_OPS_PER_BUFFER=100000` to the serving environment for ≈+12% decode on M3 Ultra.
The fused-KDA Metal kernel and long-context gathered-prefill findings documented on the
[4bit card](https://huggingface.co/avlp12/GLM-5.3-Flash-Alis-MLX-4bit) (+20% decode, +50% prefill
at 131k, both bit-identical) are architecture-level and apply to this build too; measured on the
4-bit tree.

## Runtime

Requires mlx-vlm with `glm5_next` (merged 2026-08-26); text-stack correctness fixes live in the
PR [#2044](https://github.com/Blaizzy/mlx-vlm/pull/2044)/[#2074](https://github.com/Blaizzy/mlx-vlm/pull/2074)
branches — recommended until merged.
