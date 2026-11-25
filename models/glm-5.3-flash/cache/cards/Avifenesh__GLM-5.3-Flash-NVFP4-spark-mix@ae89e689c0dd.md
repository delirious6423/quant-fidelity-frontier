---
license: mit
base_model:
  - zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: Avifenesh
library_name: safetensors
tags:
  - nvfp4
  - mxfp8
  - fp8
  - modelopt
  - mixed-precision
  - dgx-spark
  - gb10
---

# GLM-5.3-Flash NVFP4, DGX Spark mix

A mixed-precision variant of [Avifenesh/GLM-5.3-Flash-NVFP4](https://huggingface.co/Avifenesh/GLM-5.3-Flash-NVFP4) (revision `2fa9fe62`), built in October 2026 for serving GLM-5.3-Flash at tensor parallel 2 on two DGX Sparks (GB10). It was the default checkpoint of that setup.

Changes from the base mint:

- The KDA linear-attention projections (`in_proj_qkvbfg_a`, `q/k/v_proj`, `b_proj`, `f_a_proj`, `g_a_proj`) of the 34 KDA layers are NVFP4 (W4A16, group 16). The base mint keeps every KDA tensor in BF16.
- `o_proj` of the 34 KDA layers is MXFP8 (`model-mix-kda-mxfp8.safetensors`).
- The shared experts, the 3 dense MLPs, and `o_proj` / `q_b_proj` of the 12 MLA layers use the vendor's FP8 block-128 weights (`model-mix-vendor-fp8.safetensors`, from zai-org's FP8 release).
- The routed experts stay NVFP4 as in the base mint. `remint-layers.txt` lists the layers that were re-minted.

Totals from `config.json`: 281 NVFP4, 208 FP8 block, 34 MXFP8 quantized modules, 390 kept in BF16. The quantization config is ModelOpt `MIXED_PRECISION` with a per-module `quant_algo`.

On the two Sparks it gave decode steps about 5-6% shorter and +2-4% tokens/s against the base mint (measured 2026-10). No separate accuracy gate was run on this mix beyond the base mint's.

The chat templates, tokenizer and generation config are the base mint's.
