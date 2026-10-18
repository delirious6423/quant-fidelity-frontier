---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- glm5_next
- speculative-decoding
- draft-model
library_name: mlx
---

# GLM-5.3-Flash-Alis-MTP-Drafter (bf16)

The **standalone native MTP (nextn) drafter** of [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
extracted from the official FP8 release (`84c6a6aa`) for self-speculative decoding with
mlx-vlm PR [#2044](https://github.com/Blaizzy/mlx-vlm/pull/2044). To our knowledge the first
*standalone* GLM-5.3-Flash MTP drafter on the Hub — normal MLX conversions drop layer 45, so
quantized targets otherwise have no MTP path.

- 14 GB, 29 tensors (`mtp.*`), bf16 (+ fp32 router correction bias), `model_type: glm5_next_mtp`, block size 2.
- Loads as `Glm5NextMTPDraftModel` via `load_drafter(path, kind="mtp")` on the PR #2044/#2074 branches.
- Measured (M3 Ultra 512GB, 4-bit-class target, ctx 512, greedy): **1.12× decode, bit-identical
  to plain decode**, ≈89% single-draft acceptance.

## Why a dequantized extraction

The PR branch's splitter currently writes raw FP8 + unstacked `weight_scale_inv` tensors from the
FP8 source and drops the quantization config, producing an unloadable artifact (reported upstream:
[#2044 comment](https://github.com/Blaizzy/mlx-vlm/pull/2044#issuecomment-5466856387)). This repo
applies the reference blockwise dequant (block 128, `mx.from_fp8`, per-expert scale grids rejoined
across the stacked `switch_mlp`) to publish a clean bf16 drafter. Receipts in-repo:
`SPLIT_RECEIPT.json` (extraction) and `DEQUANT_RECEIPT.json` (post-process).

```python
from mlx_vlm.speculative.drafters import load_drafter
drafter, kind = load_drafter("avlp12/GLM-5.3-Flash-Alis-MTP-Drafter", kind="mtp")
```

MIT, derived from the MIT base release.
