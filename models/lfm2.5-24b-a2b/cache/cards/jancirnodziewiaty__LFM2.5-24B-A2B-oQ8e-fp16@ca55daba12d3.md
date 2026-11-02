---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- LiquidAI/LFM2.5-24B-A2B
---

# LFM2.5-24B-A2B-oQ8e-fp16

> [!NOTE]
> **Updated 2026-11-02:** Re-quantized with oMLX v0.7.3. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3) mixed-precision quantization.

The effective average is 8.502 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 8-bit, group 64 | 23.966B | 99.978% | 23.715 GiB | 8.50 |
| FP16 | 0.005B | 0.022% | 0.010 GiB | 16.00 |
| FP32 | 0.000B | 0.000% | 0.000 GiB | 32.00 |
| **Total** | **23.972B** | **100%** | **23.725 GiB** | **8.502** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 23.972B | 25.475 GB / 23.725 GiB |
| **Total** | **23.972B** | **25.475 GB / 23.725 GiB** |