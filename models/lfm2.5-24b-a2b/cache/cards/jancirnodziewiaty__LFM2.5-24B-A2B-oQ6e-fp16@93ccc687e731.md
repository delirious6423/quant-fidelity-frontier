---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- LiquidAI/LFM2.5-24B-A2B
---

# LFM2.5-24B-A2B-oQ6e-fp16

> [!NOTE]
> **Updated 2026-11-02:** Re-quantized with oMLX v0.7.3. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3) mixed-precision quantization.

The effective average is 6.565 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 6-bit, group 64 | 23.213B | 96.837% | 17.566 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.753B | 3.141% | 0.745 GiB | 8.50 |
| FP16 | 0.005B | 0.022% | 0.010 GiB | 16.00 |
| FP32 | 0.000B | 0.000% | 0.000 GiB | 32.00 |
| **Total** | **23.972B** | **100%** | **18.320 GiB** | **6.565** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 23.972B | 19.671 GB / 18.320 GiB |
| **Total** | **23.972B** | **19.671 GB / 18.320 GiB** |