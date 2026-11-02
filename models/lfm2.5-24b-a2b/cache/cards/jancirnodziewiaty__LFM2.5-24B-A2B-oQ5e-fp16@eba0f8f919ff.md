---
library_name: mlx
tags:
- mlx
- oq
- quantized
base_model:
- LiquidAI/LFM2.5-24B-A2B
---

# LFM2.5-24B-A2B-oQ5e-fp16

> [!NOTE]
> **Updated 2026-11-02:** Re-quantized with oMLX v0.7.3. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3) mixed-precision quantization.

The effective average is 5.629 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 5-bit, group 64 | 22.951B | 95.743% | 14.695 GiB | 5.50 |
| Affine 8-bit, group 64 | 1.015B | 4.234% | 1.004 GiB | 8.50 |
| FP16 | 0.005B | 0.022% | 0.010 GiB | 16.00 |
| FP32 | 0.000B | 0.000% | 0.000 GiB | 32.00 |
| **Total** | **23.972B** | **100%** | **15.710 GiB** | **5.629** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 23.972B | 16.868 GB / 15.710 GiB |
| **Total** | **23.972B** | **16.868 GB / 15.710 GiB** |