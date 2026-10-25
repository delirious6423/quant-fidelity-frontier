---
library_name: mlx
tags:
- mlx
- oq
- quantized
- mtp
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
---

# Ornith-1.5-35B-A3B-oQ4e-fp16-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 4.999 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 33.018B | 91.838% | 17.297 GiB | 4.50 |
| Affine 5-bit, group 128 | 0.252B | 0.700% | 0.154 GiB | 5.25 |
| Affine 5-bit, group 64 | 0.581B | 1.616% | 0.372 GiB | 5.50 |
| Affine 6-bit, group 64 | 0.969B | 2.695% | 0.733 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.129B | 0.359% | 0.124 GiB | 8.25 |
| Affine 8-bit, group 64 | 0.526B | 1.464% | 0.521 GiB | 8.50 |
| FP16 | 0.031B | 0.086% | 0.058 GiB | 16.00 |
| FP32 | 0.447B | 1.242% | 1.664 GiB | 32.00 |
| **Total** | **35.952B** | **100%** | **20.922 GiB** | **4.999** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 20.182 GB / 18.796 GiB |
| Vision | 0.447B | 1.786 GB / 1.664 GiB |
| MTP head | 0.845B | 0.497 GB / 0.463 GiB |
| **Total** | **35.952B** | **22.465 GB / 20.922 GiB** |