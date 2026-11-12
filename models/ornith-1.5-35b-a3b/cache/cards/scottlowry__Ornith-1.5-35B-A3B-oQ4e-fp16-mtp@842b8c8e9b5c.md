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
> **Updated 2026-11-11:** Re-quantized with oMLX v0.7.4.dev1. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.4.dev1) mixed-precision quantization.

The effective average is 5.124 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 33.018B | 91.838% | 17.297 GiB | 4.50 |
| Affine 8-bit, group 128 | 0.129B | 0.359% | 0.124 GiB | 8.25 |
| Affine 8-bit, group 64 | 2.328B | 6.474% | 2.303 GiB | 8.50 |
| FP16 | 0.031B | 0.086% | 0.058 GiB | 16.00 |
| FP32 | 0.447B | 1.242% | 1.664 GiB | 32.00 |
| **Total** | **35.952B** | **100%** | **21.445 GiB** | **5.124** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 20.738 GB / 19.313 GiB |
| Vision | 0.447B | 1.786 GB / 1.664 GiB |
| MTP head | 0.845B | 0.503 GB / 0.468 GiB |
| **Total** | **35.952B** | **23.027 GB / 21.445 GiB** |