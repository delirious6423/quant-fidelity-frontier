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

# Ornith-1.5-35B-A3B-oQ5e-mtp

> [!NOTE]
> **Updated 2026-11-11:** Re-quantized with oMLX v0.7.4.dev1. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.4.dev1) mixed-precision quantization.

The effective average is 5.834 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 5-bit, group 64 | 33.018B | 91.838% | 21.141 GiB | 5.50 |
| Affine 6-bit, group 64 | 0.168B | 0.467% | 0.127 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.129B | 0.359% | 0.124 GiB | 8.25 |
| Affine 8-bit, group 64 | 2.160B | 6.008% | 2.137 GiB | 8.50 |
| BF16 | 0.478B | 1.329% | 0.890 GiB | 16.00 |
| **Total** | **35.952B** | **100%** | **24.418 GiB** | **5.834** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 24.722 GB / 23.024 GiB |
| Vision | 0.447B | 0.893 GB / 0.832 GiB |
| MTP head | 0.845B | 0.604 GB / 0.562 GiB |
| **Total** | **35.952B** | **26.219 GB / 24.418 GiB** |