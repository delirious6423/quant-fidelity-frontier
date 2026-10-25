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

# Ornith-1.5-35B-A3B-oQ4e-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 4.800 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 33.018B | 91.838% | 17.297 GiB | 4.50 |
| Affine 5-bit, group 128 | 0.252B | 0.700% | 0.154 GiB | 5.25 |
| Affine 5-bit, group 64 | 0.598B | 1.662% | 0.383 GiB | 5.50 |
| Affine 6-bit, group 64 | 0.944B | 2.625% | 0.714 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.129B | 0.359% | 0.124 GiB | 8.25 |
| Affine 8-bit, group 64 | 0.535B | 1.487% | 0.529 GiB | 8.50 |
| BF16 | 0.478B | 1.329% | 0.890 GiB | 16.00 |
| **Total** | **35.952B** | **100%** | **20.090 GiB** | **4.800** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 20.182 GB / 18.796 GiB |
| Vision | 0.447B | 0.893 GB / 0.832 GiB |
| MTP head | 0.845B | 0.497 GB / 0.463 GiB |
| **Total** | **35.952B** | **21.571 GB / 20.090 GiB** |