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

# Ornith-1.5-35B-A3B-oQ6e-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 6.746 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 6-bit, group 128 | 0.252B | 0.700% | 0.183 GiB | 6.25 |
| Affine 6-bit, group 64 | 33.018B | 91.838% | 24.984 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.129B | 0.359% | 0.124 GiB | 8.25 |
| Affine 8-bit, group 64 | 2.076B | 5.774% | 2.054 GiB | 8.50 |
| BF16 | 0.478B | 1.329% | 0.890 GiB | 16.00 |
| **Total** | **35.952B** | **100%** | **28.235 GiB** | **6.746** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 28.720 GB / 26.747 GiB |
| Vision | 0.447B | 0.893 GB / 0.832 GiB |
| MTP head | 0.845B | 0.704 GB / 0.656 GiB |
| **Total** | **35.952B** | **30.317 GB / 28.235 GiB** |