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

# Ornith-1.5-35B-A3B-oQ8e-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 8.590 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 8-bit, group 128 | 1.340B | 3.727% | 1.287 GiB | 8.25 |
| Affine 8-bit, group 64 | 34.134B | 94.945% | 33.777 GiB | 8.50 |
| BF16 | 0.478B | 1.329% | 0.890 GiB | 16.00 |
| **Total** | **35.952B** | **100%** | **35.953 GiB** | **8.590** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 36.806 GB / 34.279 GiB |
| Vision | 0.447B | 0.893 GB / 0.832 GiB |
| MTP head | 0.845B | 0.905 GB / 0.843 GiB |
| **Total** | **35.952B** | **38.605 GB / 35.953 GiB** |