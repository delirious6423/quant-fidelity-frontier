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

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.2) mixed-precision quantization.

The effective average is 4.809 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 33.601B | 93.460% | 17.602 GiB | 4.50 |
| Affine 5-bit, group 128 | 0.566B | 1.575% | 0.346 GiB | 5.25 |
| Affine 6-bit, group 128 | 0.001B | 0.003% | 0.001 GiB | 6.25 |
| Affine 6-bit, group 64 | 0.008B | 0.023% | 0.006 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.129B | 0.359% | 0.124 GiB | 8.25 |
| Affine 8-bit, group 64 | 1.169B | 3.251% | 1.157 GiB | 8.50 |
| BF16 | 0.478B | 1.329% | 0.890 GiB | 16.00 |
| **Total** | **35.952B** | **100%** | **20.126 GiB** | **4.809** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 20.225 GB / 18.836 GiB |
| Vision | 0.447B | 0.893 GB / 0.832 GiB |
| MTP head | 0.845B | 0.492 GB / 0.458 GiB |
| **Total** | **35.952B** | **21.610 GB / 20.126 GiB** |