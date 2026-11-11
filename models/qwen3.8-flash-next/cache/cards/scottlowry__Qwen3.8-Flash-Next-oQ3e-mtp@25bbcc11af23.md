---
library_name: mlx
tags:
- mlx
- oq
- quantized
- mtp
base_model:
- Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next-oQ3e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.4.dev1) mixed-precision quantization.

The effective average is 3.762 bits/weight excluding the N-gram embedding table, or 4.114 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 120.796B | 67.109% | 49.219 GiB | 3.50 |
| Affine 4-bit, group 64 | 2.517B | 1.398% | 1.318 GiB | 4.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 4.731B | 2.628% | 4.681 GiB | 8.50 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 28.445% | 29.802 GiB | 5.00 |
| **Total** | **180.000B** | **100%** | **86.213 GiB** | **4.114** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 125.744B | 58.159 GB / 54.165 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 1.513 GB / 1.409 GiB |
| **Total** | **180.000B** | **92.570 GB / 86.213 GiB** |