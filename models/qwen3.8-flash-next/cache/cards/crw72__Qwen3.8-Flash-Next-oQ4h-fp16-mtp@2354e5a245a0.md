---
library_name: mlx
tags:
- mlx
- oq
- quantized
- mtp
license: apache-2.0
base_model:
- Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next-oQ4h-fp16-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.0) mixed-precision quantization.

The effective average is 23.610 bits/weight excluding the N-gram embedding table, or 11.253 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 2.517B | 3.264% | 1.318 GiB | 4.50 |
| Affine 8-bit, group 128 | 0.241B | 0.312% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 4.731B | 6.135% | 4.681 GiB | 8.50 |
| FP16 | 1.955B | 2.535% | 3.641 GiB | 16.00 |
| FP32 | 0.449B | 0.582% | 1.672 GiB | 32.00 |
| U32 | 16.017B | 20.772% | 59.668 GiB | 32.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 66.400% | 29.802 GiB | 5.00 |
| **Total** | **77.109B** | **100%** | **101.014 GiB** | **11.253** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 22.852B | 73.154 GB / 68.130 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision | 0.449B | 1.796 GB / 1.672 GiB |
| MTP head | 2.607B | 1.513 GB / 1.409 GiB |
| **Total** | **77.109B** | **108.463 GB / 101.014 GiB** |