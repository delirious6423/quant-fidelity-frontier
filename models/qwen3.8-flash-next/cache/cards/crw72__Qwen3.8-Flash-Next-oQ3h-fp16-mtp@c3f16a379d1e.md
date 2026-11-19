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

# Qwen3.8-Flash-Next-oQ3h-fp16-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.0) mixed-precision quantization.

The effective average is 23.124 bits/weight excluding the N-gram embedding table, or 10.317 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 2.517B | 3.473% | 1.318 GiB | 4.50 |
| Affine 8-bit, group 128 | 0.241B | 0.332% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 4.731B | 6.529% | 4.681 GiB | 8.50 |
| FP16 | 0.161B | 0.223% | 0.301 GiB | 16.00 |
| FP32 | 0.449B | 0.620% | 1.672 GiB | 32.00 |
| U32 | 13.160B | 18.162% | 49.023 GiB | 32.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 70.662% | 29.802 GiB | 5.00 |
| **Total** | **72.458B** | **100%** | **87.030 GiB** | **10.317** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 18.202B | 58.139 GB / 54.146 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision | 0.449B | 1.796 GB / 1.672 GiB |
| MTP head | 2.607B | 1.513 GB / 1.409 GiB |
| **Total** | **72.458B** | **93.448 GB / 87.030 GiB** |