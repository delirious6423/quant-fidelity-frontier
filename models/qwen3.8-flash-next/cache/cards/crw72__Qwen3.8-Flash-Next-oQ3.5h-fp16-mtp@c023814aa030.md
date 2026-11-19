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

# Qwen3.8-Flash-Next-oQ3.5h-fp16-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.0) mixed-precision quantization.

The effective average is 23.527 bits/weight excluding the N-gram embedding table, or 10.659 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 2.601B | 3.528% | 1.363 GiB | 4.50 |
| Affine 6-bit, group 64 | 0.883B | 1.198% | 0.668 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.241B | 0.327% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 3.763B | 5.105% | 3.723 GiB | 8.50 |
| FP16 | 0.161B | 0.219% | 0.301 GiB | 16.00 |
| FP32 | 0.449B | 0.609% | 1.672 GiB | 32.00 |
| U32 | 14.418B | 19.559% | 53.711 GiB | 32.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 69.456% | 29.802 GiB | 5.00 |
| **Total** | **73.717B** | **100%** | **91.472 GiB** | **10.659** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 19.460B | 62.951 GB / 58.628 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision | 0.449B | 1.796 GB / 1.672 GiB |
| MTP head | 2.607B | 1.471 GB / 1.370 GiB |
| **Total** | **73.717B** | **98.218 GB / 91.472 GiB** |