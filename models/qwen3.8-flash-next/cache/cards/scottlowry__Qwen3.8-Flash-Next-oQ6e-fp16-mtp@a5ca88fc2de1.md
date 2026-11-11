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

# Qwen3.8-Flash-Next-oQ6e-fp16-mtp

> [!NOTE]
> **Updated 2026-11-11:** Re-quantized with oMLX v0.7.4.dev1. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.4.dev1) mixed-precision quantization.

The effective average is 6.671 bits/weight excluding the N-gram embedding table, or 6.764 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 6-bit, group 64 | 123.313B | 68.507% | 93.311 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 4.731B | 2.628% | 4.681 GiB | 8.50 |
| FP16 | 0.067B | 0.037% | 0.124 GiB | 16.00 |
| FP32 | 0.449B | 0.249% | 1.672 GiB | 32.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 6-bit, group 32 | 51.200B | 28.445% | 41.723 GiB | 7.00 |
| **Total** | **180.000B** | **100%** | **141.743 GiB** | **6.764** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 125.744B | 103.458 GB / 96.352 GiB |
| N-gram embedding | 51.200B | 44.800 GB / 41.723 GiB |
| Vision | 0.449B | 1.796 GB / 1.672 GiB |
| MTP head | 2.607B | 2.142 GB / 1.995 GiB |
| **Total** | **180.000B** | **152.196 GB / 141.743 GiB** |