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
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 6.661 bits/weight excluding the N-gram embedding table, or 6.757 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 6-bit, group 128 | 0.566B | 0.315% | 0.412 GiB | 6.25 |
| Affine 6-bit, group 64 | 123.313B | 68.507% | 93.311 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 4.164B | 2.314% | 4.121 GiB | 8.50 |
| FP16 | 0.067B | 0.037% | 0.124 GiB | 16.00 |
| FP32 | 0.449B | 0.249% | 1.672 GiB | 32.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 6-bit, group 32 | 51.200B | 28.445% | 41.723 GiB | 7.00 |
| **Total** | **180.000B** | **100%** | **141.595 GiB** | **6.757** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 125.744B | 103.298 GB / 96.204 GiB |
| N-gram embedding | 51.200B | 44.800 GB / 41.723 GiB |
| Vision | 0.449B | 1.796 GB / 1.672 GiB |
| MTP head | 2.607B | 2.142 GB / 1.995 GiB |
| **Total** | **180.000B** | **152.036 GB / 141.595 GiB** |