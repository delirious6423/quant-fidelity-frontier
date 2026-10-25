---
library_name: mlx
tags:
- mlx
- oq
- quantized
- mtp
base_model:
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-oQ5e-fp16-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 6.108 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 5-bit, group 64 | 23.155B | 83.347% | 14.826 GiB | 5.50 |
| Affine 6-bit, group 64 | 4.110B | 14.796% | 3.110 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.000B | 0.001% | 0.000 GiB | 8.50 |
| FP16 | 0.055B | 0.198% | 0.103 GiB | 16.00 |
| FP32 | 0.461B | 1.658% | 1.716 GiB | 32.00 |
| **Total** | **27.781B** | **100%** | **19.755 GiB** | **6.108** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 26.896B | 19.008 GB / 17.703 GiB |
| Vision | 0.461B | 1.843 GB / 1.716 GiB |
| MTP head | 0.425B | 0.361 GB / 0.336 GiB |
| **Total** | **27.781B** | **21.212 GB / 19.755 GiB** |