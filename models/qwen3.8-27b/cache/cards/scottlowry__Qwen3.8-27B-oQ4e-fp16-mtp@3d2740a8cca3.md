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

# Qwen3.8-27B-oQ4e-fp16-mtp

> [!NOTE]
> **Updated 2026-11-11:** Re-quantized with oMLX v0.7.4.dev1. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.4.dev1) mixed-precision quantization.

The effective average is 5.310 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 19.393B | 69.807% | 10.160 GiB | 4.50 |
| Affine 5-bit, group 64 | 6.579B | 23.680% | 4.212 GiB | 5.50 |
| Affine 6-bit, group 64 | 1.282B | 4.615% | 0.970 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.011B | 0.041% | 0.011 GiB | 8.50 |
| FP16 | 0.055B | 0.198% | 0.103 GiB | 16.00 |
| FP32 | 0.461B | 1.658% | 1.716 GiB | 32.00 |
| **Total** | **27.781B** | **100%** | **17.172 GiB** | **5.310** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 26.896B | 16.272 GB / 15.155 GiB |
| Vision | 0.461B | 1.843 GB / 1.716 GiB |
| MTP head | 0.425B | 0.323 GB / 0.301 GiB |
| **Total** | **27.781B** | **18.439 GB / 17.172 GiB** |