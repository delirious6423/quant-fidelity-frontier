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

# Ornith-1.5-35B-A3B-oQ8e-fp16-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 8.789 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 8-bit, group 128 | 1.340B | 3.727% | 1.287 GiB | 8.25 |
| Affine 8-bit, group 64 | 34.134B | 94.945% | 33.777 GiB | 8.50 |
| FP16 | 0.031B | 0.086% | 0.058 GiB | 16.00 |
| FP32 | 0.447B | 1.242% | 1.664 GiB | 32.00 |
| **Total** | **35.952B** | **100%** | **36.785 GiB** | **8.789** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 36.806 GB / 34.279 GiB |
| Vision | 0.447B | 1.786 GB / 1.664 GiB |
| MTP head | 0.845B | 0.905 GB / 0.843 GiB |
| **Total** | **35.952B** | **39.498 GB / 36.785 GiB** |