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

# Ornith-1.5-35B-A3B-oQ6e-fp16-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 6.945 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 6-bit, group 128 | 0.252B | 0.700% | 0.183 GiB | 6.25 |
| Affine 6-bit, group 64 | 33.018B | 91.838% | 24.984 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.129B | 0.359% | 0.124 GiB | 8.25 |
| Affine 8-bit, group 64 | 2.076B | 5.774% | 2.054 GiB | 8.50 |
| FP16 | 0.031B | 0.086% | 0.058 GiB | 16.00 |
| FP32 | 0.447B | 1.242% | 1.664 GiB | 32.00 |
| **Total** | **35.952B** | **100%** | **29.067 GiB** | **6.945** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 34.661B | 28.720 GB / 26.747 GiB |
| Vision | 0.447B | 1.786 GB / 1.664 GiB |
| MTP head | 0.845B | 0.704 GB / 0.656 GiB |
| **Total** | **35.952B** | **31.211 GB / 29.067 GiB** |