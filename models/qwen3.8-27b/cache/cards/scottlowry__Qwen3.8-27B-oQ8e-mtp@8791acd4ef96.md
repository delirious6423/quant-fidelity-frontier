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

# Qwen3.8-27B-oQ8e-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 8.639 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 8-bit, group 64 | 27.266B | 98.143% | 26.980 GiB | 8.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **27.941 GiB** | **8.639** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 26.896B | 28.579 GB / 26.617 GiB |
| Vision | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.500 GB / 0.466 GiB |
| **Total** | **27.781B** | **30.001 GB / 27.941 GiB** |