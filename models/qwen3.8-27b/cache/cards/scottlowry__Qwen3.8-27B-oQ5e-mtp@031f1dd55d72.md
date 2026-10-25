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

# Qwen3.8-27B-oQ5e-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 5.843 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 5-bit, group 64 | 23.155B | 83.347% | 14.826 GiB | 5.50 |
| Affine 6-bit, group 64 | 4.110B | 14.796% | 3.110 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.000B | 0.001% | 0.000 GiB | 8.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **18.897 GiB** | **5.843** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 26.896B | 19.008 GB / 17.703 GiB |
| Vision | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.361 GB / 0.336 GiB |
| **Total** | **27.781B** | **20.291 GB / 18.897 GiB** |