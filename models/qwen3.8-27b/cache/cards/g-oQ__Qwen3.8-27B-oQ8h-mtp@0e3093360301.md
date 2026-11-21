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

# Qwen3.8-27B-oQ8h-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.2) mixed-precision quantization.

The effective average is 23.628 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 8-bit, group 64 | 2.915B | 29.438% | 2.885 GiB | 8.50 |
| BF16 | 0.900B | 9.086% | 1.676 GiB | 16.00 |
| U32 | 6.088B | 61.477% | 22.678 GiB | 32.00 |
| **Total** | **9.902B** | **100%** | **27.239 GiB** | **23.628** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 9.017B | 27.825 GB / 25.914 GiB |
| Vision | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.500 GB / 0.466 GiB |
| **Total** | **9.902B** | **29.247 GB / 27.239 GiB** |