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

# Qwen3.8-27B-oQ6h-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.2) mixed-precision quantization.

The effective average is 22.054 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 6-bit, group 64 | 1.602B | 18.502% | 1.212 GiB | 6.50 |
| Affine 8-bit, group 64 | 1.313B | 15.171% | 1.300 GiB | 8.50 |
| BF16 | 0.900B | 10.393% | 1.676 GiB | 16.00 |
| U32 | 4.842B | 55.935% | 18.039 GiB | 32.00 |
| **Total** | **8.657B** | **100%** | **22.226 GiB** | **22.054** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 7.772B | 22.526 GB / 20.979 GiB |
| Vision | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.418 GB / 0.389 GiB |
| **Total** | **8.657B** | **23.865 GB / 22.226 GiB** |