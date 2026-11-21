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

# Qwen3.8-27B-oQ2h-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.2) mixed-precision quantization.

The effective average is 16.535 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 1.271B | 22.208% | 0.518 GiB | 3.50 |
| Affine 4-bit, group 64 | 0.372B | 6.502% | 0.195 GiB | 4.50 |
| Affine 5-bit, group 64 | 1.271B | 22.208% | 0.814 GiB | 5.50 |
| BF16 | 0.523B | 9.138% | 0.974 GiB | 16.00 |
| U32 | 2.287B | 39.944% | 8.519 GiB | 32.00 |
| **Total** | **5.725B** | **100%** | **11.020 GiB** | **16.535** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 4.840B | 10.597 GB / 9.869 GiB |
| Vision | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.314 GB / 0.293 GiB |
| **Total** | **5.725B** | **11.833 GB / 11.020 GiB** |