---
library_name: mlx
tags:
- mlx
- oq
- quantized
- mtp
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-oQ4h-fp16-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.0) mixed-precision quantization.

The effective average is 20.451 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 0.330B | 4.579% | 0.173 GiB | 4.50 |
| Affine 5-bit, group 64 | 1.303B | 18.063% | 0.834 GiB | 5.50 |
| Affine 6-bit, group 64 | 1.271B | 17.627% | 0.962 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.010B | 0.145% | 0.010 GiB | 8.50 |
| FP16 | 0.439B | 6.086% | 0.818 GiB | 16.00 |
| FP32 | 0.461B | 6.388% | 1.716 GiB | 32.00 |
| U32 | 3.398B | 47.111% | 12.659 GiB | 32.00 |
| **Total** | **7.213B** | **100%** | **17.172 GiB** | **20.451** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 6.327B | 16.272 GB / 15.155 GiB |
| Vision | 0.461B | 1.843 GB / 1.716 GiB |
| MTP head | 0.425B | 0.323 GB / 0.301 GiB |
| **Total** | **7.213B** | **18.439 GB / 17.172 GiB** |