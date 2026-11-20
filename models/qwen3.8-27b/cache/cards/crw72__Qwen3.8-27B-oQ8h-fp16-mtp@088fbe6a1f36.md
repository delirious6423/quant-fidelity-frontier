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

# Qwen3.8-27B-oQ8h-fp16-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.0) mixed-precision quantization.

The effective average is 24.373 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 8-bit, group 64 | 2.915B | 29.438% | 2.885 GiB | 8.50 |
| FP16 | 0.439B | 4.433% | 0.818 GiB | 16.00 |
| FP32 | 0.461B | 4.653% | 1.716 GiB | 32.00 |
| U32 | 6.088B | 61.477% | 22.678 GiB | 32.00 |
| **Total** | **9.902B** | **100%** | **28.097 GiB** | **24.373** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 9.017B | 27.825 GB / 25.914 GiB |
| Vision | 0.461B | 1.843 GB / 1.716 GiB |
| MTP head | 0.425B | 0.500 GB / 0.466 GiB |
| **Total** | **9.902B** | **30.169 GB / 28.097 GiB** |