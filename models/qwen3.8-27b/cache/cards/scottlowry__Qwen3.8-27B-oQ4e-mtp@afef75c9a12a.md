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

# Qwen3.8-27B-oQ4e-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 4.910 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 23.085B | 83.096% | 12.094 GiB | 4.50 |
| Affine 5-bit, group 64 | 2.909B | 10.471% | 1.863 GiB | 5.50 |
| Affine 6-bit, group 64 | 1.271B | 4.576% | 0.962 GiB | 6.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **15.879 GiB** | **4.910** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 26.896B | 15.814 GB / 14.728 GiB |
| Vision | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.314 GB / 0.293 GiB |
| **Total** | **27.781B** | **17.050 GB / 15.879 GiB** |