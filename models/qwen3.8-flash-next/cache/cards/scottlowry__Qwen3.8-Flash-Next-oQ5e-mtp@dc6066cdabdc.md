---
library_name: mlx
tags:
- mlx
- oq
- quantized
- mtp
base_model:
- Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next-oQ5e-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 5.643 bits/weight excluding the N-gram embedding table, or 5.745 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 5-bit, group 128 | 0.566B | 0.315% | 0.346 GiB | 5.25 |
| Affine 5-bit, group 64 | 123.313B | 68.507% | 78.955 GiB | 5.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 4.164B | 2.314% | 4.121 GiB | 8.50 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 5-bit, group 32 | 51.200B | 28.445% | 35.763 GiB | 6.00 |
| **Total** | **180.000B** | **100%** | **120.377 GiB** | **5.745** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 125.744B | 88.128 GB / 82.076 GiB |
| N-gram embedding | 51.200B | 38.400 GB / 35.763 GiB |
| Vision | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 1.827 GB / 1.702 GiB |
| **Total** | **180.000B** | **129.254 GB / 120.377 GiB** |