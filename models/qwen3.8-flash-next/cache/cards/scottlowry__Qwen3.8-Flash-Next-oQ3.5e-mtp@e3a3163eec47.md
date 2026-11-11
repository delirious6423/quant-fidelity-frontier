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

# Qwen3.8-Flash-Next-oQ3.5e-mtp

> [!NOTE]
> **Updated 2026-11-11:** Re-quantized with oMLX v0.7.4.dev1. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.4.dev1) mixed-precision quantization.

The effective average is 4.058 bits/weight excluding the N-gram embedding table, or 4.326 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 80.531B | 44.739% | 32.812 GiB | 3.50 |
| Affine 4-bit, group 64 | 42.866B | 23.815% | 22.456 GiB | 4.50 |
| Affine 6-bit, group 64 | 0.883B | 0.491% | 0.668 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 3.763B | 2.090% | 3.723 GiB | 8.50 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 28.445% | 29.802 GiB | 5.00 |
| **Total** | **180.000B** | **100%** | **90.655 GiB** | **4.326** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 125.744B | 62.971 GB / 58.647 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 1.471 GB / 1.370 GiB |
| **Total** | **180.000B** | **97.340 GB / 90.655 GiB** |