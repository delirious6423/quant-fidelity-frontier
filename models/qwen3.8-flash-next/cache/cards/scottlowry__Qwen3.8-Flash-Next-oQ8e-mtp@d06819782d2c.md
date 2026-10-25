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

# Qwen3.8-Flash-Next-oQ8e-mtp

> [!NOTE]
> **Updated 2026-10-25:** Re-quantized with oMLX v0.7.3.dev2. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev2) mixed-precision quantization.

The effective average is 8.525 bits/weight excluding the N-gram embedding table, or 8.660 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 8-bit, group 128 | 2.406B | 1.337% | 2.311 GiB | 8.25 |
| Affine 8-bit, group 64 | 125.878B | 69.932% | 124.560 GiB | 8.50 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 8-bit, group 32 | 51.200B | 28.445% | 53.644 GiB | 9.00 |
| **Total** | **180.000B** | **100%** | **181.476 GiB** | **8.660** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language model | 125.744B | 133.590 GB / 124.415 GiB |
| N-gram embedding | 51.200B | 57.600 GB / 53.644 GiB |
| Vision | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 2.770 GB / 2.580 GiB |
| **Total** | **180.000B** | **194.858 GB / 181.476 GiB** |