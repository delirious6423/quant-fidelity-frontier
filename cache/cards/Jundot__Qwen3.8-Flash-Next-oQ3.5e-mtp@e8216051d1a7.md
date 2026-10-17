---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- quantized
- qwen4_exp
- mtp
---

# Qwen3.8-Flash-Next-oQ3.5e-mtp

> [!NOTE]
> **Updated 2026-10-12:** Re-quantized with Improved oQe Quantization and 4-bit N-gram embedding tables. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview of the architecture behind Qwen4: a mixture-of-experts vision-language model with 125B parameters and 6B activated, plus a 51B n-gram embedding and a 4B MTP module, that pairs Gated DeltaNet with Qwen Sparse Attention (QSA) and uses Gated Residual streams. It includes the language model, the n-gram embedding, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.990 bits/weight excluding the N-gram embedding table, or 4.277 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ3.5e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias. Below oQ4, the N-gram embedding table stays at 4 bits and is left out of the bit budget, because oMLX can serve it from SSD instead of memory.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 80.537B | 44.743% | 32.815 GiB | 3.50 |
| Affine 4-bit, group 64 | 44.519B | 24.733% | 23.322 GiB | 4.50 |
| Affine 5-bit, group 128 | 0.566B | 0.315% | 0.346 GiB | 5.25 |
| Affine 5-bit, group 64 | 0.100B | 0.056% | 0.064 GiB | 5.50 |
| Affine 6-bit, group 128 | 0.004B | 0.002% | 0.003 GiB | 6.25 |
| Affine 6-bit, group 64 | 0.913B | 0.507% | 0.691 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 1.405B | 0.780% | 1.390 GiB | 8.50 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 28.445% | 29.802 GiB | 5.00 |
| **Total** | **180.000B** | **100%** | **89.625 GiB** | **4.277** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 125.744B | 61.856 GB / 57.608 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision encoder | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 1.480 GB / 1.378 GiB |
| **Total** | **180.000B** | **96.234 GB / 89.625 GiB** |

## License

Qwen Community License 1.0, following the source model. See `LICENSE`.
