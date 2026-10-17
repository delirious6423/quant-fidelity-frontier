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

# Qwen3.8-Flash-Next-oQ8e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview of the architecture behind Qwen4: a mixture-of-experts vision-language model with 125B parameters and 6B activated, plus a 51B n-gram embedding and a 4B MTP module, that pairs Gated DeltaNet with Qwen Sparse Attention (QSA) and uses Gated Residual streams. It includes the language model, the n-gram embedding, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 8.660 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ8e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 8-bit, group 128 | 2.406B | 1.337% | 2.311 GiB | 8.25 |
| Affine 8-bit, group 32 | 51.200B | 28.445% | 53.644 GiB | 9.00 |
| Affine 8-bit, group 64 | 125.878B | 69.932% | 124.560 GiB | 8.50 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| **Total** | **180.000B** | **100%** | **181.476 GiB** | **8.660** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 125.744B | 133.590 GB / 124.415 GiB |
| N-gram embedding | 51.200B | 57.600 GB / 53.644 GiB |
| Vision encoder | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 2.770 GB / 2.580 GiB |
| **Total** | **180.000B** | **194.858 GB / 181.476 GiB** |

## License

Qwen Community License 1.0, following the source model. See `LICENSE`.
