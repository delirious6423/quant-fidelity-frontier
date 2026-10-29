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

# Qwen3.8-Flash-Next-oQ4e-mtp

> [!NOTE]
> **Updated 2026-10-29:** Re-quantized with the improved oQe quantization in oMLX 0.7.3. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3) mixed-precision quantization with improved oQe ([jundot/omlx#4930](https://github.com/jundot/omlx/pull/4930)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview of the architecture behind Qwen4: a mixture-of-experts vision-language model with 125B parameters and 6B activated, plus a 51B n-gram embedding and a 4B MTP module, that pairs Gated DeltaNet with Qwen Sparse Attention (QSA) and uses Gated Residual streams. It includes the language model, the n-gram embedding, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 4.661 bits/weight for the language backbone and 4.785 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ4e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 123.313B | 68.507% | 64.600 GiB | 4.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 4.731B | 2.628% | 4.681 GiB | 8.50 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 28.445% | 29.802 GiB | 5.00 |
| **Total** | **180.000B** | **100%** | **100.275 GiB** | **4.785** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 125.744B | 73.259 GB / 68.227 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision encoder | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 1.513 GB / 1.409 GiB |
| **Total** | **180.000B** | **107.670 GB / 100.275 GiB** |

## License

Qwen Community License 1.0, following the source model. See `LICENSE`.
