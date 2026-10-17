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
> **Updated 2026-10-11:** Re-quantized with Improved oQe Quantization. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview of the architecture behind Qwen4: a mixture-of-experts vision-language model with 125B parameters and 6B activated, plus a 51B n-gram embedding and a 4B MTP module, that pairs Gated DeltaNet with Qwen Sparse Attention (QSA) and uses Gated Residual streams. It includes the language model, the n-gram embedding, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 4.724 bits/weight for the complete checkpoint.

## KL Divergence from the Original Model

Mean next-token KL divergence from the original BF16 model on the oMLX benchmark corpora (lower is better). Each corpus is scored on 16 windows of 2,048 tokens spread across the text (32,768 tokens).

| Corpus | Previous oQ4e (2026-08-27) | Updated oQ4e | Change |
|---|---|---|---|
| Code (Python) | 0.0877 | 0.0776 | -11.6% |
| Novel (English) | 0.1909 | 0.1713 | -10.2% |

## Quantization and Bit Distribution

oQ4e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 32 | 51.200B | 28.445% | 29.802 GiB | 5.00 |
| Affine 4-bit, group 64 | 124.724B | 69.291% | 65.339 GiB | 4.50 |
| Affine 5-bit, group 128 | 0.981B | 0.545% | 0.600 GiB | 5.25 |
| Affine 5-bit, group 64 | 0.302B | 0.168% | 0.193 GiB | 5.50 |
| Affine 6-bit, group 128 | 0.003B | 0.001% | 0.002 GiB | 6.25 |
| Affine 6-bit, group 64 | 0.629B | 0.349% | 0.476 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 1.405B | 0.780% | 1.390 GiB | 8.50 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| **Total** | **180.000B** | **100%** | **98.994 GiB** | **4.724** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 125.744B | 71.913 GB / 66.974 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision encoder | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 1.483 GB / 1.381 GiB |
| **Total** | **180.000B** | **106.294 GB / 98.994 GiB** |

## License

Qwen Community License 1.0, following the source model. See `LICENSE`.
