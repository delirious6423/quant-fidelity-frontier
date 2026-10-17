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

# Qwen3.8-Flash-Next-oQ4e-fp16-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview of the architecture behind Qwen4: a mixture-of-experts vision-language model with 125B parameters and 6B activated, plus a 51B n-gram embedding and a 4B MTP module, that pairs Gated DeltaNet with Qwen Sparse Attention (QSA) and uses Gated Residual streams. It includes the language model, the n-gram embedding, the vision encoder and the multi-token prediction (MTP) head.

This FP16 build stores the quantization scales and biases and the remaining unquantized tensors in FP16 instead of BF16, and keeps the vision encoder in FP32.

The effective average is 4.687 bits/weight excluding the N-gram embedding table, or 4.776 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ4e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 123.319B | 68.511% | 64.603 GiB | 4.50 |
| Affine 5-bit, group 128 | 0.566B | 0.315% | 0.346 GiB | 5.25 |
| Affine 5-bit, group 64 | 1.707B | 0.948% | 1.093 GiB | 5.50 |
| Affine 6-bit, group 128 | 0.003B | 0.001% | 0.002 GiB | 6.25 |
| Affine 6-bit, group 64 | 0.932B | 0.518% | 0.705 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 1.517B | 0.843% | 1.501 GiB | 8.50 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| FP16 | 0.067B | 0.037% | 0.124 GiB | 16.00 |
| FP32 | 0.449B | 0.249% | 1.672 GiB | 32.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 28.445% | 29.802 GiB | 5.00 |
| **Total** | **180.000B** | **100%** | **100.080 GiB** | **4.776** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 125.744B | 72.179 GB / 67.222 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision encoder | 0.449B | 1.796 GB / 1.672 GiB |
| MTP head | 2.607B | 1.486 GB / 1.384 GiB |
| **Total** | **180.000B** | **107.461 GB / 100.080 GiB** |

## License

Qwen Community License 1.0, following the source model. See `LICENSE`.
