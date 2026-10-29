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

# Qwen3.8-Flash-Next-oQ3.5e-fp16-mtp

> [!NOTE]
> **Updated 2026-10-29:** Re-quantized with the improved oQe quantization in oMLX 0.7.3. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3) mixed-precision quantization with improved oQe ([jundot/omlx#4930](https://github.com/jundot/omlx/pull/4930)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview of the architecture behind Qwen4: a mixture-of-experts vision-language model with 125B parameters and 6B activated, plus a 51B n-gram embedding and a 4B MTP module, that pairs Gated DeltaNet with Qwen Sparse Attention (QSA) and uses Gated Residual streams. It includes the language model, the n-gram embedding, the vision encoder and the multi-token prediction (MTP) head.

This FP16 build stores the quantization scales and biases and the remaining unquantized tensors in FP16 instead of BF16, and keeps the vision encoder in FP32.

The effective average is 4.006 bits/weight for the language backbone and 4.366 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ3.5e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias. Below oQ4, the N-gram embedding table stays at 4 bits and is left out of the bit budget, because oMLX can serve it from SSD instead of memory.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 80.531B | 44.739% | 32.812 GiB | 3.50 |
| Affine 4-bit, group 64 | 42.866B | 23.815% | 22.456 GiB | 4.50 |
| Affine 6-bit, group 64 | 0.883B | 0.491% | 0.668 GiB | 6.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 3.763B | 2.090% | 3.723 GiB | 8.50 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| FP16 | 0.067B | 0.037% | 0.124 GiB | 16.00 |
| FP32 | 0.449B | 0.249% | 1.672 GiB | 32.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 28.445% | 29.802 GiB | 5.00 |
| **Total** | **180.000B** | **100%** | **91.491 GiB** | **4.366** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 125.744B | 62.971 GB / 58.647 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision encoder | 0.449B | 1.796 GB / 1.672 GiB |
| MTP head | 2.607B | 1.471 GB / 1.370 GiB |
| **Total** | **180.000B** | **98.238 GB / 91.491 GiB** |

## License

Qwen Community License 1.0, following the source model. See `LICENSE`.
