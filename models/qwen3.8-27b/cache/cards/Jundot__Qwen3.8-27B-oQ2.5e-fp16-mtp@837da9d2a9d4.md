---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- quantized
- qwen3_5
- mtp
---

# Qwen3.8-27B-oQ2.5e-fp16-mtp

> [!NOTE]
> **Updated 2026-10-29:** Re-quantized with the improved oQe quantization in oMLX 0.7.3. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3) mixed-precision quantization with improved oQe ([jundot/omlx#4930](https://github.com/jundot/omlx/pull/4930)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense vision-language model built on the Qwen3.5 architecture, with a hybrid layout of Gated DeltaNet and Gated Attention layers, image and video understanding and a 262,144-token native context. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

This FP16 build stores the quantization scales and biases and the remaining unquantized tensors in FP16 instead of BF16, and keeps the vision encoder in FP32.

The effective average is 3.431 bits/weight for the language backbone and 3.943 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ2.5e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 2-bit, group 64 | 2.333B | 8.398% | 0.679 GiB | 2.50 |
| Affine 3-bit, group 128 | 20.502B | 73.796% | 7.757 GiB | 3.25 |
| Affine 3-bit, group 64 | 1.271B | 4.576% | 0.518 GiB | 3.50 |
| Affine 4-bit, group 64 | 0.378B | 1.360% | 0.198 GiB | 4.50 |
| Affine 5-bit, group 64 | 2.781B | 10.012% | 1.781 GiB | 5.50 |
| Affine 6-bit, group 64 | 0.000B | 0.001% | 0.000 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.000B | 0.001% | 0.000 GiB | 8.50 |
| FP16 | 0.055B | 0.198% | 0.103 GiB | 16.00 |
| FP32 | 0.461B | 1.658% | 1.716 GiB | 32.00 |
| **Total** | **27.781B** | **100%** | **12.752 GiB** | **3.943** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 26.896B | 11.535 GB / 10.743 GiB |
| Vision encoder | 0.461B | 1.843 GB / 1.716 GiB |
| MTP head | 0.425B | 0.314 GB / 0.293 GiB |
| **Total** | **27.781B** | **13.692 GB / 12.752 GiB** |

## License

Apache-2.0, following the source model. See `LICENSE`.
