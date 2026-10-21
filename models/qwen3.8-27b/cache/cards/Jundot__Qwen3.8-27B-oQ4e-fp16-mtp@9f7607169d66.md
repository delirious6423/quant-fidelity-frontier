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

# Qwen3.8-27B-oQ4e-fp16-mtp

> [!NOTE]
> **Updated 2026-10-13:** Re-quantized with Improved oQe Quantization. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense vision-language model built on the Qwen3.5 architecture, with a hybrid layout of Gated DeltaNet and Gated Attention layers, image and video understanding and a 262,144-token native context. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

This FP16 build stores the quantization scales and biases and the remaining unquantized tensors in FP16 instead of BF16, and keeps the vision encoder in FP32.

The effective average is 5.152 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ4e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 22.443B | 80.784% | 11.757 GiB | 4.50 |
| Affine 5-bit, group 64 | 4.823B | 17.359% | 3.088 GiB | 5.50 |
| FP16 | 0.055B | 0.198% | 0.103 GiB | 16.00 |
| FP32 | 0.461B | 1.658% | 1.716 GiB | 32.00 |
| **Total** | **27.781B** | **100%** | **16.664 GiB** | **5.152** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 26.896B | 15.736 GB / 14.655 GiB |
| Vision encoder | 0.461B | 1.843 GB / 1.716 GiB |
| MTP head | 0.425B | 0.314 GB / 0.293 GiB |
| **Total** | **27.781B** | **17.893 GB / 16.664 GiB** |

## License

Apache-2.0, following the source model. See `LICENSE`.
