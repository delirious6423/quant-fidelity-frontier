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

# Qwen3.8-27B-oQ6e-mtp

> [!NOTE]
> **Updated 2026-10-11:** Re-quantized with Improved oQe Quantization. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense vision-language model built on the Qwen3.5 architecture, with a hybrid layout of Gated DeltaNet and Gated Attention layers, image and video understanding and a 262,144-token native context. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 6.829 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ6e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 6-bit, group 64 | 25.141B | 90.497% | 19.024 GiB | 6.50 |
| Affine 8-bit, group 64 | 2.124B | 7.646% | 2.102 GiB | 8.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **22.087 GiB** | **6.829** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 26.896B | 22.387 GB / 20.850 GiB |
| Vision encoder | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.407 GB / 0.379 GiB |
| **Total** | **27.781B** | **23.716 GB / 22.087 GiB** |

## License

Apache-2.0, following the source model. See `LICENSE`.
