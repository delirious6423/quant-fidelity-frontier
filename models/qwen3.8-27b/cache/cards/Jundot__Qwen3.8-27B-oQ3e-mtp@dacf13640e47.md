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

# Qwen3.8-27B-oQ3e-mtp

> [!NOTE]
> **Updated 2026-10-24:** Re-quantized with Better Low-Bit oQ. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev1) mixed-precision quantization with Better Low-Bit oQ ([jundot/omlx#4776](https://github.com/jundot/omlx/pull/4776)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense vision-language model built on the Qwen3.5 architecture, with a hybrid layout of Gated DeltaNet and Gated Attention layers, image and video understanding and a 262,144-token native context. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.976 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ3e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 23.052B | 82.978% | 9.393 GiB | 3.50 |
| Affine 4-bit, group 64 | 1.644B | 5.917% | 0.861 GiB | 4.50 |
| Affine 5-bit, group 64 | 2.564B | 9.229% | 1.642 GiB | 5.50 |
| Affine 6-bit, group 64 | 0.005B | 0.019% | 0.004 GiB | 6.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **12.860 GiB** | **3.976** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 26.896B | 12.573 GB / 11.710 GiB |
| Vision encoder | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.314 GB / 0.293 GiB |
| **Total** | **27.781B** | **13.809 GB / 12.860 GiB** |

## License

Apache-2.0, following the source model. See `LICENSE`.
