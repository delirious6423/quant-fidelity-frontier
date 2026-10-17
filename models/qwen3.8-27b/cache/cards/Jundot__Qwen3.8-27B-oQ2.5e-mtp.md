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

# Qwen3.8-27B-oQ2.5e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense vision-language model built on the Qwen3.5 architecture, with a hybrid layout of Gated DeltaNet and Gated Attention layers, image and video understanding and a 262,144-token native context. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.643 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ2.5e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 2-bit, group 64 | 21.422B | 77.109% | 6.235 GiB | 2.50 |
| Affine 3-bit, group 64 | 0.000B | 0.001% | 0.000 GiB | 3.50 |
| Affine 4-bit, group 64 | 0.372B | 1.340% | 0.195 GiB | 4.50 |
| Affine 5-bit, group 64 | 2.923B | 10.521% | 1.872 GiB | 5.50 |
| Affine 6-bit, group 64 | 0.005B | 0.019% | 0.004 GiB | 6.50 |
| Affine 8-bit, group 64 | 2.543B | 9.153% | 2.516 GiB | 8.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **11.782 GiB** | **3.643** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 26.896B | 11.415 GB / 10.631 GiB |
| Vision encoder | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.314 GB / 0.293 GiB |
| **Total** | **27.781B** | **12.651 GB / 11.782 GiB** |

## License

Apache-2.0, following the source model. See `LICENSE`.
