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

> [!NOTE]
> **Updated 2026-10-24:** Re-quantized with Better Low-Bit oQ. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev1) mixed-precision quantization with Better Low-Bit oQ ([jundot/omlx#4776](https://github.com/jundot/omlx/pull/4776)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense vision-language model built on the Qwen3.5 architecture, with a hybrid layout of Gated DeltaNet and Gated Attention layers, image and video understanding and a 262,144-token native context. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.640 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ2.5e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 2-bit, group 64 | 3.707B | 13.342% | 1.079 GiB | 2.50 |
| Affine 3-bit, group 128 | 19.133B | 68.870% | 7.239 GiB | 3.25 |
| Affine 3-bit, group 64 | 1.271B | 4.576% | 0.518 GiB | 3.50 |
| Affine 4-bit, group 64 | 0.372B | 1.341% | 0.195 GiB | 4.50 |
| Affine 5-bit, group 64 | 2.781B | 10.012% | 1.781 GiB | 5.50 |
| Affine 6-bit, group 64 | 0.000B | 0.001% | 0.000 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.000B | 0.001% | 0.000 GiB | 8.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **11.773 GiB** | **3.640** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 26.896B | 11.405 GB / 10.622 GiB |
| Vision encoder | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.314 GB / 0.293 GiB |
| **Total** | **27.781B** | **12.641 GB / 11.773 GiB** |

## License

Apache-2.0, following the source model. See `LICENSE`.
