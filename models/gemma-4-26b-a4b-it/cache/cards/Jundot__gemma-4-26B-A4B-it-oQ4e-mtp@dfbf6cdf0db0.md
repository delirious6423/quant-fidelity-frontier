---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- quantized
- gemma4
- mtp
---

# gemma-4-26B-A4B-it-oQ4e-mtp

> [!NOTE]
> **Updated 2026-10-24:** Re-quantized with Better Low-Bit oQ. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.3.dev1) mixed-precision quantization with Better Low-Bit oQ ([jundot/omlx#4776](https://github.com/jundot/omlx/pull/4776)).

A mixed-precision MLX checkpoint of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), the mixture-of-experts model of the Gemma 4 family from Google DeepMind, with 25.2B total and 3.8B active parameters, text and image input, configurable thinking modes and a 256K-token context. It includes the language model, the vision encoder and a multi-token prediction (MTP) head merged from [google/gemma-4-26B-A4B-it-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant).

The effective average is 5.077 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ4e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 4-bit, group 64 | 22.838B | 87.083% | 11.964 GiB | 4.50 |
| Affine 5-bit, group 64 | 1.642B | 6.260% | 1.051 GiB | 5.50 |
| Affine 6-bit, group 64 | 0.504B | 1.920% | 0.381 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.242B | 0.921% | 0.239 GiB | 8.50 |
| BF16 | 1.001B | 3.816% | 1.864 GiB | 16.00 |
| **Total** | **26.226B** | **100%** | **15.499 GiB** | **5.077** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 25.233B | 14.662 GB / 13.655 GiB |
| Vision encoder | 0.573B | 1.141 GB / 1.063 GiB |
| MTP head | 0.420B | 0.839 GB / 0.782 GiB |
| **Total** | **26.226B** | **16.642 GB / 15.499 GiB** |

## License

Apache-2.0, following the source model. See the [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license).
