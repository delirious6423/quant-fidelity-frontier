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

# gemma-4-26B-A4B-it-oQ2.5e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), the mixture-of-experts model of the Gemma 4 family from Google DeepMind, with 25.2B total and 3.8B active parameters, text and image input, configurable thinking modes and a 256K-token context. It includes the language model, the vision encoder and a multi-token prediction (MTP) head merged from [google/gemma-4-26B-A4B-it-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant).

The effective average is 3.536 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ2.5e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 2-bit, group 64 | 22.841B | 87.095% | 6.648 GiB | 2.50 |
| Affine 3-bit, group 64 | 0.122B | 0.465% | 0.050 GiB | 3.50 |
| Affine 6-bit, group 64 | 0.012B | 0.045% | 0.009 GiB | 6.50 |
| Affine 8-bit, group 64 | 2.250B | 8.579% | 2.226 GiB | 8.50 |
| BF16 | 1.001B | 3.816% | 1.864 GiB | 16.00 |
| **Total** | **26.226B** | **100%** | **10.797 GiB** | **3.536** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 25.233B | 9.613 GB / 8.953 GiB |
| Vision encoder | 0.573B | 1.140 GB / 1.062 GiB |
| MTP head | 0.420B | 0.839 GB / 0.782 GiB |
| **Total** | **26.226B** | **11.593 GB / 10.797 GiB** |

## License

Apache-2.0, following the source model. See the [Gemma 4 license](https://ai.google.dev/gemma/docs/gemma_4_license).
