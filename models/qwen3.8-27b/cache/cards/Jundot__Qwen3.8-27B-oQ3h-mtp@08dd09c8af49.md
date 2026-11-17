---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- oqh
- quantized
- qwen3_5
- mtp
---

# Qwen3.8-27B-oQ3h-mtp

> [!IMPORTANT]
> This is an oQh checkpoint. It needs oMLX 0.8.0 or later with oQh support. Earlier oMLX versions cannot load it.

This model was quantized using [oQ](https://github.com/jundot/omlx) mixed-precision quantization in the oQh format ([jundot/omlx#5317](https://github.com/jundot/omlx/pull/5317)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense vision-language model built on the Qwen3.5 architecture, with a hybrid layout of Gated DeltaNet and Gated Attention layers, image and video understanding and a 262,144-token native context. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.903 bits/weight for the language backbone and 4.135 bits/weight for the complete checkpoint.

## What oQh changes

oQh stores each decoder linear as integer codes on an input rotated by a signed block Hadamard transform. Each output row has one scale. Dropping the per-group scales and biases of oQe saves about 0.5 bits/weight, and the planner spends those bytes on extra bits for the most sensitive tensors, so an oQh file has the same size as oQe at the same level. From oQ4, oQh keeps a scale and an offset per 128 rotated weights instead. The codes are picked against each layer's input statistics.

## Quality vs oQe

KL divergence from the BF16 model, lower is better. oQe and oQh at the same level are the same size. FLORES+ and Aya are prompts in 7 languages, and code is Python source. The sizes below leave out the vision encoder and the MTP head.

| Level | Size | FLORES+ KL, oQe -> oQh | Aya KL, oQe -> oQh | Code KL, oQe -> oQh |
|---|---|---|---|---|
| oQ2 | 9.9 GiB | 0.19 -> **0.10** (-47%) | 0.21 -> **0.11** (-46%) | 0.13 -> **0.078** (-41%) |
| **oQ3** | 12.2 GiB | 0.066 -> **0.040** (-39%) | 0.082 -> **0.040** (-51%) | 0.052 -> **0.037** (-28%) |
| oQ4 | 15.2 GiB | 0.016 -> **0.012** (-26%) | 0.018 -> **0.013** (-27%) | 0.013 -> **0.012** (-13%) |

## Quantization and Bit Distribution

oQ3h starts from the oQ3 budget and spends the bytes saved on group scales on wider codes for the narrowest and most sensitive tensors. The token embedding, the output head and the MTP head stay in affine groups.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| oQh 3-bit, row scale | 8.735B | 31.441% | 3.055 GiB | 3.00 |
| oQh 4-bit, row scale | 13.458B | 48.444% | 6.274 GiB | 4.00 |
| oQh 5-bit, row scale | 2.146B | 7.726% | 1.251 GiB | 5.01 |
| oQh 6-bit, row scale | 0.011B | 0.039% | 0.008 GiB | 6.18 |
| oQh 8-bit, row scale | 0.000B | 0.001% | 0.000 GiB | 8.34 |
| Affine 4-bit, group 64 | 1.644B | 5.916% | 0.861 GiB | 4.50 |
| Affine 6-bit, group 64 | 1.271B | 4.576% | 0.962 GiB | 6.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **13.372 GiB** | **4.135** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 26.896B | 13.123 GB / 12.221 GiB |
| Vision encoder | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.314 GB / 0.293 GiB |
| **Total** | **27.781B** | **14.358 GB / 13.372 GiB** |

## License

Apache-2.0, following the source model. See `LICENSE`.
