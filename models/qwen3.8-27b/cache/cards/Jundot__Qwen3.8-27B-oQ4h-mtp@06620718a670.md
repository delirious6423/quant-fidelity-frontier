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

# Qwen3.8-27B-oQ4h-mtp

> [!IMPORTANT]
> This is an oQh checkpoint. It needs oMLX 0.8.0 or later with oQh support. Earlier oMLX versions cannot load it.

This model was quantized using [oQ](https://github.com/jundot/omlx) mixed-precision quantization in the oQh format ([jundot/omlx#5317](https://github.com/jundot/omlx/pull/5317)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a 27B dense vision-language model built on the Qwen3.5 architecture, with a hybrid layout of Gated DeltaNet and Gated Attention layers, image and video understanding and a 262,144-token native context. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 4.840 bits/weight for the language backbone and 5.044 bits/weight for the complete checkpoint.

## What oQh changes

oQe already gives high quality and is the recommended setting in oMLX. It uses the standard MLX format, so any MLX engine that supports the model can run it. oQh is built for oMLX to push quality further at low bits.

oQh rotates the weights with a Hadamard transform before quantizing them. Rotated weights need fewer scales, and oQh spends the saved space on extra bits where they help most. The file stays the same size as oQe.

## Quality vs oQe

KL divergence from the BF16 model, lower is better. oQe and oQh at the same level are the same size. FLORES+ and Aya are prompts in 7 languages, and code is Python source. The sizes below leave out the vision encoder and the MTP head.

| Level | Size | FLORES+ KL, oQe -> oQh | Aya KL, oQe -> oQh | Code KL, oQe -> oQh |
|---|---|---|---|---|
| oQ2 | 9.9 GiB | 0.19 -> **0.10** (-47%) | 0.21 -> **0.11** (-46%) | 0.13 -> **0.078** (-41%) |
| oQ3 | 12.2 GiB | 0.066 -> **0.040** (-39%) | 0.082 -> **0.040** (-51%) | 0.052 -> **0.037** (-28%) |
| **oQ4** | 15.2 GiB | 0.016 -> **0.012** (-26%) | 0.018 -> **0.013** (-27%) | 0.013 -> **0.012** (-13%) |

## Quantization and Bit Distribution

oQ4h starts from the oQ4 budget and spends the bytes saved on group scales on wider codes for the narrowest and most sensitive tensors. The token embedding, the output head and the MTP head stay in affine groups.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| oQh 4-bit, group 128 | 13.034B | 46.916% | 6.450 GiB | 4.25 |
| oQh 5-bit, group 128 | 11.301B | 40.679% | 6.912 GiB | 5.25 |
| oQh 6-bit, group 128 | 0.015B | 0.053% | 0.011 GiB | 6.36 |
| oQh 8-bit, group 128 | 0.001B | 0.004% | 0.001 GiB | 8.58 |
| Affine 4-bit, group 64 | 0.330B | 1.189% | 0.173 GiB | 4.50 |
| Affine 5-bit, group 64 | 1.303B | 4.690% | 0.834 GiB | 5.50 |
| Affine 6-bit, group 64 | 1.271B | 4.576% | 0.962 GiB | 6.50 |
| Affine 8-bit, group 64 | 0.010B | 0.038% | 0.010 GiB | 8.50 |
| BF16 | 0.516B | 1.857% | 0.961 GiB | 16.00 |
| **Total** | **27.781B** | **100%** | **16.314 GiB** | **5.044** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 26.896B | 16.272 GB / 15.155 GiB |
| Vision encoder | 0.461B | 0.921 GB / 0.858 GiB |
| MTP head | 0.425B | 0.323 GB / 0.301 GiB |
| **Total** | **27.781B** | **17.517 GB / 16.314 GiB** |

## License

Apache-2.0, following the source model. See `LICENSE`.
