---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- oqh
- quantized
- qwen4_exp
- mtp
---

# Qwen3.8-Flash-Next-oQ4h-mtp

> [!IMPORTANT]
> This is an oQh checkpoint. It needs oMLX 0.8.0 or later with oQh support. Earlier oMLX versions cannot load it.

This model was quantized using [oQ](https://github.com/jundot/omlx) mixed-precision quantization in the oQh format ([jundot/omlx#5317](https://github.com/jundot/omlx/pull/5317)).

A mixed-precision MLX checkpoint of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), an experimental preview of the architecture behind Qwen4: a mixture-of-experts vision-language model with 125B parameters and 6B activated, plus a 51B n-gram embedding and a 4B MTP module, that pairs Gated DeltaNet with Qwen Sparse Attention (QSA) and uses Gated Residual streams. It includes the language model, the n-gram embedding, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 4.654 bits/weight for the language backbone and 4.781 bits/weight for the complete checkpoint.

## What oQh changes

oQe already gives high quality and is the recommended setting in oMLX. It uses the standard MLX format, so any MLX engine that supports the model can run it. oQh is built for oMLX to push quality further at low bits.

oQh rotates the expert weights with a Hadamard transform before quantizing them. Rotated weights need fewer scales, and oQh spends the saved space on extra bits where they help most. The file stays the same size as oQe.

## Quality vs oQe

KL divergence from the original FP8 model, lower is better. oQe and oQh at the same level are the same size. FLORES+ and Aya are prompts in 7 languages, and code is Python source. The sizes below leave out the MTP head.

| Level | Size | FLORES+ KL, oQe -> oQh | Aya KL, oQe -> oQh | Code KL, oQe -> oQh |
|---|---|---|---|---|
| oQ2 | 70.8 GiB | 0.27 -> **0.17** (-38%) | 0.36 -> **0.14** (-62%) | 0.32 -> **0.24** (-24%) |
| oQ3 | 84.9 GiB | 0.082 -> **0.060** (-27%) | 0.055 -> **0.042** (-23%) | 0.14 -> **0.11** (-18%) |
| **oQ4** | 98.8 GiB | 0.034 -> **0.029** (-15%) | 0.018 -> **0.018** (+3%) | 0.065 -> **0.058** (-12%) |

At oQ4 one 1,768-token Aya document holds the whole Aya difference: without it oQh is 20% below oQe on the other 303 documents.

## Quantization and Bit Distribution

oQ4h starts from the oQ4 budget and spends the bytes saved on group scales on wider codes for the narrowest and most sensitive tensors. Attention, the shared experts, the token embedding, the output head and the MTP head stay in affine groups, and the n-gram embedding stays at 4 bits.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| oQh 4-bit, group 128 | 91.436B | 50.798% | 45.240 GiB | 4.25 |
| oQh 5-bit, group 128 | 29.360B | 16.311% | 17.944 GiB | 5.25 |
| Affine 4-bit, group 64 | 2.517B | 1.398% | 1.318 GiB | 4.50 |
| Affine 8-bit, group 128 | 0.241B | 0.134% | 0.231 GiB | 8.25 |
| Affine 8-bit, group 64 | 4.731B | 2.628% | 4.681 GiB | 8.50 |
| BF16 | 0.516B | 0.287% | 0.961 GiB | 16.00 |
| I64 | 0.000B | 0.000% | 0.000 GiB | 64.00 |
| N-gram affine 4-bit, group 32 | 51.200B | 28.445% | 29.802 GiB | 5.00 |
| **Total** | **180.000B** | **100%** | **100.178 GiB** | **4.781** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 125.744B | 73.154 GB / 68.130 GiB |
| N-gram embedding | 51.200B | 32.000 GB / 29.802 GiB |
| Vision encoder | 0.449B | 0.898 GB / 0.836 GiB |
| MTP head | 2.607B | 1.513 GB / 1.409 GiB |
| **Total** | **180.000B** | **107.565 GB / 100.178 GiB** |

## License

Qwen Community License 1.0, following the source model. See `LICENSE`.
