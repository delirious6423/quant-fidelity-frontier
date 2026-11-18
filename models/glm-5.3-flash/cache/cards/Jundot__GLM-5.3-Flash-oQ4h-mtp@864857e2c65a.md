---
library_name: mlx
license: mit
license_link: LICENSE
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- oqh
- quantized
- glm5_next
- mtp
---

# GLM-5.3-Flash-oQ4h-mtp

> [!IMPORTANT]
> This is an oQh checkpoint. It needs oMLX 0.8.0 or later with oQh support. Earlier oMLX versions cannot load it.

This model was quantized using [oQ](https://github.com/jundot/omlx) mixed-precision quantization in the oQh format ([jundot/omlx#5317](https://github.com/jundot/omlx/pull/5317)).

A mixed-precision MLX checkpoint of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), the first natively multimodal model in the GLM-5 series, with 320B total parameters and 18B active. It combines sparse and linear attention with Manifold-Constrained Hyper-Connections. It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 4.617 bits/weight for the language backbone and 4.637 bits/weight for the complete checkpoint.

## What oQh changes

oQe already gives high quality and is the recommended setting in oMLX. It uses the standard MLX format, so any MLX engine that supports the model can run it. oQh is built for oMLX to push quality further at low bits.

oQh rotates the expert weights with a Hadamard transform before quantizing them. Rotated weights need fewer scales, and oQh spends the saved space on extra bits where they help most. The file stays the same size as oQe.

## Quality vs oQe

KL divergence from the original FP8 model, lower is better. oQe and oQh at the same level are the same size. FLORES+ and Aya are prompts in 7 languages, and code is Python source. The sizes below leave out the MTP head.

| Level | Size | FLORES+ KL, oQe -> oQh | Aya KL, oQe -> oQh | Code KL, oQe -> oQh |
|---|---|---|---|---|
| oQ2 | 98.6 GiB | 0.34 -> **0.24** (-28%) | 0.80 -> **0.69** (-14%) | 0.25 -> **0.19** (-24%) |
| oQ3 | 134.1 GiB | 0.13 -> **0.11** (-15%) | 0.41 -> **0.37** (-10%) | 0.10 -> **0.11** (+5%) |
| **oQ4** | 169.4 GiB | 0.069 -> **0.070** (+2%) | 0.25 -> **0.25** (-1%) | 0.068 -> **0.057** (-16%) |

The gain is largest at oQ2. At oQ3 oQh is 5% above oQe on code, and at oQ4 FLORES+ and Aya are on par while code is 16% lower.

## Quantization and Bit Distribution

oQ4h starts from the oQ4 budget and spends the bytes saved on group scales on wider codes for the narrowest and most sensitive tensors. Attention, the shared experts, the token embedding and the output head stay in 8-bit affine groups, the routed experts of the MTP head in 4-bit affine groups, and the vision encoder in BF16.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| oQh 4-bit, group 128 | 229.512B | 71.427% | 113.555 GiB | 4.25 |
| oQh 5-bit, group 128 | 74.893B | 23.308% | 45.774 GiB | 5.25 |
| Affine 8-bit, group 64 | 8.976B | 2.793% | 8.882 GiB | 8.50 |
| Affine 4-bit, group 64 | 7.248B | 2.256% | 3.797 GiB | 4.50 |
| BF16 | 0.607B | 0.189% | 1.131 GiB | 16.00 |
| FP32 | 0.086B | 0.027% | 0.322 GiB | 32.00 |
| **Total** | **321.323B** | **100%** | **173.461 GiB** | **4.637** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 313.327B | 180.816 GB / 168.398 GiB |
| Vision encoder | 0.564B | 1.127 GB / 1.050 GiB |
| MTP head | 7.433B | 4.309 GB / 4.013 GiB |
| **Total** | **321.323B** | **186.252 GB / 173.461 GiB** |

## License

MIT License, following the source model. See `LICENSE`.
