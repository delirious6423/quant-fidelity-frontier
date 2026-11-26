---
library_name: mlx
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- oqh
- quantized
- deepseek_v41
- mtp
---

# DeepSeek-V4.1-Flash-1017-oQ2.9h-mtp

This quant is optimized for the 256GB M5 Ultra.

> [!IMPORTANT]
> This is an oQh checkpoint. It needs oMLX 0.8.0 or later with oQh support. Earlier oMLX versions cannot load it.

This model was quantized using [oQ](https://github.com/jundot/omlx) mixed-precision quantization in the oQh format ([jundot/omlx#5317](https://github.com/jundot/omlx/pull/5317)).

A mixed-precision MLX checkpoint of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), a minor update to DeepSeek-V4.1-Flash with gains in reasoning, coding, agentic and visual tasks. It is a mixture-of-experts vision-language model with 552B parameters in the language model, plus 197B of Engram n-gram tables and a 14B MTP module. It includes the language model, the Engram tables, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.443 bits/weight for the language backbone and 3.874 bits/weight for the complete checkpoint.

## What oQh changes

oQe already gives high quality and is the recommended setting in oMLX. It uses the standard MLX format, so any MLX engine that supports the model can run it. oQh is built for oMLX to push quality further at low bits.

oQh rotates the expert weights with a Hadamard transform before quantizing them. Rotated weights need fewer scales, and oQh spends the saved space on extra bits where they help most. The file stays the same size as oQe.

## Quality vs oQe

KL divergence from the original checkpoint (FP4 routed experts, FP8 elsewhere), lower is better. oQe and oQh at the same level are the same size. FLORES+ and Aya are prompts in 7 languages, and code is Python source. Both size columns include the MTP head.

| Level | Size | Size without Engram | FLORES+ KL, oQe -> oQh | Aya KL, oQe -> oQh | Code KL, oQe -> oQh |
|---|---|---|---|---|---|
| oQ2 | 291.5 GiB | 176.8 GiB | 0.59 -> **0.40** (-32%) | 0.20 -> **0.14** (-31%) | 0.16 -> **0.11** (-32%) |
| oQ2.5 | 318.9 GiB | 204.2 GiB | 0.43 -> **0.28** (-34%) | 0.14 -> **0.084** (-38%) | 0.11 -> **0.068** (-38%) |
| oQ2.7 | 329.5 GiB | 214.7 GiB | **0.27** | **0.069** | - |
| **oQ2.9** | 344.2 GiB | 229.5 GiB | **0.25** | **0.058** | - |
| oQ3 | 354.8 GiB | 240.0 GiB | 0.26 -> **0.23** (-9%) | 0.060 -> **0.051** (-15%) | 0.056 -> **0.047** (-16%) |

## Quantization and Bit Distribution

oQ2.9h spends the bytes saved on per-group scales on wider codes for the most sensitive routed tensors. The attention projections and the shared experts stay in 8-bit affine groups, and the token embedding, the output head, the router and the attention output projection wo_a in BF16. The Engram tables are stored in 4-bit affine groups of 32. The MTP head keeps the source FP4 and FP8 formats, and the vision encoder stays in BF16.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| oQh 3-bit, row scale | 366.918B | 48.076% | 128.281 GiB | 3.00 |
| Engram affine 4-bit, group 32 | 196.614B | 25.762% | 114.444 GiB | 5.00 |
| oQh 4-bit, row scale | 176.664B | 23.148% | 82.409 GiB | 4.01 |
| MXFP4, group 32 (source format) | 13.590B | 1.781% | 6.724 GiB | 4.25 |
| Affine 8-bit, group 64 | 5.200B | 0.681% | 5.145 GiB | 8.50 |
| BF16 | 3.399B | 0.445% | 6.332 GiB | 16.00 |
| MXFP8, group 32 (source format) | 0.779B | 0.102% | 0.748 GiB | 8.25 |
| FP32 | 0.042B | 0.006% | 0.158 GiB | 32.00 |
| **Total** | **763.205B** | **100%** | **344.240 GiB** | **3.874** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 551.566B | 237.399 GB / 221.095 GiB |
| Engram | 196.929B | 123.208 GB / 114.747 GiB |
| Vision encoder | 0.485B | 0.971 GB / 0.904 GiB |
| MTP head | 14.225B | 8.047 GB / 7.495 GiB |
| **Total** | **763.205B** | **369.625 GB / 344.240 GiB** |

## License

MIT License, following the source model. See `LICENSE`.
