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

# DeepSeek-V4.1-Flash-1017-oQ2.7h-mtp

> [!IMPORTANT]
> This is an oQh checkpoint. It needs oMLX 0.8.0 or later with oQh support. Earlier oMLX versions cannot load it.

This model was quantized using [oQ](https://github.com/jundot/omlx) mixed-precision quantization in the oQh format ([jundot/omlx#5317](https://github.com/jundot/omlx/pull/5317)).

A mixed-precision MLX checkpoint of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), a minor update to DeepSeek-V4.1-Flash with gains in reasoning, coding, agentic and visual tasks. It is a mixture-of-experts vision-language model with 552B parameters in the language model, plus 197B of Engram n-gram tables and a 14B MTP module. It includes the language model, the Engram tables, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.213 bits/weight for the language backbone and 3.708 bits/weight for the complete checkpoint.

## What oQh changes

oQe already gives high quality and is the recommended setting in oMLX. It uses the standard MLX format, so any MLX engine that supports the model can run it. oQh is built for oMLX to push quality further at low bits.

oQh rotates the expert weights with a Hadamard transform before quantizing them. Rotated weights need fewer scales, and oQh spends the saved space on extra bits where they help most. The file stays the same size as oQe.

## Quality vs oQe

KL divergence from the original checkpoint (FP4 routed experts, FP8 elsewhere), lower is better. oQe and oQh at the same level are the same size. FLORES+ and Aya are prompts in 7 languages, and code is Python source. The sizes below leave out the MTP head.

| Level | Size | FLORES+ KL, oQe -> oQh | Aya KL, oQe -> oQh | Code KL, oQe -> oQh |
|---|---|---|---|---|
| oQ2 | 284.0 GiB | 0.59 -> **0.40** (-32%) | 0.20 -> **0.14** (-31%) | 0.16 -> **0.11** (-32%) |
| oQ2.5 | 311.4 GiB | 0.43 -> **0.28** (-34%) | 0.14 -> **0.084** (-38%) | 0.11 -> **0.068** (-38%) |
| **oQ2.7** | 322.0 GiB | **0.27** | **0.069** | - |
| oQ3 | 347.3 GiB | 0.26 -> **0.23** (-9%) | 0.060 -> **0.051** (-15%) | 0.056 -> **0.047** (-16%) |

At oQ2 and oQ2.5 oQh lowers the KL by about a third on all three sets. At oQ3 the gain is 9-16%.

There is no oQ2.7e build, so the oQ2.7 row shows oQh only. Its numbers come from a later run that also measured oQ2.5h and came within 2% of the values above, with 0.28 on FLORES+ and 0.083 on Aya. In that run oQ2.7h has 6% lower KL than oQ2.5h on FLORES+, 17% lower on Aya and 15% lower on Python source. It is 10.5 GiB larger.

## Quantization and Bit Distribution

oQ2.7h starts from the oQ2.7 budget and spends the bytes saved on per-group scales on wider codes for the most sensitive routed tensors. Compared with oQ2.5h, ten routed gate and up projections move from 2 to 3 bits and ten routed down projections from 3 to 4 bits, so no routed tensor stays at 2 bits. The attention projections and the shared experts stay in 8-bit affine groups, and the token embedding, the output head, the router and the attention output projection wo_a in BF16. The Engram tables are stored in 4-bit affine groups of 32. The MTP head keeps the source FP4 and FP8 formats, and the vision encoder stays in BF16.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| oQh 3-bit, row scale | 493.753B | 64.695% | 172.680 GiB | 3.00 |
| Engram affine 4-bit, group 32 | 196.614B | 25.762% | 114.444 GiB | 5.00 |
| oQh 4-bit, row scale | 49.828B | 6.529% | 23.243 GiB | 4.01 |
| MXFP4, group 32 (source format) | 13.590B | 1.781% | 6.724 GiB | 4.25 |
| Affine 8-bit, group 64 | 5.200B | 0.681% | 5.145 GiB | 8.50 |
| BF16 | 3.399B | 0.445% | 6.332 GiB | 16.00 |
| MXFP8, group 32 (source format) | 0.779B | 0.102% | 0.748 GiB | 8.25 |
| FP32 | 0.042B | 0.006% | 0.158 GiB | 32.00 |
| **Total** | **763.205B** | **100%** | **329.474 GiB** | **3.708** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 551.566B | 221.544 GB / 206.329 GiB |
| Engram | 196.929B | 123.208 GB / 114.747 GiB |
| Vision encoder | 0.485B | 0.971 GB / 0.904 GiB |
| MTP head | 14.225B | 8.047 GB / 7.495 GiB |
| **Total** | **763.205B** | **353.770 GB / 329.474 GiB** |

## License

MIT License, following the source model. See `LICENSE`.
