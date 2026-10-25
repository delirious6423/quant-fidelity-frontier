---
library_name: mlx
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- quantized
- deepseek_v41
- mtp
---

# DeepSeek-V4.1-Flash-oQ3e-mtp

> [!NOTE]
> **Updated 2026-10-12:** Re-quantized with Improved oQe Quantization and 4-bit Engram tables. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), a multimodal mixture-of-experts model with 552B backbone parameters, a Causal Encoder-Decoder layout, Compressed Sparse Attention 2 (CSA2), Engram conditional memory and DSpark speculative decoding. It includes the language model, the Engram tables, the vision encoder, the multimodal aligner and the DSpark MTP weights.

The effective average is 3.629 bits/weight excluding the Engram tables, or 3.982 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ3e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias. Below oQ4, the Engram tables stay at 4 bits and are left out of the bit budget, because oMLX can serve them from SSD instead of memory.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 543.582B | 71.224% | 221.484 GiB | 3.50 |
| Affine 4-bit, group 64 | 1.953B | 0.256% | 1.023 GiB | 4.50 |
| MXFP4, group 32 | 13.590B | 1.781% | 6.724 GiB | 4.25 |
| Affine 6-bit, group 64 | 1.258B | 0.165% | 0.952 GiB | 6.50 |
| Affine 8-bit, group 64 | 1.988B | 0.261% | 1.968 GiB | 8.50 |
| MXFP8, group 32 | 0.779B | 0.102% | 0.748 GiB | 8.25 |
| BF16 | 3.399B | 0.445% | 6.332 GiB | 16.00 |
| FP32 | 0.042B | 0.006% | 0.158 GiB | 32.00 |
| Engram affine 4-bit, group 32 | 196.614B | 25.762% | 114.444 GiB | 5.00 |
| **Total** | **763.205B** | **100%** | **353.832 GiB** | **3.982** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 551.881B | 248.023 GB / 230.989 GiB |
| Engram tables | 196.614B | 122.884 GB / 114.444 GiB |
| Vision encoder | 0.485B | 0.971 GB / 0.904 GiB |
| MTP head | 14.225B | 8.047 GB / 7.495 GiB |
| **Total** | **763.205B** | **379.925 GB / 353.832 GiB** |

## License

MIT, following the source model. See `LICENSE`.
