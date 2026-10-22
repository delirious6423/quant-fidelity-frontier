---
library_name: mlx
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- quantized
- deepseek_v41
- mtp
---

# DeepSeek-V4.1-Flash-1017-oQ4e-ngram-mtp

N-gram (Engram) tables are kept at the original FP8 precision. All other weights match [DeepSeek-V4.1-Flash-1017-oQ4e-mtp](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ4e-mtp), so resident memory is the same with Engram SSD offload.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), a minor update to [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) with improved reasoning, coding, agentic and visual performance. It is a multimodal mixture-of-experts model with 552B backbone parameters, a Causal Encoder-Decoder layout, Compressed Sparse Attention 2 (CSA2), Engram conditional memory and DSpark speculative decoding. It includes the language model, the Engram tables, the vision encoder, the multimodal aligner and the DSpark MTP weights.

The effective average is 4.365 bits/weight excluding the Engram tables, or 5.366 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ4e keeps the original FP4 and FP8 projection precision, which already fits the budget. The Engram tables also keep the original FP8 storage, E4M3 values with one E8M0 scale per 32 channels.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| MXFP4, group 32 | 557.171B | 73.004% | 275.669 GiB | 4.25 |
| MXFP8, group 32 | 5.959B | 0.781% | 5.723 GiB | 8.25 |
| BF16 | 3.419B | 0.448% | 6.369 GiB | 16.00 |
| FP32 | 0.042B | 0.006% | 0.158 GiB | 32.00 |
| Engram MXFP8, group 32 | 196.614B | 25.762% | 188.833 GiB | 8.25 |
| **Total** | **763.205B** | **100%** | **476.751 GiB** | **5.366** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 551.881B | 300.132 GB / 279.520 GiB |
| Engram tables | 196.614B | 202.758 GB / 188.833 GiB |
| Vision encoder | 0.485B | 0.971 GB / 0.904 GiB |
| MTP head | 14.225B | 8.047 GB / 7.495 GiB |
| **Total** | **763.205B** | **511.908 GB / 476.751 GiB** |

## License

MIT, following the source model. See `LICENSE`.
