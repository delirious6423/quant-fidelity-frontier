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

# DeepSeek-V4.1-Flash-1017-oQ2e-ngram-mtp

N-gram (Engram) tables are kept at the original FP8 precision. All other weights match [DeepSeek-V4.1-Flash-1017-oQ2e-mtp](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2e-mtp), so resident memory is the same with Engram SSD offload.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), a minor update to [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) with improved reasoning, coding, agentic and visual performance. It is a multimodal mixture-of-experts model with 552B backbone parameters, a Causal Encoder-Decoder layout, Compressed Sparse Attention 2 (CSA2), Engram conditional memory and DSpark speculative decoding. It includes the language model, the Engram tables, the vision encoder, the multimodal aligner and the DSpark MTP weights.

The effective average is 2.688 bits/weight excluding the Engram tables, or 4.121 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ2e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias. The Engram tables keep the original FP8 storage, E4M3 values with one E8M0 scale per 32 channels, and are left out of the bit budget because oMLX can serve them from SSD instead of memory.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 2-bit, group 64 | 543.582B | 71.224% | 158.203 GiB | 2.50 |
| MXFP4, group 32 | 13.590B | 1.781% | 6.724 GiB | 4.25 |
| Affine 8-bit, group 64 | 5.200B | 0.681% | 5.145 GiB | 8.50 |
| MXFP8, group 32 | 0.779B | 0.102% | 0.748 GiB | 8.25 |
| BF16 | 3.399B | 0.445% | 6.332 GiB | 16.00 |
| FP32 | 0.042B | 0.006% | 0.158 GiB | 32.00 |
| Engram MXFP8, group 32 | 196.614B | 25.762% | 188.833 GiB | 8.25 |
| **Total** | **763.205B** | **100%** | **366.143 GiB** | **4.121** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 551.881B | 181.366 GB / 168.911 GiB |
| Engram tables | 196.614B | 202.758 GB / 188.833 GiB |
| Vision encoder | 0.485B | 0.971 GB / 0.904 GiB |
| MTP head | 14.225B | 8.047 GB / 7.495 GiB |
| **Total** | **763.205B** | **393.143 GB / 366.143 GiB** |

## License

MIT, following the source model. See `LICENSE`.
