---
library_name: mlx
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- mlx
- oq
- quantized
- deepseek_v4
- mtp
---

# DeepSeek-V4-Flash-0731-oQ2.7e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), the official release of DeepSeek-V4-Flash with enhanced agentic capabilities. It has the same structure as DeepSeek-V4-Flash-DSpark, with a speculative decoding module attached. It includes the language model and the DSpark MTP weights.

The effective average is 3.255 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ2.7e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 2-bit, group 64 | 128.849B | 42.359% | 37.500 GiB | 2.50 |
| Affine 3-bit, group 64 | 148.176B | 48.713% | 60.375 GiB | 3.50 |
| MXFP4, group 32 | 19.327B | 6.354% | 9.562 GiB | 4.25 |
| Affine 8-bit, group 64 | 1.369B | 0.450% | 1.354 GiB | 8.50 |
| I32 | 0.002B | 0.001% | 0.009 GiB | 32.00 |
| MXFP8, group 32 | 6.254B | 2.056% | 6.006 GiB | 8.25 |
| BF16 | 0.167B | 0.055% | 0.310 GiB | 16.00 |
| FP32 | 0.036B | 0.012% | 0.135 GiB | 32.00 |
| **Total** | **304.180B** | **100%** | **115.252 GiB** | **3.255** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 284.335B | 112.826 GB / 105.077 GiB |
| MTP head | 19.846B | 10.926 GB / 10.175 GiB |
| **Total** | **304.180B** | **123.751 GB / 115.252 GiB** |

## License

MIT, following the source model. See `LICENSE`.
