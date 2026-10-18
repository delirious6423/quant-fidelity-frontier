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

# DeepSeek-V4-Flash-0731-oQ3e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), the official release of DeepSeek-V4-Flash with enhanced agentic capabilities. It has the same structure as DeepSeek-V4-Flash-DSpark, with a speculative decoding module attached. It includes the language model and the DSpark MTP weights.

The effective average is 3.639 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ3e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (1024 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 277.025B | 91.073% | 112.875 GiB | 3.50 |
| Affine 4-bit, group 64 | 1.635B | 0.538% | 0.857 GiB | 4.50 |
| MXFP4, group 32 | 19.327B | 6.354% | 9.562 GiB | 4.25 |
| Affine 5-bit, group 64 | 0.497B | 0.163% | 0.318 GiB | 5.50 |
| Affine 6-bit, group 64 | 2.416B | 0.794% | 1.828 GiB | 6.50 |
| Affine 8-bit, group 64 | 1.078B | 0.354% | 1.067 GiB | 8.50 |
| I32 | 0.002B | 0.001% | 0.009 GiB | 32.00 |
| MXFP8, group 32 | 1.996B | 0.656% | 1.917 GiB | 8.25 |
| BF16 | 0.167B | 0.055% | 0.310 GiB | 16.00 |
| FP32 | 0.036B | 0.012% | 0.135 GiB | 32.00 |
| **Total** | **304.180B** | **100%** | **128.879 GiB** | **3.639** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 284.335B | 127.557 GB / 118.797 GiB |
| MTP head | 19.846B | 10.825 GB / 10.082 GiB |
| **Total** | **304.180B** | **138.383 GB / 128.879 GiB** |

## License

MIT, following the source model. See `LICENSE`.
