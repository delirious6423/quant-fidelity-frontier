---
library_name: mlx
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- quantized
- glm5_next
- mtp
---

# GLM-5.3-Flash-oQ3e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), a natively multimodal mixture-of-experts model with 320B total and 18B active parameters that combines sparse and linear attention with Manifold-Constrained Hyper-Connections (mHC). It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.636 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ3e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 3-bit, group 64 | 304.406B | 94.735% | 124.031 GiB | 3.50 |
| Affine 4-bit, group 64 | 10.357B | 3.223% | 5.425 GiB | 4.50 |
| Affine 5-bit, group 64 | 0.225B | 0.070% | 0.144 GiB | 5.50 |
| Affine 6-bit, group 64 | 2.718B | 0.846% | 2.057 GiB | 6.50 |
| Affine 8-bit, group 64 | 2.924B | 0.910% | 2.894 GiB | 8.50 |
| BF16 | 0.607B | 0.189% | 1.131 GiB | 16.00 |
| FP32 | 0.086B | 0.027% | 0.322 GiB | 32.00 |
| **Total** | **321.323B** | **100%** | **136.004 GiB** | **3.636** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 313.327B | 140.631 GB / 130.973 GiB |
| Vision encoder | 0.564B | 1.127 GB / 1.050 GiB |
| MTP head | 7.433B | 4.275 GB / 3.982 GiB |
| **Total** | **321.323B** | **146.033 GB / 136.004 GiB** |

## License

MIT, following the source model. See `LICENSE`.
