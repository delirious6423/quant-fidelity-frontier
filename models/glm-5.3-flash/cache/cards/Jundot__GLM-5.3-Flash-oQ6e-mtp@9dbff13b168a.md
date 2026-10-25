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

# GLM-5.3-Flash-oQ6e-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.7.1) mixed-precision quantization with Improved oQe Quantization ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)).

A mixed-precision MLX checkpoint of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), a natively multimodal mixture-of-experts model with 320B total and 18B active parameters that combines sparse and linear attention with Manifold-Constrained Hyper-Connections (mHC). It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 6.549 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ6e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 6-bit, group 64 | 316.819B | 98.598% | 239.737 GiB | 6.50 |
| Affine 8-bit, group 64 | 3.810B | 1.186% | 3.770 GiB | 8.50 |
| BF16 | 0.607B | 0.189% | 1.131 GiB | 16.00 |
| FP32 | 0.086B | 0.027% | 0.322 GiB | 32.00 |
| **Total** | **321.323B** | **100%** | **244.960 GiB** | **6.549** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 313.327B | 255.806 GB / 238.237 GiB |
| Vision encoder | 0.564B | 1.127 GB / 1.050 GiB |
| MTP head | 7.433B | 6.091 GB / 5.673 GiB |
| **Total** | **321.323B** | **263.024 GB / 244.960 GiB** |

## License

MIT, following the source model. See `LICENSE`.
