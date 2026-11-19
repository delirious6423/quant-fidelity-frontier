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

# GLM-5.3-Flash-oQ2.5e-mtp

> [!NOTE]
> **Updated 2026-11-19:** Re-quantized with the refined oQe rounding in oMLX 0.8.0, which lowers quantization error at the same size and speed. Please re-download if you have an earlier copy.

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.8.0) mixed-precision quantization with improved oQe ([jundot/omlx#4930](https://github.com/jundot/omlx/pull/4930)) and oQe rounding refinement ([jundot/omlx#5292](https://github.com/jundot/omlx/pull/5292)).

A mixed-precision MLX checkpoint of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), a natively multimodal mixture-of-experts model with 320B total and 18B active parameters that combines sparse and linear attention with Manifold-Constrained Hyper-Connections (mHC). It includes the language model, the vision encoder and the multi-token prediction (MTP) head.

The effective average is 3.109 bits/weight for the language backbone and 3.167 bits/weight for the complete checkpoint.

## Quantization and Bit Distribution

oQ2.5e allocates bits by measured layer sensitivity, then rounds each group with an importance matrix (128 samples x 512 tokens) and a weighted least-squares refit of its scale and bias.

| Storage format | Logical weights | Share of total | Tensor storage | Effective bits/weight |
|---|---|---|---|---|
| Affine 2-bit, group 64 | 169.114B | 52.631% | 49.219 GiB | 2.50 |
| Affine 3-bit, group 64 | 135.291B | 42.105% | 55.125 GiB | 3.50 |
| Affine 4-bit, group 64 | 7.248B | 2.256% | 3.797 GiB | 4.50 |
| Affine 8-bit, group 64 | 8.976B | 2.793% | 8.882 GiB | 8.50 |
| BF16 | 0.607B | 0.189% | 1.131 GiB | 16.00 |
| FP32 | 0.086B | 0.027% | 0.322 GiB | 32.00 |
| **Total** | **321.323B** | **100%** | **118.475 GiB** | **3.167** |

## Storage Breakdown

| Component | Logical weights | Storage |
|---|---|---|
| Language backbone | 313.327B | 121.776 GB / 113.413 GiB |
| Vision encoder | 0.564B | 1.127 GB / 1.050 GiB |
| MTP head | 7.433B | 4.309 GB / 4.013 GiB |
| **Total** | **321.323B** | **127.212 GB / 118.475 GiB** |

## License

MIT, following the source model. See `LICENSE`.
