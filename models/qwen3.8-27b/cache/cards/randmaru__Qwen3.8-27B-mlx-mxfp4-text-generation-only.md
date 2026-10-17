---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
library_name: mlx
tags:
- text-generation-inference
pipeline_tag: text-generation
---

# randmaru/Qwen3.8-27B-mlx-mxfp4-text-generation-only

This is an MXFP4 MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple Silicon inference.
> [!Warning] **Text-generation-only:** all 333 vision weights in BF16 were removed.

---

**MXFP4 vs 4Bit Quantization Comparison**

| Parameter | `MXFP4` | `4Bit` |
|----------|---------|--------|
| Quantization format | 4‑bit floating point with microscaling, group 32, shared exponent E8M0 | 4‑bit integer (INT4/NF4) |
| Tensor types | U8, U32, BF16 | BF16, U32 |
| Parameter size (`safetensors`) | **~14.3 GB** | ~15.13 GB |
| Total storage (all files) | **~14.32 GB** | ~15.15 GB |
| Hardware support | Most efficient on GPUs with microscaling / FP8 tensor core support | Broad support, but often requires specialized INT4 kernels |
| Apple Silicon compatibility | Designed with hardware microscaling support in Apple Neural Engine / GPU | Works, but without specialized Neural Engine optimization |
| Inference speed | **Higher** on compatible hardware: FP path, lower dequantization overhead, higher throughput | Kernel‑dependent; usually lower or comparable at similar quality |
| Quality | Better preserves dynamic range, less degradation on outliers | Higher risk of accuracy loss on outliers at the same bitrate |

**Key takeaways:**
- **Parameter size:** The MXFP4 version has a **smaller** `safetensors` footprint: ~14.3 GB vs ~15.13 GB for the 4-bit version.
- **Total storage:** MXFP4 also occupies **less** total disk space: ~14.32 GB vs ~15.15 GB when summing all repository files.
- **Vision weights removed:** Both repositories are text-generation-only. **All 333 vision weights in BF16 were removed** to reduce size and focus inference on text generation.
- **Performance:** MXFP4 typically delivers **higher** inference throughput on hardware with microscaling/FP8 support, especially on Apple Silicon. The shared exponent per group of 32 elements reduces dequantization overhead and enables use of floating‑point tensor cores.
- **Apple Silicon optimization:** MXFP4 is designed with Apple Neural Engine / GPU microscaling support in mind, making it the recommended choice for MacBook.
- **Quality:** MXFP4 preserves dynamic range **better**, so generation quality can be higher at the same compression level.

> [!Note] Actual speed depends on the backend, GPU, batch size, and quantization implementation.