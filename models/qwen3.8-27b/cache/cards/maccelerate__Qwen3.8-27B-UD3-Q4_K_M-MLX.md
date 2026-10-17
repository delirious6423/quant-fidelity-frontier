---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- qwen3_5
- qwen3.8
- apple-silicon
- mtp
- speculative-decoding
- mixed-precision
- unsloth-dynamic-v3
- ud3
---

# Qwen3.8-27B-UD3-Q4_K_M-MLX

Native MLX affine re-encoding of the official Unsloth Dynamic v3.0 `Q4_K_M` per-tensor allocation for Qwen3.8-27B.

This is a 16.72 GB, text-only MLX model with 4.894 effective bits per quantized weight. It retains the model's inline MTP head and is intended for [`mlx-serve`](https://github.com/ddalcu/mlx-serve) on Apple Silicon.

> **Before downloading:** use this model with `mlx-serve`. Stock `mlx-lm` is not a supported runtime for these published Qwen3.8 MTP layouts.

## What this model is

The upstream GGUF determines the bit width of each tensor. maccelerate reads that allocation and quantizes the corresponding clean bf16 Qwen weights onto MLX's affine format at the same width.

This model has the **same per-tensor bit-width allocation** as the source GGUF, re-encoded for MLX. It is not a lossless GGUF conversion and does not claim numerical parity with llama.cpp's K-quant or IQ codebooks.

| Claim | Status |
|---|---|
| Same upstream per-tensor allocation | Yes |
| Quantized from clean bf16 weights | Yes |
| Exact numerical parity with the GGUF | No |
| MTP head | Retained inline; one MTP layer |
| Vision tower | Omitted; this is a text-only release |

## Allocation

| MLX width | Tensors |
|---|---:|
| 8-bit | 106 |
| 6-bit | 30 |
| 5-bit | 131 |
| 4-bit | 228 |
| 3-bit | 11 |
| **Total** | **506** |

Widths are not a house preset: each tensor retains the width selected by the upstream Dynamic allocator. The allocation and structural checks for this exact build are recorded in the included `manifest.json`.

## Serving

### Apple Silicon requirements

- Apple Silicon Mac running macOS.
- Allow at least 17 GB of disk for the model files, plus working space for the download.
- The model files occupy 16.72 GB. Actual unified-memory needs also depend on the KV cache, context length, and concurrent requests; no minimum-memory configuration is claimed by this card.

### Run it with mlx-serve

Install and run [`mlx-serve`](https://github.com/ddalcu/mlx-serve) according to its README, download this repository locally, then point it at that directory:

```bash
mlx-serve --model /path/to/Qwen3.8-27B-UD3-Q4_K_M-MLX --kv-quant 8
```

`--kv-quant 8` is a reference configuration, not a memory guarantee for every context length or workload. The pack is mixed-width; MTP is present in the checkpoint, so use a runtime configuration that enables speculative decoding if desired.

### Runtime compatibility

Use `mlx-serve` for this release. Stock `mlx-lm` is not supported for the published Qwen3.8 MTP layout; loading can succeed without providing a supported result.

## Reproducibility and generation code

This checkpoint was generated with [maccelerate](https://github.com/maccelerate-ai/maccelerate), the open-source converter that reads a Dynamic GGUF's per-tensor allocation and re-encodes the matching clean bf16 model for MLX.

## Provenance

| Item | Value |
|---|---|
| Source GGUF | [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), `Qwen3.8-27B-UD-Q4_K_M.gguf` |
| Source GGUF revision | Not recorded in the published manifest |
| Source GGUF SHA-256 | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| bf16 source | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| bf16 source revision | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| imatrix | `imatrix_unsloth.gguf` |
| Converter | [`maccelerate`](https://github.com/maccelerate-ai/maccelerate) (revision not recorded in the published manifest) |
| Output | 8 SafeTensors shards; 16.72 GB |

The included `manifest.json` records the source file and hash, allocation, structural validation, and shard hashes for this exact build.

## License

The model weights, source GGUF, and conversion code are licensed under Apache-2.0. See the included `LICENSE` and the upstream model cards for attribution and terms.
