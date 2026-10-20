---
license: apache-2.0
language:
-  multilingual
base_model: google/gemma-4-26B-A4B-it
tags:
- auto-round
- intel
- gguf
- quantization
- vlm
---

# Gemma-4-26B-A4B-it GGUF (AutoRound Quantized)

This repository contains GGUF quantized versions of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) created using Intel's [AutoRound](https://github.com/intel/auto-round) quantization method.

## Quantization Details

The models were quantized using various schemes provided by the `auto-round` tool. For better compatibility and smaller size, we provide unified multimodal projector (`mmproj`) files in F16, BF16, and F32 formats.

### Files and Sizes

| File Name | Quant Type | Size | Description |
|-----------|------------|------|-------------|
| `gemma-4-26B-A4B-it-Q2_K_S.gguf` | Q2_K_S | 11 GB | Extremely high compression, significant quality loss. |
| `gemma-4-26B-A4B-it-Q2_K_MIXED.gguf` | Q2_K_MIXED | 12 GB | **Recommended** high-compression option. Uses Q4 for KV cache with good quality. |
| `gemma-4-26B-A4B-it-Q3_K_S.gguf` | Q3_K_S | 13 GB | Very high compression, notable quality loss. |
| `gemma-4-26B-A4B-it-Q3_K_M.gguf` | Q3_K_M | 13 GB | Balanced 3-bit quantization. |
| `gemma-4-26B-A4B-it-Q3_K_L.gguf` | Q3_K_L | 13 GB | High quality 3-bit quantization. |
| `gemma-4-26B-A4B-it-Q4_0.gguf` | Q4_0 | 14 GB | Standard 4-bit quantization, good balance. |
| `gemma-4-26B-A4B-it-Q4_1.gguf` | Q4_1 | 15 GB | Higher quality 4-bit quantization than Q4_0. |
| `gemma-4-26B-A4B-it-Q4_K_S.gguf` | Q4_K_S | 15 GB | Small 4-bit K-quant, good efficiency. |
| `gemma-4-26B-A4B-it-Q4_K_M.gguf` | Q4_K_M | 15 GB | **Recommended** 4-bit K-quant, excellent balance. |
| `gemma-4-26B-A4B-it-Q5_0.gguf" | Q5_0 | 17 GB | Standard 5-bit quantization, very high quality. |
| `gemma-4-26B-A4B-it-Q5_1.gguf` | Q5_1 | 18 GB | Higher quality 5-bit quantization than Q5_0. |
| `gemma-4-26B-A4B-it-Q5_K_S.gguf` | Q5_K_S | 17 GB | Small 5-bit K-quant, very high quality. |
| `gemma-4-26B-A4B-it-Q5_K_M.gguf` | Q5_K_M | 17 GB | **Recommended** 5-bit K-quant, near-lossless. |
| `gemma-4-26B-A4B-it-Q6_K.gguf` | Q6_K | 22 GB | 6-bit K-quant, virtually indistinguishable from F16. |
| `gemma-4-26B-A4B-it-Q8_0.gguf` | Q8_0 | 26 GB | 8-bit quantization, near-lossless. |
| `mmproj-model-f16.gguf` | F16 | 1.2 GB | Unified Projector in Float16 format. |
| `mmproj-model-bf16.gguf` | BF16 | 1.2 GB | Unified Projector in BFloat16 format. |
| `mmproj-model-f32.gguf` | F32 | 2.2 GB | Unified Projector in Float32 format. |

## Generate the Model

The models were generated using Intel's AutoRound with the following command:

```bash
auto-round --model google/gemma-4-26B-A4B-it --output_dir ./quantized/ --scheme <SCHEME> --iters 0
```

**Note:** To reproduce this quantization, you need the following Pull Request:
- [Intel AutoRound PR #1655](https://github.com/intel/auto-round/pull/1655)
- And the specific fix mentioned in [this comment](https://github.com/intel/auto-round/pull/1655#issuecomment-4259531072).

## Usage with llama.cpp

These models can be used with `llama.cpp`. For multimodal usage, you must specify the projector file:

```bash
./llama-cli -m gemma-4-26B-A4B-it-Q4_K_M.gguf --mmproj mmproj-model-f16.gguf --image your_image.jpg -p "Describe this image."
```

## About AutoRound

[AutoRound](https://github.com/intel/auto-round) is an advanced quantization technique from Intel that aims to minimize accuracy loss through automated rounding optimization.
