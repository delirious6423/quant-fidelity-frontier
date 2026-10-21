---
license: apache-2.0
language:
- en
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
tags:
- gguf
- llama.cpp
- qwen3.5-moe
- moe
- quantized
- text-generation
- inference
library_name: gguf
pipeline_tag: text-generation
---

# Ornith-1.5-35B-A3B GGUF

GGUF quantizations of **[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)** for use with **llama.cpp**, KoboldCpp, LM Studio, Open WebUI, Jan, and other GGUF-compatibledel

Ornith-1.5-35B-A3B is a Qwen3.5-MoE vision-language model featuring:

- 40 transformer layers
- Hybrid attention architecture combining:
  - Gated DeltaNet linear attention (3 out of every 4 layers)
  - Full attention layers
- Mixture-of-Experts architecture
  - 256 routed experts
  - 8 experts active per token
- 1 MTP layer
- MRoPE positional encoding
- Hugging Face architecture:
  - `Qwen3_5MoeForConditionalGeneration`
  - `model_type: qwen3_5_moe`

The upstream llama.cpp runtime supports the model family natively, including conversion, inference, importance matrix generation, and quantization.

> **Note:** These GGUF files are text-only. The vision tower present in the original VLM checkpoint is skipped during GGUF conversion, which is standard practice for current llama.cpp VLM conversions.

## Conversion Notes

This repository was generated directly from the original Hugging Face checkpoint using the current upstream `llama.cpp` conversion pipeline.

Because the original safetensors checkpoint is approximately **72 GB**, generating a BF16 GGUF would require more temporary disk space than was available on the build machine.

Instead:

1. The checkpoint was converted directly to a **Q8_0 GGUF**
2. The Q8_0 file was uploaded as the highest-quality quant
3. All other quantizations were generated from that Q8_0 source using `llama-quantize`

This approach is standard in the llama.cpp ecosystem and produces effectively lossless source material for subsequent K-Quant and IQ-Quant generation.

## Quantizations

### Standard Quants

| Quant | Recommended Use |
|---------|---------|
| Q2_K | Maximum compression |
| Q3_K_S | Very low-memory inference |
| Q3_K_M | Budget quality/runtime balance |
| Q3_K_L | Best Q3 variant |
| Q4_0 | Legacy compatibility |
| Q4_1 | Legacy quality-focused quant |
| Q4_K_S | Small Q4_K variant |
| Q4_K_M | Recommended general-purpose quant |
| Q5_K_S | Compact high-quality quant |
| Q5_K_M | Excellent quality/size tradeoff |
| Q6_K | Near-Q8 quality |
| Q8_0 | Highest quality quant in this repo |

### IQ Quants

Generated using a llama.cpp importance matrix.

| Quant |
|--------|
| IQ2_M |
| IQ3_XXS |
| IQ3_XS |
| IQ3_M |
| IQ4_XS |
| IQ4_NL |

## Importance Matrix

IQ quantizations were generated using a llama.cpp importance matrix built from WikiText-2 calibration data.

To accommodate the build machine's memory constraints:

- Imatrix generation was performed on a temporary **Q2_K** model
- WikiText-2 was used as calibration data
- The resulting imatrix was then applied during IQ quantization

## Recommended Files

### Largest Quality

**Q8_0**

Use if storage and RAM are not concerns and you want the best quality available in this repository.

### Best Overall

**Q4_K_M**

Recommended for most users.

Provides an excellent balance between quality, speed, memory usage, and disk footprint.

### Higher Quality

**Q5_K_M**

Recommended for users with additional RAM who want quality closer to Q6/Q8 while keeping storage reasonable.

### Maximum Compression

**Q2_K**

For extremely constrained systems where fitting the model matters more than quality.

## llama.cpp Usage

Example:

```bash
llama-cli \
  -m Ornith-1.5-35B-A3B-Q4_K_M.gguf \
  -c 8192
```

Server:

```bash
llama-server \
  -m Ornith-1.5-35B-A3B-Q4_K_M.gguf
```

## Compatibility

These files should work in:

- llama.cpp
- LM Studio
- KoboldCpp
- Open WebUI
- Jan
- Text Generation WebUI (llama.cpp backends)
- Any GGUF-compatible runtime with support for Qwen3.5-MoE architectures

For best compatibility, use a recent build of llama.cpp or a frontend based on a recent llama.cpp release.

## Credits

### Original Model

- **ornith-ai** for Ornith-1.5-35B-A3B

### GGUF Conversion & Quantization

- Generated using upstream **llama.cpp**
- Quantized and uploaded by **NANI-Nithin**

## Disclaimer

This repository contains quantized derivatives of the original model. All model behavior, training data, capabilities, and limitations originate from the base model authors.

Please review the original model card for licensing terms, intended use, limitations, and safety considerations.