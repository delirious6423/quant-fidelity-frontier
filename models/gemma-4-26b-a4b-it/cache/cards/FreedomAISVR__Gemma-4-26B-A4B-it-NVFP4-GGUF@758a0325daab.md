---
license: gemma
base_model: google/gemma-4-26B-A4B-it
language:
- en
library_name: gguf
pipeline_tag: text-generation
tags:
- google
- gemma-4
- nvfp4
- 4-bit
- blackwell
- fp4
- moe
- mixture-of-experts
- multimodal
- vision
- instruct
- gguf
quantized_by: FreedomAISVR
---

# Gemma 4 26B A4B IT — NVFP4 GGUF

## Model Description

This is an NVFP4-quantized GGUF version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), Google's state-of-the-art 26B-parameter MoE instruction-tuned model (4B active parameters, 128 experts, 8 experts per token).

NVFP4 is NVIDIA's native Blackwell FP4 quantization format, delivering excellent quality at 4-bit precision with optimized hardware support on Blackwell GPUs.

This repository includes the multimodal vision encoder (mmproj) for image understanding capabilities.

## Quantization Details

| Property | Value |
|----------|-------|
| Base model | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Quantization | NVFP4 (Blackwell-native FP4) |
| Model size | ~13.45 GB |
| Bits per weight | 4.57 BPW |
| Architecture | Gemma4 (30 layers, 128 MoE experts, 4B active) |
| Output type | `gguf` (version 3) |

## Files

| File | Size | Description |
|------|------|-------------|
| `gemma-4-26b-NVFP4.gguf` | 13.45 GB | Quantized text model (NVFP4) |
| `mmproj-gemma-4-26b-f16.gguf` | 1.11 GB | Vision encoder (multimodal projector) |

## Usage

### llama.cpp

```bash
# Text-only inference
./llama-cli -m gemma-4-26b-NVFP4.gguf -p "Hello, how are you?" -n 256

# Multimodal (image) inference
./llama-cli -m gemma-4-26b-NVFP4.gguf --mmproj mmproj-gemma-4-26b-f16.gguf \
  --image path/to/image.jpg -p "Describe this image" -n 256
```

### Inference with llama.cpp servers

```bash
./llama-server -m gemma-4-26b-NVFP4.gguf --mmproj mmproj-gemma-4-26b-f16.gguf \
  --host 0.0.0.0 --port 8080
```

## Technical Notes

- **Thinking is disabled by default**. To enable thinking, pass `enable_thinking=true` in your request.
- Gemma-4 uses a hybrid attention mechanism with sliding window (local) and global (full) attention layers in a 5:1 pattern.
- The model uses proportional RoPE for global attention layers and standard RoPE for sliding window layers.
- NVFP4 requires llama.cpp built with CUDA support and a Blackwell-compatible GPU for best performance.
- CPU inference is supported but slower than GPU-accelerated inference.

## Hardware Requirements

- **GPU**: Blackwell (RTX 5090, B200, etc.) for optimal NVFP4 performance
- **RAM**: 8 GB+ (model fits in most modern GPUs at 4-bit)
- **Storage**: ~14 GB for the model files

## Acknowledgements

- [Google](https://ai.google.dev/) for the original Gemma 4 model
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the GGUF format and quantization tools
- [NVIDIA](https://www.nvidia.com/) for the NVFP4 specification
