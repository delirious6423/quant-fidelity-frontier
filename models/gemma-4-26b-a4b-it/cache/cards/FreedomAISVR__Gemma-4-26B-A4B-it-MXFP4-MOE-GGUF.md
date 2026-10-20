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
- mxfp4
- mxfp4_moe
- 4-bit
- fp4
- moe
- mixture-of-experts
- multimodal
- vision
- instruct
- gguf
quantized_by: FreedomAISVR
---

# Gemma 4 26B A4B IT — MXFP4-MOE GGUF

## Model Description

This is an MXFP4_MOE-quantized GGUF version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), Google's state-of-the-art 26B-parameter MoE instruction-tuned model (4B active parameters, 128 experts, 8 experts per token).

MXFP4_MOE is a specialized MXFP4 quantization variant designed for Mixture-of-Experts architectures, providing efficient 4-bit quantization while preserving the expert routing quality.

This repository includes the multimodal vision encoder (mmproj) for image understanding capabilities.

## Quantization Details

| Property | Value |
|----------|-------|
| Base model | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Quantization | MXFP4_MOE (MXFP4 for MoE) |
| Model size | ~13.72 GB |
| Bits per weight | 4.66 BPW |
| Architecture | Gemma4 (30 layers, 128 MoE experts, 4B active) |
| Output type | `gguf` (version 3) |

## Files

| File | Size | Description |
|------|------|-------------|
| `gemma-4-26b-MXFP4-MOE.gguf` | 13.72 GB | Quantized text model (MXFP4_MOE) |
| `mmproj-gemma-4-26b-f16.gguf` | 1.11 GB | Vision encoder (multimodal projector) |

## Usage

### llama.cpp

```bash
# Text-only inference
./llama-cli -m gemma-4-26b-MXFP4-MOE.gguf -p "Hello, how are you?" -n 256

# Multimodal (image) inference
./llama-cli -m gemma-4-26b-MXFP4-MOE.gguf --mmproj mmproj-gemma-4-26b-f16.gguf \
  --image path/to/image.jpg -p "Describe this image" -n 256
```

### Inference with llama.cpp servers

```bash
./llama-server -m gemma-4-26b-MXFP4-MOE.gguf --mmproj mmproj-gemma-4-26b-f16.gguf \
  --host 0.0.0.0 --port 8080
```

## Technical Notes

- **Thinking is disabled by default**. To enable thinking, pass `enable_thinking=true` in your request.
- Gemma-4 uses a hybrid attention mechanism with sliding window (local) and global (full) attention layers in a 5:1 pattern.
- The model uses proportional RoPE for global attention layers and standard RoPE for sliding window layers.
- MXFP4_MOE is designed specifically for MoE architectures and provides better quality than general-purpose quantization formats for expert-heavy models.
- CPU and GPU inference are both supported.

## Hardware Requirements

- **RAM**: 8 GB+ (model fits in most modern GPUs at 4-bit)
- **Storage**: ~15 GB for the model files

## Acknowledgements

- [Google](https://ai.google.dev/) for the original Gemma 4 model
- [llama.cpp](https://github.com/ggml-org/llama.cpp) for the GGUF format and quantization tools
- [AMD](https://www.amd.com/) for the MXFP4 specification
