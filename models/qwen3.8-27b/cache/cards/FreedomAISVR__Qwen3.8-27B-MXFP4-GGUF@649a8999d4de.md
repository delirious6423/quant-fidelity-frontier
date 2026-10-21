---
library_name: gguf
license: apache-2.0
tags:
- qwen3
- qwen3.8
- vision
- mxfp4
- gguf
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B MXFP4 GGUF

> GGUF quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for llama.cpp inference on NVIDIA Blackwell GPUs.

Following the widespread community adoption of the Qwen3.5 and Qwen3.6 series, Qwen3.8 is the most capable generation in the Qwen open-model family to date. Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon agentic tasks.

**Qwen3.8-27B** brings these advances to a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control, designed to carry complex, multi-step tasks through to completion with greater reliability.

## Qwen3.8 Highlights

- **Core Capabilities**: Comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
- **Agent Execution**: Stronger autonomous planning and better handling of environment feedback, leading to more reliable end-to-end task completion.
- **Downstream Compatibility**: Broader support for popular harnesses and development tools, making it easier to integrate into your existing stack.
- **Flexible Thinking Control**: Thinking mode is on by default and can be disabled per request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical messages is retained via `preserve_thinking`.
- **Vision-Language Understanding**: Native support for image and video understanding, from STEM diagrams and documents to hour-scale videos.

## Model Overview

| Property | Value |
|----------|-------|
| Base Model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Architecture | Qwen3_5ForConditionalGeneration (vision + language) |
| Language Model Parameters | 27B |
| Hidden Dimension | 5120 |
| Token Embedding | 248,320 (Padded) |
| Number of Layers | 64 |
| Hidden Layout | 16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)) |
| Gated DeltaNet Heads | 48 for V, 16 for QK (head dim 128) |
| Gated Attention Heads | 24 for Q, 4 for KV (head dim 256, RoPE dim 64) |
| Feed Forward Intermediate Dim | 17,408 |
| Context Length | 262,144 natively (extensible to 1M) |
| MTP | Included (next-token prediction head) |
| License | Apache-2.0 |

## Quantization Details

| Property | Value |
|----------|-------|
| Quantization Format | MXFP4 (OCP Microscaled FP4) |
| Bits Per Weight | 4.36 BPW |
| Original Model Size | ~52 GB (F16) |
| Quantized Size | ~14.2 GB |
| KV Cache (recommended) | Q8_0 |
| Quantized With | llama.cpp llama-quantize (MXFP4 ftype) |

### What is MXFP4?

MXFP4 uses the OCP (Open Compute Project) microscaled FP4 format with E8M0 power-of-two scaling per 32 values. The power-of-two scaling means dequantization is just bit-shifts — no floating-point multiply needed — making decode significantly faster than NVFP4 on consumer Blackwell GPUs.

MXFP4 has 50% less scale metadata per weight (0.25 bits overhead) compared to NVFP4 (0.5 bits overhead), further reducing memory bandwidth pressure during decode.

**MXFP4 is the recommended format for consumer RTX 50-series GPUs** due to its superior decode speed and smaller file size.

## Files

| File | Size | Description |
|------|------|-------------|
| `qwen3.8-27b-mxfp4.gguf` | ~14.2 GB | Text model (MXFP4 quantized) |
| `mmproj-qwen3.8-27b-f16.gguf` | ~928 MB | Vision projector (F16) |

## Hardware Requirements

- **GPU**: NVIDIA RTX 50-series (Blackwell) with 16+ GB VRAM
- **RAM**: 16+ GB system RAM recommended
- **Storage**: ~20 GB free disk space

## Usage with llama.cpp

### Text Only
```bash
llama-cli -m qwen3.8-27b-mxfp4.gguf -ngl 99 -p "Hello"
```

### With Vision
```bash
llama-cli -m qwen3.8-27b-mxfp4.gguf   --mmproj mmproj-qwen3.8-27b-f16.gguf   -ngl 99   --chat-template chatml   --conversation
```

### Server Mode (Recommended)
```bash
llama-server   -m qwen3.8-27b-mxfp4.gguf   --mmproj mmproj-qwen3.8-27b-f16.gguf   -ngl 99   --ctx-size 32768   -ctk q8_0 -ctv q8_0   -fa on   -b 512   --ubatch-size 128
```

### Recommended Settings (RTX 5060 Ti 16GB)

| Setting | Value | Why |
|---------|-------|-----|
| `-ngl 99` | All layers on GPU | Full GPU offload for speed |
| `--ctx-size 32768` | 32K context | Sweet spot for speed/quality |
| `-ctk q8_0 -ctv q8_0` | Q8 KV cache | Near-lossless, 2x VRAM savings vs F16 |
| `-fa on` | Flash attention | Faster attention kernels |
| `-b 512` | Batch size | Optimal for single-user |
| `--ubatch-size 128` | Micro-batch | Balances throughput and latency |

## Performance (RTX 5060 Ti 16GB)

| Metric | Value |
|--------|-------|
| Prompt Processing | ~106 t/s |
| Generation | ~25 t/s |
| VRAM Usage | ~15.9 GB |

### Context Length vs Speed

| Context | Generation Speed |
|---------|-----------------|
| 8K | ~26 t/s |
| 32K | ~25 t/s |
| 64K | ~10 t/s |

## Why MXFP4 Over NVFP4?

| Property | MXFP4 | NVFP4 |
|----------|-------|-------|
| Generation Speed | ~25 t/s | ~10 t/s |
| Prompt Processing | ~106 t/s | ~30 t/s |
| File Size | 14.2 GB | 15 GB |
| Scale Type | E8M0 power-of-two | E4M3 FP8 |
| Scale Overhead | 0.25 bits/weight | 0.5 bits/weight |
| Best For | Consumer GPUs | Datacenter Blackwell |

## Citation

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```

## License

Apache-2.0 (same as base model)
