---
library_name: gguf
license: apache-2.0
tags:
- qwen3
- qwen3.8
- vision
- nvfp4
- gguf
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B NVFP4 GGUF

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
| Quantization Format | NVFP4 (NVIDIA FP4) |
| Bits Per Weight | 4.60 BPW |
| Original Model Size | ~52 GB (F16) |
| Quantized Size | ~15 GB |
| KV Cache (recommended) | Q8_0 |
| Quantized With | llama.cpp llama-quantize (NVFP4 ftype) |

### What is NVFP4?

NVFP4 uses NVIDIA's FP4 format with E4M3 per-16-value scaling plus an FP32 tensor scale. This format is designed for Blackwell (RTX 50-series) GPUs with native FP4 tensor core support. It provides the best accuracy among 4-bit formats due to its smaller 16-element scaling blocks.

**Note on decode speed**: On consumer RTX 50-series GPUs, NVFP4 decode is slower than MXFP4 (~10 t/s vs ~25 t/s on RTX 5060 Ti 16GB) because decode is memory-bandwidth-bound and NVFP4's E4M3 scaling requires extra memory reads. NVFP4 excels at prefill (prompt processing) and datacenter Blackwell GPUs with high-bandwidth HBM3e.

## Files

| File | Size | Description |
|------|------|-------------|
| `qwen3.8-27b-nvfp4.gguf` | ~15 GB | Text model (NVFP4 quantized) |
| `mmproj-qwen3.8-27b-f16.gguf` | ~928 MB | Vision projector (F16) |

## Hardware Requirements

- **GPU**: NVIDIA RTX 50-series (Blackwell) with 16+ GB VRAM
- **RAM**: 16+ GB system RAM recommended
- **Storage**: ~20 GB free disk space

## Usage with llama.cpp

### Text Only
```bash
llama-cli -m qwen3.8-27b-nvfp4.gguf -ngl 99 -p "Hello"
```

### With Vision
```bash
llama-cli -m qwen3.8-27b-nvfp4.gguf   --mmproj mmproj-qwen3.8-27b-f16.gguf   -ngl 99   --chat-template chatml   --conversation
```

### Server Mode (Recommended)
```bash
llama-server   -m qwen3.8-27b-nvfp4.gguf   --mmproj mmproj-qwen3.8-27b-f16.gguf   -ngl 99   --ctx-size 32768   -ctk q8_0 -ctv q8_0   -fa on   -b 512   --ubatch-size 128
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
| Prompt Processing | ~30 t/s |
| Generation | ~10 t/s |
| VRAM Usage | ~15.9 GB |

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
