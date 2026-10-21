---
base_model:
- google/gemma-4-26B-A4B-it
library_name: vllm
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
tags:
- gemma4
- nvfp4
- vllm
- modelopt
- blackwell
- moe
- fp4
pipeline_tag: text-generation
---

# FusenAI/gemma-4-26B-A4B-it-NVFP4

NVFP4-quantized version of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) in **modelopt format**, optimized for vLLM on NVIDIA Blackwell (SM120) GPUs.

## What makes this different

This model follows **NVIDIA's quantization pattern** from [Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4):

- **MoE experts + MLP layers**: Quantized to NVFP4 (4-bit floating point)
- **Self-attention (q/k/v/o projections)**: Kept in **BF16** (not quantized)
- **Router, vision tower, lm_head**: Kept in BF16

Other available NVFP4 quantizations of this model quantize attention layers too, which causes:
1. Quality degradation from aggressive attention quantization
2. vLLM's QKV fusion scale bug (fused projections take `max()` of global scales, causing underflow)

## Performance

Tested on RTX 5090 (Blackwell SM120, 32 GB VRAM) with vLLM 0.19.x:

### Best config: CUDA graphs + BF16 KV cache

```bash
vllm serve FusenAI/gemma-4-26B-A4B-it-NVFP4 \
  --quantization modelopt \
  --max-model-len 4096
```

| Batch | Gen tok/s | Avg Latency |
|-------|----------|-------------|
| B=1 | 127 | 2.0s |
| B=4 | 383 | 2.7s |
| B=16 | 1,221 | 3.3s |
| B=32 | **2,071** | 3.9s |
| B=128 | 2,059 | 4.0s |

### Config comparison (single request)

| Config | tok/s | Notes |
|--------|-------|-------|
| `--enforce-eager` | 18 | No CUDA graphs |
| Default (CUDA graphs) | **127** | **7x faster** |
| `--kv-cache-dtype fp8` | 31 | 2x KV capacity but 4x slower (FlashInfer FP8 overhead) |

### Model details

| Property | Value |
|----------|-------|
| VRAM (model weights) | 17.2 GiB |
| KV cache (BF16) | ~10 GiB / 43K tokens / 15x concurrency at 4096 ctx |
| KV cache (FP8) | ~10 GiB / 87K tokens / 30x concurrency at 4096 ctx |
| Size on disk | 17 GB |
| Quantization | NVFP4 (MLP/MoE only), BF16 (attention) |
| Group size | 16 |

## Quick Start

```bash
# Recommended (CUDA graphs enabled by default)
vllm serve FusenAI/gemma-4-26B-A4B-it-NVFP4 \
  --quantization modelopt \
  --max-model-len 4096

# For maximum KV cache capacity (2x tokens, slower per-request)
vllm serve FusenAI/gemma-4-26B-A4B-it-NVFP4 \
  --quantization modelopt \
  --kv-cache-dtype fp8 \
  --max-model-len 4096
```

## How this was created

1. Started from [RedHatAI/gemma-4-26B-A4B-it-NVFP4](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-NVFP4) (compressed-tensors format)
2. Dequantized all 115 self-attention projections (q/k/v/o across 30 layers) back to BF16 using FP4-E2M1 lookup table
3. Converted from compressed-tensors to modelopt format:
   - `weight_packed` -> `weight` (same data)
   - `weight_global_scale` (divisor) -> `weight_scale_2` (1/divisor)
   - `input_global_scale` (divisor) -> `input_scale` (1/divisor)
4. Updated quantization config to match NVIDIA's modelopt pattern with proper `exclude_modules`

## Tensor Format

**NVFP4 layers (MoE/MLP):**
- `weight`: uint8 (packed FP4-E2M1, two values per byte)
- `weight_scale`: float8_e4m3fn (per-block scale, group_size=16)
- `weight_scale_2`: float32 (global scale, scalar)
- `input_scale`: float32 (activation global scale, scalar)

**BF16 layers (attention):**
- `weight`: bfloat16 (standard dense weight)

## vLLM Compatibility

- **Tested with**: vLLM 0.19.x with `--quantization modelopt`
- **Required patches** (for Gemma 4 heterogeneous attention): PRs #38891, #39084, #39406 (merged in vLLM 0.19.x)
- **GPU**: NVIDIA Blackwell (SM120) or later for native NVFP4 support
- **Backends**: FlashAttention v2 (sliding) + Triton (global attention) + FlashInfer+Cutlass (NVFP4 linear) + VLLM_CUTLASS (NVFP4 MoE)

## License

This model inherits the [Apache 2.0 license](https://ai.google.dev/gemma/docs/gemma_4_license) from Google's Gemma 4.

## Acknowledgments

- [Google DeepMind](https://deepmind.google/) for Gemma 4
- [RedHatAI](https://huggingface.co/RedHatAI) for the initial NVFP4 quantization
- [NVIDIA](https://huggingface.co/nvidia) for the modelopt format and Gemma-4-31B-IT-NVFP4 reference
