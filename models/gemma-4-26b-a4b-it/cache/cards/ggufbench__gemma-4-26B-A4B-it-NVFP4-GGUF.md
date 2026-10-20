---
license: apache-2.0
library_name: gguf
tags:
- gemma
- gemma-4
- nvidia
- modelopt
- nvfp4
- fp4
- gguf
- moe
- mixture-of-experts
base_model:
- google/gemma-4-26B-A4B-it
- nvidia/Gemma-4-26B-A4B-NVFP4
---

At [ggufbench.com](https://ggufbench.com), we are always looking to advance local AI on average consumer hardware. We would love it if you took a minute to submit your performance results and launch arguments for this model, and others, on our website!

# Gemma 4 26B A4B IT — NVFP4 GGUF

GGUF conversion of [NVIDIA's Gemma 4 26B A4B NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) — Google's Gemma 4 model quantized to NVIDIA's native FP4 format using ModelOpt — repacked into the GGUF format for use with [llama.cpp](https://github.com/ggml-org/llama.cpp).

Yes, this is using the latest chat template from Google.

## Requirements

**llama.cpp b9080 or above is required** 

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **NVFP4 Quant** | [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) |
| **Architecture** | Gemma 4 (Mixture of Experts) |
| **Quantization** | NVFP4 (NVIDIA FP4, via ModelOpt) |
| **GGUF File Type** | `MOSTLY_NVFP4` (type 39) |
| **Parameters** | 25.23B total (2.6B active per token) |
| **Experts** | 128, top-8 active |
| **Context Length** | 262,144 tokens |
| **Hidden Size** | 2,816 |
| **Attention Heads** | 16 (8 KV, global + sliding window layers) |
| **Vocab Size** | 262,144 tokens |
| **Model Size** | ~16.45 GiB (5.60 BPW) |

## Conversion Notes

This conversion uses the fixes from PR #22804 (now merged in b9080)

- Individual `ffn_gate_exps` / `ffn_up_exps` tensors with native `.scale` / `.input_scale` support
- Per-expert `scale2` values stored as full F32 scalars — no precision loss
- `per_expert_scale` tensors properly folded at conversion time
- All tensor name mappings correctly handled for Gemma 4 NVFP4 checkpoints

## History

This release supersedes an earlier experimental conversion that used a workaround approach:

- **v1 (experimental)**: Fused `ffn_gate_up_exps` tensor with per-expert `scale2` correction factors folded directly into the block-level E4M3 scales during packing. This was necessary because the C++ inference code had no `create_tensor` support for fused gate_up scale tensors. The folding introduced minor precision loss since E4M3 has limited precision (3-bit mantissa).

- **v2 (this release)**: Individual `ffn_gate_exps` / `ffn_up_exps` tensors with native `.scale` / `.input_scale` support, thanks to the C++ changes in PR #22804 which added proper `build_moe_ffn` plumbing for separate gate/up expert tensors. Per-expert scale2 values are stored as full F32 scalars — no precision loss.

## License

The base model [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) is released under **Apache 2.0** by Google.
Conversion scripts are modifications of [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT license).
