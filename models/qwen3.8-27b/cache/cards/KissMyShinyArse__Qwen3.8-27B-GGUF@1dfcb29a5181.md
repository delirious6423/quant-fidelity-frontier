---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
---

# What it is

This quant is based on Unsloth's `UD-Q8_K_XL` quant, with the Q8 weights swapped for Q8_CR (ConvRot).

⚠️ **Proof of concept - CUDA only.**
| Quant | Size (GiB) | PPL (Q) | PPL Ratio | ΔPPL | Mean KLD | RMS Δp (%) | Same Top-p (%) |
|---|---|---|---|---|---|---|---|
| **UD-Q8_K_XL-Q8_to_Q8_CR** *(this one)* | 29.30 | 6.9565 | 1.00089 | 0.0062 | **0.00037** | **0.587** | **99.263** |
| Q8_CR | 27.05 | 6.9585 | 1.00118 | 0.0082 | 0.00043 | 0.598 | 99.099 |
| [UD-Q8_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-UD-Q8_K_XL.gguf) *(Unsloth, baseline)* | 29.30 | 6.9537 | 1.00049 | 0.0034 | 0.00083 | 0.823 | 98.992 |
| [Q8_0](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q8_0.gguf) | 27.05 | 6.9560 | 1.00082 | 0.0057 | 0.00095 | 0.942 | 98.742 |


# How to use it
Build llama.cpp with the patch from this repo:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git checkout 749f688fcaa4c472ec034b08cb8a907c45cfaa02
patch -Np1 < /path/to/q8_cr.patch
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON .
cmake --build build --config Release
```

This quantization is CUDA-only: inference requires an NVIDIA GPU. On a system
without CUDA, model loading fails with `failed to find a compatible buffer
type`. For full speed, offload all layers to the GPU (`-ngl 99`). Partial
offload works, but every rotated matmul still executes on the GPU (activations
and weights are copied per layer), so generation becomes extremely slow.

# Quantizing BF16 to Q8_CR

Quantize the [BF16 GGUFs](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main/BF16) using the patched `llama-quantize`:

```bash
llama-quantize Qwen3.8-27B-BF16-00001-of-00002.gguf Qwen3.8-27B_Q8_CR.gguf Q8_CR
```

# Research

See [RESEARCH.md](https://huggingface.co/KissMyShinyArse/Qwen3.8-27B-GGUF/raw/main/RESEARCH.md).