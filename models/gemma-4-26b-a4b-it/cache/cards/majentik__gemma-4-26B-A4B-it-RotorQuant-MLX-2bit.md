---
base_model: google/gemma-4-26B-A4B-it
library_name: mlx
tags:
  - rotorquant
  - kv-cache-quantization
  - gemma
  - gemma4
  - multimodal
  - quantized
  - mlx
  - 2bit
license: apache-2.0
pipeline_tag: image-text-to-text
---

> [!TIP]
> **KV-cache quantization without any fork (recommended, 2026):** upstream
> llama.cpp/Ollama now cover this natively — use `-ctk q8_0 -ctv q8_0`
> (~half KV memory, negligible quality loss: perplexity +0.002–0.05) or
> `-ctk q4_0 -ctv q4_0` (~quarter memory, ≈7.6% perplexity increase). In
> Ollama: `OLLAMA_KV_CACHE_TYPE=q8_0` with `OLLAMA_FLASH_ATTENTION=1`. Keep
> K and V types symmetric to stay on the fast fused Flash-Attention path.
> Since April 2026, mainline llama.cpp also applies Hadamard rotation to
> KV activations ([PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038)),
> which greatly improves low-bit KV quality (opt-out:
> `LLAMA_ATTN_ROT_DISABLE=1`).
>
> The RotorQuant/TurboQuant fork flow below is **experimental/legacy**: the
> TurboQuant llama.cpp PR was closed without merging (June 2026) and the fork
> is unmaintained relative to mainline. It is NOT required to use this model.
<!-- kv-upstream-note -->

# Gemma 4 26B-A4B-it - RotorQuant MLX 2-bit

**2-bit weight-quantized MLX version** of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) with the legacy RotorQuant KV-cache fork (superseded by upstream llama.cpp KV options). Optimized for Apple Silicon inference via the [MLX](https://github.com/ml-explore/mlx) framework. The most aggressive quantization, fitting the full model in the smallest possible footprint. Only 4B parameters are active per token despite 26B total, making this model significantly more efficient at inference time than its parameter count suggests.

Approximate model size: **~7 GB**

## Model Specifications

| Property | Value |
|---|---|
| **Base Model** | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Parameters** | 26 billion total (4 billion active per token) |
| **Architecture** | Mixture-of-Experts (MoE) (4B active per token) |
| **Modality** | Multimodal: image + text input, text output |
| **License** | Apache 2.0 |
| **Weight Quantization** | 2-bit (~7 GB) |
| **KV-Cache Quantization** | RotorQuant |
| **Framework** | MLX (Apple Silicon) |

## Quickstart

```python
import mlx.core as mx
from mlx_lm import load, generate

model, tokenizer = load("majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-2bit")

prompt = "Describe this image in detail."
response = generate(model, tokenizer, prompt=prompt, max_tokens=512)
print(response)
```

For multimodal usage with images:

```python
from mlx_vlm import load, generate

model, processor = load("majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-2bit")

prompt = "What do you see in this image?"
output = generate(model, processor, prompt=prompt, image="path/to/image.jpg", max_tokens=512)
print(output)
```

## About the RotorQuant / TurboQuant labels

RotorQuant and TurboQuant are this project's **release labels**, not distinct
quantization algorithms — for any given tier, both brand repos carry
byte-identical weights produced with the standard MLX / llama.cpp quantizers.
No brand-specific speedup is claimed or measured. The KV-cache fork these
labels originally referred to is legacy; for KV-cache memory savings use the
upstream options described above (`-ctk/-ctv q8_0`, `OLLAMA_KV_CACHE_TYPE`).

## KV-Cache Quantization Comparison

| Method | Prefill Speed | Decode Speed | Memory Savings | Reference |
|---|---|---|---|---|
| **TurboQuant** | 1x (baseline) | 1x (baseline) | High | [arXiv: 2504.19874](https://arxiv.org/abs/2504.19874) |


## Memory Estimates (Gemma 4 26B-A4B-it)

| Precision | Approximate Size | MLX Variant |
|---|---|---|
| FP16 (original) | ~52 GB | -- |
| 8-bit quantized | ~26 GB | [RotorQuant-MLX-8bit](https://huggingface.co/majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-8bit) |
| 4-bit quantized | ~14 GB | [RotorQuant-MLX-4bit](https://huggingface.co/majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-4bit) |
| **2-bit quantized** | **~7 GB** | **This model** |

## Hardware Requirements

This model requires approximately 7 GB of unified memory. Recommended hardware:
- Apple M1 (16 GB+)
- Apple M2 (16 GB+)
- Apple M3 (16 GB+)
- Apple M4 (16 GB+)
- Any Apple Silicon Mac with 16 GB+ unified memory

## See Also

- [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) -- Base model
- [majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-8bit](https://huggingface.co/majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-8bit) -- MLX 8-bit variant
- [majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-4bit](https://huggingface.co/majentik/gemma-4-26B-A4B-it-RotorQuant-MLX-4bit) -- MLX 4-bit variant
- [RotorQuant GitHub](https://github.com/scrya-com/rotorquant)
- [MLX Framework](https://github.com/ml-explore/mlx)

## Quant trade-off (MLX lane)

| Bits | Approx size | Use case | Recommendation |
|---|---|---|---|
| **2-bit** | ~6.8 GB | Aggressive quantization | **Very low-RAM Macs** |
| 3-bit | ~9.4 GB | Lossy but small | Low-RAM Macs |
| 4-bit | ~11 GB | Balanced default | Recommended for most Macs |
| 5-bit | ~13 GB | Higher fidelity | Quality-sensitive |
| 6-bit | ~16 GB | Approaching FP16 quality | High-fidelity |
| 8-bit | ~20 GB | Near-lossless reference | Fidelity-critical work |

(Current variant — **2bit** — is bolded.)

## Variants in this family

(Showing 14 sibling variants under `majentik/gemma-4-26b-a4b-it-*`. The current variant — `RotorQuant-MLX-2bit` — is **bolded**.)

| Variant | Runtime | Approx size | Use case |
|---|---|---|---|
| [RotorQuant-GGUF-IQ4_XS](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-IQ4_XS) | llama.cpp | ~22 GB | Lossy 4-bit, low-RAM CPU/edge |
| [RotorQuant-GGUF-Q2_K](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q2_K) | llama.cpp | ~16 GB | Lossy, low-RAM CPU/edge |
| [RotorQuant-GGUF-Q3_K_M](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q3_K_M) | llama.cpp | ~20 GB | Smaller 3-bit, CPU-friendly |
| [RotorQuant-GGUF-Q4_K_M](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q4_K_M) | llama.cpp | ~29 GB | Balanced default |
| [RotorQuant-GGUF-Q5_K_M](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q5_K_M) | llama.cpp | ~34 GB | Higher fidelity, more RAM |
| [RotorQuant-GGUF-Q8_0](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q8_0) | llama.cpp | ~55 GB | Near-lossless reference |
| **RotorQuant-MLX-2bit** | mlx-lm | ~8.3 GB | Apple Silicon, smallest |
| [RotorQuant-MLX-4bit](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-mlx-4bit) | mlx-lm | ~16 GB | Apple Silicon balanced |
| [RotorQuant-MLX-8bit](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-mlx-8bit) | mlx-lm | ~31 GB | Apple Silicon reference |
