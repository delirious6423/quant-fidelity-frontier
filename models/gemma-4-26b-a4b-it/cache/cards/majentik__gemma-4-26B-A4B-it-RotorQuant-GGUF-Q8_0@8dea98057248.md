---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
  - gguf
  - rotorquant
  - kv-cache-quantization
  - gemma
  - gemma4
  - moe
  - instruct
  - llama-cpp
  - quantized
library_name: gguf
pipeline_tag: image-text-to-text
---

> [!WARNING]
> **Fork compatibility (2026-07-07):** the `llama-cpp-turboquant` fork is currently based on a llama.cpp revision that **predates `gemma4` architecture support** — it fails with `unknown model architecture: 'gemma4'` and cannot run this model at all. Until the fork rebases, use **mainline llama.cpp** (which loads this GGUF fine with standard KV-cache types); the RotorQuant/TurboQuant KV-cache options are not usable with gemma-4 yet.
<!-- gemma4-fork-note -->

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

# gemma-4-26B-A4B-it-RotorQuant-GGUF-Q8_0

GGUF Q8_0 weight-quantized variant of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) optimised for use with **RotorQuant** KV cache compression via a dedicated llama.cpp fork.

> **Important:** RotorQuant KV cache types (`planar3`, `iso3`) are **not** available in upstream llama.cpp, standard Ollama, or LM Studio.
> They require a [specific llama.cpp fork](https://github.com/johndpope/llama-cpp-turboquant/tree/feature/planarquant-kv-cache).
> The GGUF file itself is a standard GGUF and works with any llama.cpp-compatible runtime using normal KV cache types (f16, q8_0, q4_0, etc.).

## Hardware compatibility

| Device | VRAM / RAM | Recommendation |
| --- | --- | --- |
| CPU host with ≥29 GB RAM | ~28.6 GB | works via llama.cpp; slower than GPU but no accelerator required |
| Apple Silicon (Metal) | ~31.2 GB | llama.cpp Metal backend; fast on M-series unified memory |
| NVIDIA GPU (partial offload) | split between GPU + RAM | offload as many layers as VRAM allows; rest on CPU |

## Overview

This model combines two independent compression techniques:

| Technique | What it does | Requirement |
|-----------|-------------|-------------|
| **GGUF Q8_0 weight quantization** | Reduces model size from ~52 GB (BF16) to ~26.0 GB | Any llama.cpp-compatible runtime |
| **RotorQuant KV cache compression** — block-diagonal Clifford-algebra rotors for 3-bit KV cache (`--cache-type-k iso3 --cache-type-v iso3`) | Block-diagonal rotations / random rotation for compressed KV cache | [llama-cpp-turboquant fork](https://github.com/johndpope/llama-cpp-turboquant/tree/feature/planarquant-kv-cache) only |

## Quickstart

### llama.cpp / LM Studio / Ollama (upstream)

The GGUF works as a normal quantised model. For KV-cache savings use the upstream path described in the tip at the top of this card (`-ctk q8_0 -ctv q8_0`, or `q4_0`); no fork is required.

**llama.cpp (upstream)**
```bash
llama-completion -no-cnv -m gemma-4-26B-A4B-it-RotorQuant-GGUF-Q8_0.gguf \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -ngl 99 -fa \
  -p "Explain quantum computing"
```

**LM Studio**
1. Download the GGUF file and load in LM Studio.
2. Enable **Developer Mode** (Settings → Developer).
3. In the model loader's advanced settings, set **Flash Attention** to ON.
4. Set **K Cache Quantization** and **V Cache Quantization** to `q8_0` (or `q4_0` for more aggressive VRAM savings).
5. Note: LM Studio does not currently support RotorQuant's `iso3` cache types. Track [this feature request](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1719) for updates.

**Ollama**
```bash
# Standard Ollama does not support RotorQuant cache types.
# Use with default or q8_0 KV cache via OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_FLASH_ATTENTION=1 ollama run majentik/gemma-4-26B-A4B-it-RotorQuant-GGUF-Q8_0
```

## Specifications

| Property | Value |
|----------|-------|
| Base Model | [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Architecture | Sparse MoE (4B active per token), instruct-tuned |
| Parameters | 26B total, 4B active per token (MoE) |
| Context Length | 128K |
| Weight Quantization | GGUF Q8_0 (near-lossless 8-bit, reference quality) |
| Original Size (BF16) | ~52 GB |
| Quantized File Size | ~26.0 GB |
| KV Cache (RotorQuant) | 3-bit via `--cache-type-k iso3 --cache-type-v iso3` (fork only) |
| KV Cache (standard) | q8_0, q4_0, f16, etc. (any llama.cpp runtime) |
| License | apache-2.0 |
| Modalities | Text + Image (image-text-to-text) |
| Compatible Runtimes | llama.cpp, LM Studio, Ollama, koboldcpp |

## About the RotorQuant / TurboQuant labels

RotorQuant and TurboQuant are this project's **release labels**, not distinct
quantization algorithms — for any given tier, both brand repos carry
byte-identical weights produced with the standard MLX / llama.cpp quantizers.
No brand-specific speedup is claimed or measured. The KV-cache fork these
labels originally referred to is legacy; for KV-cache memory savings use the
upstream options described above (`-ctk/-ctv q8_0`, `OLLAMA_KV_CACHE_TYPE`).

## Current Status of RotorQuant in the Ecosystem

| Runtime | RotorQuant Support | Standard KV Quant |
|---------|---------------------|-------------------|
| llama.cpp (upstream) | ❌ Not merged | ✅ q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 |
| llama-cpp-turboquant fork | ✅ planar3, iso3 | ✅ All standard types |
| LM Studio | ❌ [Requested](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1719) | ✅ Via advanced settings |
| Ollama | ❌ Not supported | ✅ Via OLLAMA_KV_CACHE_TYPE |
| koboldcpp | ❌ Not supported | ✅ Standard types |

## Recommended Settings

For VRAM-constrained setups, standard q8_0 KV cache quantization already halves KV cache memory with negligible quality impact. Flash Attention should always be enabled — it is required for V cache quantization and improves memory efficiency regardless.

| VRAM | Suggested Configuration |
|------|------------------------|
| 24 GB (RTX 4090) | Q8_0 + q8_0 KV cache + Flash Attention, 8K–16K context |
| 16 GB | Q8_0 + q4_0 KV cache + Flash Attention, 4K–8K context |
| 48+ GB | Q8_0 + f16 KV cache, full 32K+ context |

## See Also

- [RotorQuant GitHub](https://github.com/scrya-com/rotorquant)
- [llama-cpp-turboquant fork](https://github.com/johndpope/llama-cpp-turboquant/tree/feature/planarquant-kv-cache)
- [TurboQuant llama.cpp discussion](https://github.com/ggml-org/llama.cpp/discussions/20969)
- [TurboQuant paper (arXiv: 2504.19874)](https://arxiv.org/abs/2504.19874)
- [Base model: google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- [gemma-4-26B-A4B-it announcement](https://blog.google/technology/developers/gemma-4/)

## Quant trade-off (GGUF lane)

| Quant | Approx size | Use case | Recommendation |
|---|---|---|---|
| Q2_K | ~14 GB | Lossy, low-RAM CPU/edge | Resource-constrained inference |
| Q3_K_M | ~16 GB | Smaller-than-Q4, modest quality drop | Edge devices with ~16 GB RAM |
| IQ4_XS | ~14 GB | Importance-quant 4-bit, smaller than Q4_K_M | Best size/quality at 4-bit |
| Q4_K_M | ~19 GB | Balanced default | Recommended for most users |
| Q5_K_M | ~20 GB | Higher fidelity than Q4 | Quality-sensitive applications |
| Q6_K | ~23 GB | Approaching FP16 quality | High-fidelity CPU/edge |
| **Q8_0** | ~27 GB | Near-lossless reference | **Fidelity-critical work** |
| MXFP4_MOE | ~14 GB | Microscaling FP4 (MoE-aware) | vLLM / transformers users |

(Current variant — **Q8_0** — is bolded.)

## Variants in this family

(Showing 14 sibling variants under `majentik/gemma-4-26b-a4b-it-*`. The current variant — `RotorQuant-GGUF-Q8_0` — is **bolded**.)

| Variant | Runtime | Approx size | Use case |
|---|---|---|---|
| [RotorQuant-GGUF-IQ4_XS](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-IQ4_XS) | llama.cpp | ~22 GB | Lossy 4-bit, low-RAM CPU/edge |
| [RotorQuant-GGUF-Q2_K](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q2_K) | llama.cpp | ~16 GB | Lossy, low-RAM CPU/edge |
| [RotorQuant-GGUF-Q3_K_M](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q3_K_M) | llama.cpp | ~20 GB | Smaller 3-bit, CPU-friendly |
| [RotorQuant-GGUF-Q4_K_M](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q4_K_M) | llama.cpp | ~29 GB | Balanced default |
| [RotorQuant-GGUF-Q5_K_M](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-gguf-Q5_K_M) | llama.cpp | ~34 GB | Higher fidelity, more RAM |
| **RotorQuant-GGUF-Q8_0** | llama.cpp | ~55 GB | Near-lossless reference |
| [RotorQuant-MLX-2bit](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-mlx-2bit) | mlx-lm | ~8.3 GB | Apple Silicon, smallest |
| [RotorQuant-MLX-4bit](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-mlx-4bit) | mlx-lm | ~16 GB | Apple Silicon balanced |
| [RotorQuant-MLX-8bit](https://huggingface.co/majentik/gemma-4-26b-a4b-it-rotorquant-mlx-8bit) | mlx-lm | ~31 GB | Apple Silicon reference |
| [TurboQuant-MLX-2bit](https://huggingface.co/majentik/gemma-4-26b-a4b-it-turboquant-mlx-2bit) | mlx-lm | ~8.3 GB | Apple Silicon, smallest |
| [TurboQuant-MLX-4bit](https://huggingface.co/majentik/gemma-4-26b-a4b-it-turboquant-mlx-4bit) | mlx-lm | ~16 GB | Apple Silicon balanced |
| [TurboQuant-MLX-8bit](https://huggingface.co/majentik/gemma-4-26b-a4b-it-turboquant-mlx-8bit) | mlx-lm | ~31 GB | Apple Silicon reference |