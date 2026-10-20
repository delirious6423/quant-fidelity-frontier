---
language:
  - en
  - ko
  - ja
  - zh
license: apache-2.0
tags:
  - gguf
  - gemma
  - gemma4
  - moe
  - quantized
  - imatrix
  - apple-silicon
  - ollama
  - batiai
  - on-device
  - vision
  - multimodal
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
library_name: llama.cpp
---

# Gemma 4 26B-A4B-it GGUF — Quantized by BatiAI

<p align="center">
  <a href="https://flow.bati.ai"><img src="https://img.shields.io/badge/BatiFlow-macOS%20AI%20Automation-blue?style=for-the-badge&logo=apple" alt="BatiFlow"></a>
  <a href="https://ollama.com/batiai/gemma4-26b"><img src="https://img.shields.io/badge/Ollama-batiai%2Fgemma4--26b-green?style=for-the-badge" alt="Ollama"></a>
  <a href="https://huggingface.co/google/gemma-4-26B-A4B-it"><img src="https://img.shields.io/badge/Upstream-Gemma%204%2026B--A4B--it-orange?style=for-the-badge" alt="Upstream"></a>
</p>

> Optimized GGUF quantizations of **google/gemma-4-26B-A4B-it** for on-device AI on Mac.
> Built directly from official Google BF16 weights by [BatiAI](https://bati.ai) for [BatiFlow](https://flow.bati.ai) — free, unlimited, on-device AI automation.

Gemma 4 26B-A4B is a **Mixture-of-Experts** model (26B total parameters, ~3.8B active per token) with native multilingual + vision capabilities. Optimized for chat, code, tool calling, and now image understanding (via opt-in mmproj — see [Two modes](#two-modes--text-only-by-default-multimodal-opt-in) below).

## Quick Start

```bash
# 24 GB Mac — Best quality / speed (recommended)
ollama pull batiai/gemma4-26b:iq4

# 24 GB Mac — Smaller, imatrix-calibrated
ollama pull batiai/gemma4-26b:iq3

# 24 GB Mac — Standard 3-bit
ollama pull batiai/gemma4-26b:q3

# 32 GB+ Mac — Higher quality
ollama pull batiai/gemma4-26b:q4

# 36 GB+ Mac — Highest on-device quality
ollama pull batiai/gemma4-26b:q6
```

## Available Quantizations

| Tag | Quant | Size | Recommended For |
|-----|-------|------|-----------------|
| `:iq3` | IQ3_M (imatrix) | **12 GB** | 24 GB Mac, smaller footprint |
| `:iq4` | **IQ4_XS (imatrix)** | **13 GB** | **24 GB+ Mac, recommended** |
| `:q3` | Q3_K_M | 13 GB | Standard 3-bit, stable |
| `:q4` | Q4_K_M | 16 GB | 32 GB+ Mac, higher quality |
| `:q6` | Q6_K | 21 GB | 36 GB+ Mac, near-lossless |

> **Note for 16 GB Macs**: 26B MoE loads but runs at ~0.3 t/s due to swap. For 16 GB Macs use [batiai/gemma4-e4b](https://ollama.com/batiai/gemma4-e4b) (5 GB, ~57 t/s) instead.

## Two modes — text-only by default, multimodal opt-in

Upstream Gemma 4 26B-A4B-it is **multimodal** (text + image + video understanding — but no audio; for audio see the smaller [E2B](https://huggingface.co/batiai/Gemma-4-E2B-it-GGUF) / [E4B](https://huggingface.co/batiai/Gemma-4-E4B-it-GGUF) variants). In the GGUF ecosystem this is delivered as **two files**: a main `model.gguf` (text tower) and a separate `mmproj.gguf` (multi-modal projector — the vision tower). We ship **both**, separate, so you can pick:

| | **Text-only** (default) | **Multimodal** (opt-in) |
|---|---|---|
| Files needed | main GGUF only | main GGUF **+** `mmproj-*.gguf` |
| Capabilities | Q&A, coding, tool calling, RAG, agents | + image / video understanding (OCR, captioning, visual reasoning) |
| `ollama pull` | ✅ single command | ⚠ Ollama mmproj integration is still rough — use llama.cpp directly |
| Disk / RAM | smaller (no vision weights) | larger (+ ~770 MB to ~1.2 GB) |
| Recommended for | most users (chat, code, agents) | OCR, image understanding, multimodal tasks |

This is the same pattern unsloth / bartowski / mradermacher use for multimodal models — text-only on Ollama, full multimodal via llama.cpp + mmproj.

### Multimodal usage (llama.cpp)

Download the main GGUF + the mmproj file:

```bash
# Pick a main model (text tower)
wget https://huggingface.co/batiai/Gemma-4-26B-A4B-it-GGUF/resolve/main/google-gemma-4-26B-A4B-it-IQ4_XS.gguf

# Pick the mmproj (vision tower) — Q6_K is the sweet spot, BF16 if you want zero loss
wget https://huggingface.co/batiai/Gemma-4-26B-A4B-it-GGUF/resolve/main/mmproj-Q6_K.gguf
```

**Server mode (OpenAI-compatible Vision API):**

```bash
llama-server \
  -m google-gemma-4-26B-A4B-it-IQ4_XS.gguf \
  --mmproj mmproj-Q6_K.gguf \
  -c 32768 --host 127.0.0.1 --port 8080

# Then post images via the OpenAI Vision API shape
curl http://127.0.0.1:8080/v1/chat/completions -d '{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
      {"type": "text", "text": "What does this screenshot show?"}
    ]
  }]
}'
```

**One-shot CLI:**

```bash
llama-mtmd-cli \
  -m google-gemma-4-26B-A4B-it-IQ4_XS.gguf \
  --mmproj mmproj-Q6_K.gguf \
  --image ~/Desktop/photo.jpg \
  -p "describe this image"
```

### mmproj quantizations available

| File | Quant | Size | When to use |
|------|-------|------|-------------|
| `mmproj-Q6_K.gguf` | Q6_K | ~770 MB | balanced (recommended) |
| `mmproj-BF16.gguf` | BF16 | ~1.2 GB | absolute zero quantization loss for vision |

(Q8_0 not available because some vision tensors have shapes incompatible with Q8_0's column alignment requirement — applies to every quantizer of this model. Q6_K's K-quant block layout handles them.)

## Benchmarks

### MacBook Pro M4 Pro 48 GB — consumer Mac (real-user measurement)

| Metric | **IQ4_XS** | Q6_K | Ollama 26B (official) |
|--------|-----------:|-----:|----------------------:|
| **Token generation** | **58–63 t/s** | 48–50 t/s | 56 t/s |
| VRAM | **15.1 GB** | 23.9 GB | 19.3 GB |
| System memory free | **58 %** | 40 % | 50 % |
| Cold start | **1.7 s** | 5.8 s | 3.4 s |
| Simple response | **0.4 s** | 0.5 s | 0.5 s |
| Coding task | **6.8 s** | 7.3 s | 6.0 s |
| Reasoning (thinking) | **4.1 s** | 5.3 s | 4.8 s |
| Tool calling | ✅ verified | ✅ verified | ⚠ untested |
| Korean language | ✅ verified | ✅ verified | ⚠ untested |

**BatiAI IQ4 outperforms Ollama's official 26B** on the same hardware — both in raw speed (58–63 vs 56 t/s) and memory efficiency (15.1 vs 19.3 GB VRAM).

### MacBook Pro M4 Max 128 GB — developer Mac

| Metric | **IQ4_XS** | IQ3_M | Q3_K_M | Q4_K_M | Q6_K |
|--------|-----------:|------:|-------:|-------:|-----:|
| **Token generation** | **85.8 t/s** | 77.0 t/s | 70.7 t/s | 74.9 t/s | 74.8 t/s |
| Prompt eval | 114.9 t/s | 250 t/s | 250 t/s | 250 t/s | 164.6 t/s |
| VRAM | 22 GB | 19 GB | 20 GB | 23 GB | 31 GB |
| Korean output | ✅ | ✅ | ✅ | ✅ | ✅ |
| Tool call JSON | ✅ | ✅ | ✅ | ✅ | ✅ |

### Mac mini M4 16 GB — boundary case

| Metric | IQ3_M | Q3_K_M |
|--------|------:|-------:|
| Token generation | ~0.3 t/s | ~0.30 t/s |
| Usable? | ⚠ Very slow (swap) | ⚠ Very slow (swap) |

26B MoE on 16 GB Macs hits swap (model + macOS overhead exceeds 16 GB). **Recommendation: 24 GB+ Mac**. For 16 GB devices, see the smaller Gemma 4 E series below.

## RAM Requirements

| Your Mac RAM | IQ3 (12 GB) | IQ4 (13 GB) | Q3 (13 GB) | Q4 (16 GB) | Q6 (21 GB) |
|--------------|------------:|------------:|-----------:|-----------:|-----------:|
| 16 GB | ❌ swap | ❌ swap | ❌ swap | ❌ won't fit | ❌ won't fit |
| 24 GB | ✅ fast | ✅ fits | ⚠ tight | ❌ barely | ❌ no |
| 32 GB | ✅ fast | ✅ fast | ✅ fast | ✅ OK | ❌ no |
| 36 GB+ | ✅ fast | ✅ fast | ✅ fast | ✅ fast | ✅ fits |
| 128 GB | 77 t/s | **85.8 t/s** | 70.7 t/s | 74.9 t/s | 74.8 t/s |

## Why no Q2? (benchmark evidence)

We tested Q2_K extensively. It produces broken, unusable output on 26B MoE — infinite repetition loops. At 2-bit precision the MoE expert routing weights lose too much information. **Q3 (3-bit) is the minimum viable quantization for this model.**

## Why BatiAI?

| | BatiAI | Third-party (unsloth, etc.) |
|---|---|---|
| **Source** | Quantized directly from [official Google weights](https://huggingface.co/google/gemma-4-26B-A4B-it) | Often re-quantized from other GGUFs |
| **Compatibility** | ✅ Verified on Ollama 0.19 ~ 0.20+ | ❌ Known issues with Ollama 0.20+ ([#15235](https://github.com/ollama/ollama/issues/15235)) |
| **Tested on** | Real Mac mini M4 (16 GB) + MacBook Pro M4 Pro/Max | Often untested on consumer hardware |
| **Tool calling** | ✅ Verified with BatiFlow's 57 tool functions | Often broken on MoE models |
| **Korean** | ✅ Validated | Not tested |
| **imatrix** | ✅ IQ3/IQ4 with calibration | UD- prefix custom format |
| **Multimodal** | ✅ Vision mmproj available (this repo) | Often missing |
| **Signing** | `general.author: BatiAI` for provenance | Varies |

## About BatiFlow

<p align="center">
  <a href="https://flow.bati.ai"><strong>flow.bati.ai</strong></a>
</p>

[BatiFlow](https://flow.bati.ai) is a **macOS-native AI desktop automation app** — just 5 MB, built with Swift.

- **Free & Unlimited** — On-device AI via Ollama, no API costs
- **100 % Private** — All data stays on your Mac
- **Ultra Lightweight** — Native macOS app, only 5 MB
- **57 built-in tools** — calendar, notes, reminders, files, email, browser, messaging, and more

<p align="center">
  <a href="https://flow.bati.ai">
    <img src="https://img.shields.io/badge/Download-BatiFlow-blue?style=for-the-badge&logo=apple" alt="Download BatiFlow">
  </a>
</p>

## Related models in the BatiAI lineup

| Family | Use case | Repo |
|--------|----------|------|
| Gemma 4 E2B-it | Tiny + audio support | [batiai/Gemma-4-E2B-it-GGUF](https://huggingface.co/batiai/Gemma-4-E2B-it-GGUF) |
| Gemma 4 E4B-it | Small + audio support | [batiai/Gemma-4-E4B-it-GGUF](https://huggingface.co/batiai/Gemma-4-E4B-it-GGUF) |
| **Gemma 4 26B-A4B-it** | **MoE chat (this repo)** | — |
| Gemma 4 31B-it | Dense, larger context for desktop | [batiai/Gemma-4-31B-it-GGUF](https://huggingface.co/batiai/Gemma-4-31B-it-GGUF) |
| Qwen 3.6 35B-A3B | Newer-gen MoE alternative | [batiai/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/batiai/Qwen3.6-35B-A3B-GGUF) |

## Technical Details

- **Original Model**: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **Architecture**: Gemma 4 Mixture-of-Experts (26 B total, ~3.8 B active per token)
- **Modalities**: Text (primary) + Image / Video via opt-in mmproj. Audio not supported (use E2B / E4B for audio)
- **Context Window**: 128 K tokens
- **License**: Apache 2.0
- **Quantized with**: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- **Quantized by**: [BatiAI](https://bati.ai)
- **GGUF metadata**: `general.author: BatiAI`, `general.url: https://flow.bati.ai`

## How we quantize

```
Google official weights (BF16, ~50 GB)
  ↓ llama.cpp convert_hf_to_gguf.py
BF16 GGUF (~50 GB)
  ↓ llama-imatrix (calibration data)
Importance Matrix (imatrix.dat)
  ↓ llama-quantize  (Q3_K_M, Q4_K_M, Q6_K)
  ↓ llama-quantize --imatrix  (IQ3_M, IQ4_XS)
Quantized GGUF
  ↓ convert_hf_to_gguf.py --mmproj  (vision tower)
mmproj BF16 → Q6_K
  ↓ scripts/sign-batiai.py  (general.author = BatiAI)
Signed GGUF
  ↓ ollama push (text-only)  +  hf upload (text + mmproj)
Published
```

No third-party intermediaries. Direct from source, verified on real hardware.

## License

This model is quantized from `google/gemma-4-26B-A4B-it` under the **Apache 2.0** license. Commercial use permitted per upstream terms.

BatiAI's quantization pipeline is provided under MIT.

## Benchmarks

<!-- BENCH-START -->

| Machine | Quant | Cold start | Prompt eval | Token gen | Tested |
|---------|-------|------------|-------------|-----------|--------|
| MacBook Pro M4 Max 128GB | IQ3_XXS | 3.915s | 1768.99 t/s | 83.63 t/s | 2026-05-03 |
| MacBook Pro M4 Max 128GB | IQ4_XS | 6.96s | 1786.7 t/s | 89.66 t/s | 2026-05-03 |
| MacBook Pro M4 Max 128GB | Q3_K_M | 5.203s | 1776.86 t/s | 78.02 t/s | 2026-05-03 |
| MacBook Pro M4 Max 128GB | Q4_K_M | 5.036s | 1897.1 t/s | 87.56 t/s | 2026-05-03 |
| MacBook Pro M4 Max 128GB | Q6_K | 6.416s | 1281.27 t/s | 74.33 t/s | 2026-05-03 |

<!-- BENCH-END -->
