---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: text-generation
base_model:
- google/gemma-4-26B-A4B-it
tags:
- gguf
- llama.cpp
- conversational
- on-device
- mobile
- korean
- korean-llm
- cpu
- local-llm
- edge
- gemma
- gemma4
- mixture-of-experts
- moe
- pocket
- vidraft
---

> ### ⭐ Latest and flagship models
> [![VKORE-46M-NPU](https://img.shields.io/badge/%F0%9F%86%95_VKORE--46M--NPU-pretraining_on_NPU-6c3fd1?style=for-the-badge)](https://huggingface.co/FINAL-Bench/VKORE-46M-NPU) [![Darwin-27B-ZTC-v5](https://img.shields.io/badge/Darwin--27B--ZTC--v5-S1MB_%231_EN+JP-ff6b8b?style=for-the-badge)](https://huggingface.co/FINAL-Bench/Darwin-27B-ZTC-v5)
> [![POCKET family](https://img.shields.io/badge/%F0%9F%A4%97-POCKET_family-ffb020)](https://huggingface.co/collections/FINAL-Bench/pocket-models-6a618ee5d23eafb7e185a5c6) [![Darwin family](https://img.shields.io/badge/%F0%9F%A4%97-Darwin_family-ffb020)](https://huggingface.co/collections/FINAL-Bench/darwin-family-699987b1f652864af0122193) [![All models](https://img.shields.io/badge/%F0%9F%A4%97-FINAL--Bench-ffb020)](https://huggingface.co/FINAL-Bench) [![Website](https://img.shields.io/badge/%F0%9F%8C%90-vidraft.net-4ea8ff)](https://vidraft.net) [![THE VC](https://img.shields.io/badge/%F0%9F%93%8A-THE_VC-0a66c2)](https://thevc.kr/vidraft)

> ### 🆕 **[POCKET-Qwen3.8-Flash-Next](https://huggingface.co/FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF)** — a **180B** model running on a **laptop with 8 GB VRAM + 32 GB RAM** · **4.17 tok/s measured**.
> [![New](https://img.shields.io/badge/🆕_NEW-POCKET--180B_on_a_laptop-6c3fd1)](https://huggingface.co/FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF) [![VRAM](https://img.shields.io/badge/VRAM-8_GB-1baf7a)]() [![RAM](https://img.shields.io/badge/RAM-32_GB-1baf7a)]() [![Speed](https://img.shields.io/badge/measured-4.17_tok%2Fs-243456)]() <!-- POCKET-FLASHNEXT-BADGE -->

> ### 🆕 **[POCKET-Zimage-CPU](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU)** — photoreal **images** in **46 s** on a **CPU only**. No GPU, no CUDA, no Python.
> [![New](https://img.shields.io/badge/🆕_NEW-POCKET--Zimage--CPU-ff6b35)](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU) [![Space](https://img.shields.io/badge/🤗_Space-try_on_CPU-ffce3a)](https://huggingface.co/spaces/FINAL-Bench/POCKET-Zimage-CPU) [![RAM](https://img.shields.io/badge/RAM-6.4_GB-1baf7a)]() <!-- POCKET-Zimage-CPU-BADGE -->


> ### 📚 Collections
> **▶ [POCKET Models](https://huggingface.co/collections/FINAL-Bench/pocket-models-6a618ee5d23eafb7e185a5c6)** — this family (on-device, no GPU)
> [Darwin Family](https://huggingface.co/collections/FINAL-Bench/darwin-family-699987b1f652864af0122193) · [Aether Foundation](https://huggingface.co/collections/FINAL-Bench/aether-foundation-model-6a5c7f2fa1a4165c0414e53a) · [VKAE Accelerated](https://huggingface.co/collections/FINAL-Bench/vkae-accelerated-6a47231d7e7999dd8227675a)

![POCKET-26B](./pocket26_hero.svg)

# POCKET-26B-GGUF · 한국어

### A **Gemma4-26B-A4B**-based pocket model that loads in **any app today** — Ollama, LM Studio, PocketPal — with **no bleeding-edge runtime** needed. Korean-tuned, GPU-optional.

> 🚀 **Try it live on a CPU (no GPU), no install →** [![POCKET-26B demo](https://img.shields.io/badge/🤗_Space-POCKET--26B_CPU_chat-0f9d6e)](https://huggingface.co/spaces/FINAL-Bench/POCKET-26B-CPU) [![POCKET-35B demo](https://img.shields.io/badge/🤗_Space-POCKET--35B_CPU_chat-ffce3a)](https://huggingface.co/spaces/FINAL-Bench/POCKET-35B-CPU)

[![License](https://img.shields.io/badge/License-Apache_2.0-0f6e56)](https://www.apache.org/licenses/LICENSE-2.0) [![Runtime](https://img.shields.io/badge/runtime-any_llama.cpp-1baf7a)](https://github.com/ggml-org/llama.cpp) [![Compat](https://img.shields.io/badge/loads_in-Ollama_·_LM_Studio_·_PocketPal-2a9d8f)]() [![Base](https://img.shields.io/badge/base-Gemma--4--26B--A4B-185fa5)](https://huggingface.co/google/gemma-4-26B-A4B-it)

**Pick your build →** [![35B](https://img.shields.io/badge/POCKET--35B-GGUF-243456)](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) [![26B](https://img.shields.io/badge/POCKET--26B-GGUF-2a9d8f)](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) [![KR GGUF](https://img.shields.io/badge/POCKET--KR-GGUF-7A1F3D)](https://huggingface.co/FINAL-Bench/POCKET-KR-GGUF) [![KR MLX](https://img.shields.io/badge/POCKET--KR-MLX_(iPhone)-0f6e56)](https://huggingface.co/FINAL-Bench/POCKET-KR-MLX) [![EN GGUF](https://img.shields.io/badge/POCKET--EN-GGUF-185fa5)](https://huggingface.co/FINAL-Bench/POCKET-EN-GGUF) [![180B laptop](https://img.shields.io/badge/POCKET--180B-Qwen3.8_Flash--Next_·_laptop-6c3fd1)](https://huggingface.co/FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF) [![Image NF4](https://img.shields.io/badge/POCKET--Image-Z--Image_NF4-d35400)](https://huggingface.co/FINAL-Bench/POCKET-Image-Zimage) [![Image CPU](https://img.shields.io/badge/POCKET--Zimage-CPU-ff6b35)](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU)

## Why this one?

POCKET-26B takes Google's **Gemma4-26B-A4B** (25.2B total, ~4B active MoE, Apache-2.0) and re-quantizes it with our **proprietary Korean-tuned quantization** — **unpruned**, so quality holds. Unlike our Qwen-based POCKET (which needs a very recent `llama.cpp` build for its `qwen35moe` architecture), Gemma4 loads in **every mainstream runtime today**: Ollama, LM Studio, PocketPal, koboldcpp, and the browser.

## Quality — GPQA-Diamond, greedy, 198 questions (our harness)

| Build | GPQA-Diamond | vs base |
|---|---|---|
| Gemma4-26B-A4B (base) | 67.7% | — |
| **POCKET-26B `Q4_K_M`** | **67.7%** | **= base (lossless)** |
| **POCKET-26B `Q2_K`** (mixed) ⭐ | **67.2%** | −0.5pp (≈ lossless) |

*Single greedy pass, 198 items → ±~3 pp noise. Our proprietary Korean-tuned quantization is statistically lossless vs the base.*

## Files in this repo

| File | Size | Runs on | Best for |
|---|---|---|---|
| `POCKET-26B-Q4_K_M.gguf` | 17 GB | PC / high-RAM | top quality |
| **`POCKET-26B-Q2_K.gguf`** ⭐ | 11 GB | 12 GB phone / PC / browser | **universal daily driver** |

> Our **mixed-precision quantization** keeps the most quality-critical weights at higher precision — that is why `Q2_K` holds **67.2%** while a *plain* uniform Q2 collapses to ~44%.

## Quickstart — loads anywhere

```bash
# stock llama.cpp — brew / winget / apt, or LM Studio / Ollama / PocketPal
llama-cli -m POCKET-26B-Q2_K.gguf -p "대한민국의 수도는?" -ngl 0 -t 8
```
No fork, no bleeding-edge build — Gemma4 support has shipped in every mainstream runtime since April 2026.

## Lineage (honest)

Based on **[google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)** (Apache-2.0). We do **not** re-host it unchanged — we add our proprietary **Korean-tuned quantization** (VIDRAFT). We deliberately **do not prune** it: Gemma4's low-bit robustness collapses under pruning (measured), so we keep all 128 experts and win on quality + universal compatibility instead.

## Limitations

- For **8 GB phones** (~5 GB budget), use [POCKET-KR-GGUF](https://huggingface.co/FINAL-Bench/POCKET-KR-GGUF) (5.1 GB) — Gemma4 cannot be shrunk that far without collapse.
- On-device iPhone/Mac throughput **not yet measured by us** — community reports welcome.

## Learn more

- Why on-device LLMs matter, and how POCKET measures up: [Can you run a large LLM without a GPU?](https://vidraft.net/insights/on-device-llm-without-gpu.html)
- What model quantization is, and why a 4-bit model stays smart: [What is model quantization?](https://vidraft.net/insights/what-is-quantization-llm.html)

## License
Apache-2.0 — use, modify, redistribute freely.

---
*POCKET is a VIDRAFT model family. Runs anywhere, no GPU.*

<!-- POCKET-FAMILY -->

---

## 🧩 The POCKET Family — On-device AI by VIDRAFT

*Big models, small hardware. No GPU, no cloud.*

**Models**
- 📦 [POCKET-35B-GGUF](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) — flagship, PC / server, no GPU
- 📦 [POCKET-26B-GGUF](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) — compact 26B
- 🇰🇷 [POCKET-KR-GGUF](https://huggingface.co/FINAL-Bench/POCKET-KR-GGUF) — Korean, Android
- 🍎 [POCKET-KR-MLX](https://huggingface.co/FINAL-Bench/POCKET-KR-MLX) — Korean, iPhone / Mac
- 🌍 [POCKET-EN-GGUF](https://huggingface.co/FINAL-Bench/POCKET-EN-GGUF) — English, phone / PC
- 💻 [POCKET-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF) — **180B on a laptop** (8 GB VRAM + 32 GB RAM)
- 🖼️ [POCKET-Image-Zimage](https://huggingface.co/FINAL-Bench/POCKET-Image-Zimage) — character-perfect text in any image
- 🖥️ [POCKET-Zimage-CPU](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU) — photoreal images on a CPU only

**Demos & tools (Spaces)**
- 🎨 [POCKET-Image Studio](https://huggingface.co/spaces/FINAL-Bench/POCKET-Image-Studio) — text-in-image, generate in-page
- 🖥️ [POCKET-35B-CPU](https://huggingface.co/spaces/FINAL-Bench/POCKET-35B-CPU) — 35B answering on a CPU
- 🖥️ [POCKET-26B-CPU](https://huggingface.co/spaces/FINAL-Bench/POCKET-26B-CPU) — 26B on a CPU
- 🖼️ [POCKET-Zimage-CPU](https://huggingface.co/spaces/FINAL-Bench/POCKET-Zimage-CPU) — image generation on a CPU

📚 [Full POCKET collection](https://huggingface.co/collections/FINAL-Bench/pocket-models-6a618ee5d23eafb7e185a5c6)

<!-- /POCKET-FAMILY -->
