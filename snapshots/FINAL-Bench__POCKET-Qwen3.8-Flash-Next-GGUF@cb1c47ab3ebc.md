---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: llama.cpp
pipeline_tag: text-generation
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- llama.cpp
- conversational
- on-device
- laptop
- local-llm
- consumer-gpu
- offload
- mixture-of-experts
- moe
- quantized
- pocket
- vidraft
- qwen3.8
- korean
---
> ### 🆕 **[POCKET-Zimage-CPU](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU)** — photoreal **images** in **46 s** on a **CPU only**. No GPU, no CUDA, no Python.
> [![New](https://img.shields.io/badge/🆕_NEW-POCKET--Zimage--CPU-ff6b35)](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU) [![Space](https://img.shields.io/badge/🤗_Space-try_on_CPU-ffce3a)](https://huggingface.co/spaces/FINAL-Bench/POCKET-Zimage-CPU) [![RAM](https://img.shields.io/badge/RAM-6.4_GB-1baf7a)]() <!-- POCKET-Zimage-CPU-BADGE -->

> ### 📚 Collections
> **▶ [POCKET Models](https://huggingface.co/collections/FINAL-Bench/pocket-models-6a618ee5d23eafb7e185a5c6)** — this family (on-device, no GPU)
> [Darwin Family](https://huggingface.co/collections/FINAL-Bench/darwin-family-699987b1f652864af0122193) · [Aether Foundation](https://huggingface.co/collections/FINAL-Bench/aether-foundation-model-6a5c7f2fa1a4165c0414e53a) · [VKAE Accelerated](https://huggingface.co/collections/FINAL-Bench/vkae-accelerated-6a47231d7e7999dd8227675a)

![POCKET 180B on a laptop](./pocket_flashnext_hero.svg)

# POCKET-Qwen3.8-Flash-Next-GGUF

### A **180B** model on a **laptop with an 8 GB GPU and 32 GB RAM**. Measured on a real consumer laptop, not simulated: **4.17 ± 0.17 tok/s**.

The file is **111 GiB**. That is **3.5× the laptop's RAM** and **14× its VRAM**, and it still runs, on stock `llama.cpp`, with no server and no cloud.

[![License](https://img.shields.io/badge/License-Qwen_Community_1.0-6c3fd1)](./LICENSE) [![Runtime](https://img.shields.io/badge/runtime-stock_llama.cpp-f0992a)](https://github.com/ggml-org/llama.cpp) [![VRAM](https://img.shields.io/badge/VRAM-8_GB-1baf7a)]() [![RAM](https://img.shields.io/badge/RAM-32_GB-1baf7a)]() [![Base](https://img.shields.io/badge/base-Qwen3.8--Flash--Next_(180B,_6B_active)-185fa5)](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

**Pick your build →** [![35B](https://img.shields.io/badge/POCKET--35B-GGUF-243456)](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) [![26B](https://img.shields.io/badge/POCKET--26B-GGUF-2a9d8f)](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) [![KR GGUF](https://img.shields.io/badge/POCKET--KR-GGUF-7A1F3D)](https://huggingface.co/FINAL-Bench/POCKET-KR-GGUF) [![KR MLX](https://img.shields.io/badge/POCKET--KR-MLX_(iPhone)-0f6e56)](https://huggingface.co/FINAL-Bench/POCKET-KR-MLX) [![EN GGUF](https://img.shields.io/badge/POCKET--EN-GGUF-185fa5)](https://huggingface.co/FINAL-Bench/POCKET-EN-GGUF) [![180B laptop](https://img.shields.io/badge/POCKET--180B-Qwen3.8_Flash--Next_·_laptop-6c3fd1)](https://huggingface.co/FINAL-Bench/POCKET-Qwen3.8-Flash-Next-GGUF) [![Image NF4](https://img.shields.io/badge/POCKET--Image-Z--Image_NF4-d35400)](https://huggingface.co/FINAL-Bench/POCKET-Image-Zimage) [![Image CPU](https://img.shields.io/badge/POCKET--Zimage-CPU-ff6b35)](https://huggingface.co/FINAL-Bench/POCKET-Zimage-CPU)

## 💻 The headline: 180B on the laptop you already own

`[measured — 2026-09-16, llama-bench, 2 repetitions each]`

| Machine | Samsung Galaxy Book6 Ultra (consumer laptop) |
|---|---|
| GPU | NVIDIA RTX 5060 Laptop, **8 GB** VRAM |
| RAM | **32 GB** (31.4 GB usable) |
| CPU | Intel Core Ultra 7 356H, 16 cores |
| Storage | NVMe SSD, 1.6 GB/s sequential read |
| Model | Qwen3.8-Flash-Next, **180B total / 6B active**, Q4_K_M, **111 GiB** |

| Build | Generate (tg256) | Prompt (pp64) | VRAM used |
|---|---|---|---|
| **`Q4_K_M`** · 8 threads | **4.17 ± 0.17 tok/s** | 1.02 ± 0.12 tok/s | 4.1 GB |
| `Q4_K_M` · 4 threads | 2.80 ± 0.56 | 1.18 ± 0.06 | 4.1 GB |
| `Q4_K_M-v2` · 8 threads | 2.93 ± 1.92 | 1.21 ± 0.03 | 3.7 GB |
| `Q4_K_M-v2` · 4 threads | 2.55 ± 0.14 | **2.32 ± 0.27** | 3.7 GB |

**How a 111 GiB model runs in 32 GB of RAM.** Only **6B of the 180B** parameters are active per token (10 of 512 experts). `llama.cpp` memory-maps the file: the always-used layers sit on the 8 GB GPU, the experts stay in RAM and on the NVMe SSD, and each token only touches the few experts it needs. **RAM decides the speed, not whether it runs.**

> 🔴 **Counter-intuitive, measured: do not fill the 8 GB GPU.** Pushing expert layers onto the GPU made it *slower* (4.1 GB → 7.9 GB VRAM: 3.10 → 1.75 tok/s). The driver starts paging VRAM to system RAM. The command below is already the sweet spot.

> **Honest scope.** About 4 tokens per second is reading speed for short answers, not chat speed for long ones. The point is that a frontier-scale **180B** model runs **locally, privately, on a laptop**. Prompt processing is slow (1–2 tok/s), so keep prompts short. On a 64 GB machine more of the model stays in RAM and it gets faster (not yet measured by us).

## Quickstart (laptop, 8 GB GPU)
```bash
# recent llama.cpp with Qwen3.8-Flash-Next support; point -m at the first shard
llama-cli -m Q4_K_M/POCKET-Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf \
  -ngl 99 -ot "exps=CPU" -t 8 -p "안녕하세요"
# reproduce our numbers (always use -r 2 or more; single runs swing a lot)
llama-bench -m Q4_K_M/POCKET-Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf -ngl 99 -ot "exps=CPU" -t 8 -p 64 -n 256 -r 2
```
`-ot "exps=CPU"` keeps the experts off the GPU. That is what makes it fit in 8 GB. Use physical cores for `-t`.

## Files in this repo
| Folder | Size | Notes |
|---|---|---|
| **`Q4_K_M/`** ⭐ | 111 GiB (3 shards) | fastest generation on the laptop (**4.17 tok/s**) |
| `Q4_K_M-v2/` | 93 GiB (3 shards) | **16% smaller**, 2× faster prompt processing at 4 threads; our proprietary bit allocation |

Both builds keep **every tensor at 4-bit or higher**. Text only: the base model's vision encoder and 4B MTP head are not included (the GGUF holds 177B of the 180B).

## Quality — measured, including where we do not win

`[measured]` Korean perplexity: Wikipedia-Korean (`20231101.ko`), exactly 80 chunks (40,960 tokens), `-c 512`, lower is better:

| Build | Size | Korean PPL |
|---|---|---|
| POCKET `Q4_K_M` | 111 GiB | 6.030 ± 0.103 |
| POCKET `Q4_K_M-v2` | 93 GiB | 6.080 ± 0.104 |
| unsloth `UD-IQ4_XS` (reference) | 88 GB | 6.013 ± 0.103 |

All three are **within error of each other**: no measurable quality difference. We do **not** claim a better quantization. What we add is the **laptop measurement itself**: to our knowledge nobody else has published a measured 180B run on an 8 GB / 32 GB consumer laptop.

## Where it fits in the POCKET lineup
| Your hardware | Pick |
|---|---|
| Phone (8–12 GB) | [POCKET-KR-MLX](https://huggingface.co/FINAL-Bench/POCKET-KR-MLX) · [POCKET-KR-GGUF](https://huggingface.co/FINAL-Bench/POCKET-KR-GGUF) · [POCKET-EN-GGUF](https://huggingface.co/FINAL-Bench/POCKET-EN-GGUF) |
| Mini-PC / PC, no GPU | [POCKET-35B-GGUF](https://huggingface.co/FINAL-Bench/POCKET-35B-GGUF) · [POCKET-26B-GGUF](https://huggingface.co/FINAL-Bench/POCKET-26B-GGUF) |
| **Laptop with 8 GB GPU + 32 GB RAM, want the biggest model** | **this repo, 180B** |

## Lineage
| Component | Origin |
|---|---|
| Base model | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (Qwen), 180B total, 6B active, 512 experts / 10 active |
| Quantization | GGUF `Q4_K_M` built by VIDRAFT from the BF16 weights; `-v2` uses our proprietary bit allocation |
| Runtime | upstream llama.cpp, unmodified |
| Measurement | VIDRAFT, on the laptop above |

## Limitations
- ~3–4 tok/s generation and 1–2 tok/s prompt processing on the 8 GB / 32 GB laptop. Usable for short, private questions, not for long chats.
- 93–111 GiB download; put it on a fast NVMe SSD (speed depends on it).
- Text only (no vision encoder, no MTP head).
- Needs a llama.cpp build new enough to support Qwen3.8-Flash-Next.

## License
Derived from **Qwen3.8-Flash-Next** under the **[Qwen Community License 1.0](./LICENSE)** (included in this repo). Note clause 2: a Model-as-a-Service or AI Work Assistant business needs a separate license from Qwen for commercial use.

---
*POCKET is a VIDRAFT model family. Big models, small hardware.*

## Learn more
- On-device LLMs without a GPU — and how POCKET measures up: [Can you run a large LLM without a GPU?](https://vidraft.net/insights/on-device-llm-without-gpu.html)
- What model quantization is, and why a 4-bit model stays smart: [What is model quantization?](https://vidraft.net/insights/what-is-quantization-llm.html)

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
