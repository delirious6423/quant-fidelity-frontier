---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags:
- gemma4
- moe
- gguf
- llama.cpp
- quantized
- heterogeneous-inference
library_name: gguf
language:
- en
- multilingual
pipeline_tag: text-generation
---

# Gemma 4 26B-A4B Instruct — GGUF (Q3_K_M)

This repository provides a **Q3_K_M GGUF** quantization of `google/gemma-4-26B-A4B-it`, optimized for heterogeneous CPU+GPU inference on constrained hardware.

Per the analysis in *Heterogeneous Inference Architecture: Achieving Sustained 10+ Tokens per Second with Gemma 4 26B-A4B on Constrained Legacy Hardware*, this specific quantization tier is the mathematically optimal balance for deployment across an NVIDIA GTX 1070 (8GB VRAM) paired with 48GB DDR4 system RAM — transforming legacy hardware into a viable frontier-model inference platform.

---

## Model Architecture

| Property | Value |
|---|---|
| **Total Parameters** | 26 billion |
| **Active Parameters per Token** | ~3.8 billion (A4B = Active 4 Billion) |
| **Architecture** | Mixture-of-Experts (MoE) |
| **Experts** | 128 routed + 1 shared |
| **Active Experts per Token** | 8 of 128 (6.25%) |
| **Total Layers** | 30 |
| **Attention Type** | Hybrid: 25 sliding-window (1024 tok) + 5 global |
| **Context Window** | 256,000 tokens |
| **Release** | Google DeepMind, April 2, 2026 |

The "A4B" designation is critical: while the model stores 26B parameters (knowledge capacity), it only *computes* ~3.8B active parameters per token generation cycle. This architectural decoupling is what makes 10+ t/s inference viable on hardware that would otherwise be completely unsuitable.

---

## Quantization

| Format | File Size | Notes |
|---|---|---|
| BF16 / FP16 | ~50.5 GB | Full precision — exceeds 48GB RAM, unusable on target hardware |
| Q8_0 | ~26.9 GB | Near-lossless but too large for meaningful VRAM offload |
| **Q3_K_M** | **~13.3 GB** | **Optimal — fits the full heterogeneous memory map** |
| Q2_K | ~11.0 GB | Destroys MoE routing, causes incoherent/looping output |

**Why not Q2_K:** At 2-bit precision, the MoE expert router degrades catastrophically, producing repetitive loops and broken multilingual output. Q3_K_M is the minimum viable quantization that preserves routing integrity.

**Why not Unsloth Dynamic (UD) quants:** UD formats dynamically cast tensors to FP16. On Pascal architecture GPUs (GTX 1070), this triggers the 1:64 FP16 penalty, causing severe speed regressions. Use standard K-quants (this repo) instead.

---

## Heterogeneous Deployment — GTX 1070 + i7-6700HQ

The MoE architecture enables a unique asymmetric memory split:

| Component | Destination | Size | Bandwidth |
|---|---|---|---|
| Attention, shared expert, router | GTX 1070 VRAM | ~4.5 GB | 256 GB/s |
| KV Cache (q8_0) | GTX 1070 VRAM | ~1.5 GB | 256 GB/s |
| 128 routed expert pool | DDR4 RAM | ~8.8 GB | ~18 GB/s |

**Key insight:** Only 8 of 128 experts activate per token (~0.55 GB read from RAM per step), not the full 8.8 GB pool. This reduces the CPU memory bandwidth burden from a hard cap of ~1.3 t/s (loading full model every token) to a theoretical ~18+ t/s.

### Optimal Launch Command (llama.cpp / llama-server)

```bash
./llama-server \
  -m Gemma-4-26B-A4B-it.Q3_K_M.gguf \
  -c 16384 \
  -ngl 999 \
  -ot "exps=CPU" \
  -t 4 \
  --mlock \
  --no-mmap \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --flash-attn on \
  --prompt-lookup-decoding
```

**Flag breakdown:**

| Flag | Purpose |
|---|---|
| `-ngl 999` | Push all possible layers to VRAM |
| `-ot "exps=CPU"` | Override: force all routed expert tensors to CPU RAM |
| `-t 4` | Match thread count to physical cores (not hyperthreads) — prevents DDR4 bandwidth contention |
| `--mlock` | Pin model in RAM — prevents swap thrashing on the 13.3GB expert pool |
| `--no-mmap` | Load weights fully into RAM at startup — eliminates SATA/NVMe latency during expert fetching |
| `--cache-type-k q8_0 --cache-type-v q8_0` | Halve KV cache VRAM footprint with negligible quality loss |
| `--flash-attn on` | Fused attention for dense layers on VRAM |
| `--prompt-lookup-decoding` | N-gram speculative decoding — zero memory overhead, ~1.1–1.4× speedup on structured tasks |

---

## Throughput Mathematics

At Q3_K_M with asymmetric MoE offload:

- **GPU time per token:** 4.5 GB ÷ 256 GB/s ≈ **17.6 ms**
- **CPU time per token:** 0.55 GB active experts ÷ 18 GB/s ≈ **30.6 ms**
- **Total per token:** ~48 ms → **~20.8 t/s theoretical maximum**

Real-world sustained throughput with OS overhead, cache latency, and thermal throttling on a mobile Skylake i7: **10–15 t/s** — comfortably above the 10 t/s operational target.

Without MoE offload (naive CPU-only): 13.3 GB ÷ 18 GB/s = ~1.3 t/s maximum.

---

## Pascal GPU Warning (GTX 1070 / 1080 / 10xx)

Pascal (compute capability 6.1) has a **1:64 FP16 throughput penalty** vs FP32. Always use K-quant GGUFs (this repo) which execute final matmuls in FP32 via integer decompression. Never use FP16 or UD-format quants on Pascal hardware.

---

## Speculative Decoding

For additional throughput, pair with `--prompt-lookup-decoding` (included above) or use a draft model. Tested draft models:

| Draft Model | Task | Acceptance Rate | Speedup |
|---|---|---|---|
| `DuoNeural/Archon-Gemma-4-E4B` | Agentic Coding | ~77% | 1.40× |
| `DuoNeural/Archon-Gemma-4-E4B` | Code Generation | ~68% | 1.31× |
| `DuoNeural/Archon-Gemma-4-E4B` | Prose/Dialogue | ~62% | 1.13× |

> **Note:** Loading a discrete draft model on constrained hardware (8GB VRAM + 48GB RAM) risks memory contention. On GTX 1070 + i7-6700HQ specifically, use `--prompt-lookup-decoding` instead — zero memory overhead with meaningful gains on structured/code tasks.

---

## Usage with Ollama

```bash
ollama run hf.co/DuoNeural/Gemma-4-26B-A4B-it-GGUF:Q3_K_M
```

## Usage with llama.cpp CLI

```bash
./llama-cli \
  -m Gemma-4-26B-A4B-it.Q3_K_M.gguf \
  -ngl 999 -ot "exps=CPU" \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -p "You are a helpful assistant." \
  --prompt-lookup-decoding \
  -i
```

---

*Quantized by DuoNeural · April 2026 · Model weights: Google Gemma Terms of Use*

---

## DuoNeural

**DuoNeural** is an open AI research lab — human + AI in collaboration.

| | |
|---|---|
| 🤗 HuggingFace | [huggingface.co/DuoNeural](https://huggingface.co/DuoNeural) |
| 🐙 GitHub | [github.com/DuoNeural](https://github.com/DuoNeural) |
| 🐦 X / Twitter | [@DuoNeural](https://x.com/DuoNeural) |
| 📧 Email | duoneural@proton.me |
| 📬 Newsletter | [duoneural.beehiiv.com](https://duoneural.beehiiv.com) |
| ☕ Support | [buymeacoffee.com/duoneural](https://buymeacoffee.com/duoneural) |
| 🌐 Site | [duoneural.com](https://duoneural.com) |

### Research Team
- **Jesse** — Vision, hardware, direction
- **Archon** — AI lab partner, post-training, abliteration, experiments
- **Aura** — Research AI, literature synthesis, novel proposals

*Raw updates from the lab: model drops, training results, findings. Subscribe at [duoneural.beehiiv.com](https://duoneural.beehiiv.com).*

### DuoNeural Research Publications

| Title | DOI |
|-------|-----|
| [Nano-CTM: Ternary Continuous Thought Machines with Thought-Space Self-Prediction for Efficient Iterative Reasoning](https://doi.org/10.5281/zenodo.19775622) | [10.5281/zenodo.19775622](https://doi.org/10.5281/zenodo.19775622) |
| [Recurrence as World Model: CTM Learns Implicit Belief States in Partially Observable Physical Environments](https://doi.org/10.5281/zenodo.19810620) | [10.5281/zenodo.19810620](https://doi.org/10.5281/zenodo.19810620) |
| [Per-Object Slot Decomposition for Scalable Neural World Modeling: When Does Attention Beat Mean-Field?](https://doi.org/10.5281/zenodo.19846804) | [10.5281/zenodo.19846804](https://doi.org/10.5281/zenodo.19846804) |

*Open access, CC BY 4.0. Authored by Archon, Jesse Caldwell, Aura — DuoNeural.*

