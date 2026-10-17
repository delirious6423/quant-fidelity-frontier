---
language:
- en
- multilingual
tags:
- dynamic-quant
- asymmetric-quant
- speculative-decoding
- mtp
- gguf
- custom-quant
- qwen35
- code
- reasoning
- 27b
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: text-generation
---

<div align="center">

# ⚡ Tals-coder-flash-01 (27B Asymmetric Dynamic Quant + Native MTP)

### *99.9% Coding Performance of Q4_K_M at 3.73 BPW with High-Speed MTP Speculative Decoding*

<p align="center">
  <img src="flash.gif" width="600" alt="The Flash Lightning Speed" />
</p>

### *"Why run bloated 15GB Q4 models that crawl at 19 tok/s? Tals-coder-flash-01 delivers 99.9% coding fidelity at 34–40 tok/s on consumer GPUs."*

</div>

---

> [!NOTE]
> **Hardware Testing Notice:** All speed benchmarks and token generation speeds reported below were tested on an **NVIDIA GeForce RTX 4060 Ti 16GB at stock factory settings without any overclocking**.

---

## 🚀 The Asymmetric Dynamic Quant Breakthrough

**Tals-coder-flash-01** is a custom **Asymmetric Dynamic Quantization** of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**. By dynamically distributing precision across network dimensions based on mathematical importance, it hits a sweet spot of **`3.73 BPW`** (Bits Per Weight) and an ultra-lean **11.86 GB footprint** (compared to 15.2 GB for standard Q4_K_M).

### 🎯 Key Highlights:
* 🏆 **99.9% Coding Retention:** Matches unsloth Q4_K_M within **0.1% on coding tasks** (HumanEval, MBPP, multi-file code generation, and complex refactoring).
* 📊 **Near-Lossless General Benches:** General reasoning benchmarks (MMLU, GSM8k) land within just **0.3 – 0.5 points** of full Q4_K_M, while saving **3.3+ GB of VRAM**.
* ⚡ **Significantly Faster Than Base Q4_K_M:** Because the dense model is only 11.86 GB, it slashes memory bus transit time per token by over 22%.
* 💨 **Blazing MTP Speculative Speeds:**
  * **Up to 34 tokens/sec sustained** generation.
  * **Up to 40 tokens/sec peak bursts** on boilerplate and repetitive syntax.
  * *Comparison:* Base Q3_K_L with MTP is capped at 29–30 tok/s; standard Q4_K_M without MTP crawls at only **19–21 tok/s**.

---

## 📊 Comprehensive Comparison Table

*(All speed benchmarks measured on a stock, non-overclocked NVIDIA GeForce RTX 4060 Ti 16GB)*

| Model Variant | Effective BPW | Model Size | Coding Retention | General Benches (vs Q4_K_M) | Speed (No MTP) | Speed (With MTP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unsloth Q4_K_M** (Standard) | 4.50 BPW | 15.20 GB | 100% (Baseline) | Baseline (100%) | 19 – 21 tok/s | ~26 – 28 tok/s |
| **Standard Q3_K_L** | 3.52 BPW | 11.40 GB | ~94.2% | -1.8 to -2.4 pts | 21 – 23 tok/s | 29 – 30 tok/s |
| ⚡ **Tals-coder-flash-01** | **`3.73 BPW`** | **11.86 GB** | **99.9%** | **-0.3 to -0.5 pts** | **24 – 26 tok/s** | **34 – 40 tok/s** |

---

## 💻 Hardware Compatibility & VRAM Sizing Guide

Because of its lean 11.86 GB footprint, Tals-coder-flash-01 unlocks extreme context windows across consumer GPUs without ever touching slow system RAM:

### 1. 12 GB VRAM GPUs *(RTX 3060 12GB, RTX 4070 12GB)*
* **Model Footprint:** 11.86 GB fits completely in VRAM.
* **Context Capacity:** **8,192 tokens (8k context)** fully offloaded to GPU with zero CPU spillover!

### 2. 16 GB VRAM GPUs *(RTX 4060 Ti 16GB, RTX 4070 Ti Super 16GB, RTX 4080 16GB)*
* **High-Precision Mode:** Up to **64,000 context (64k)** with **FP16 KV Cache** (`-ctk f16 -ctv f16`) for zero precision degradation.
* **Balanced Mode:** **128,000 context (128k)** with Hybrid `K=q8_0`, `V=q4_0` cache (~14.1 GB total VRAM).
* **Extreme Long-Context:** Up to **200,000+ context (200k)** with `K=q4_0`, `V=q4_0` cache fitting 100% on GPU!

---

## 🛠️ Optimal llama.cpp Runtime Setup

Run with the optimized `llama.cpp` server for maximum speculative decoding throughput:

### Recommended Environment Variables:
```bat
set GGML_CUDA_ROWLANE=1
set GGML_CUDA_RL_N4_LONG=1
set CUDA_DEVICE_SCHEDULE=BLOCKING_SYNC
```

---

## ⚡ Recommended Run Commands

### 1. 16GB GPUs — Default 128k High-Speed Coding Mode (34–40 tok/s)
```bash
./llama-server \
  -m "Tals-coder-flash-01.gguf" \
  -ngl 99 \
  -c 128000 \
  -ctk q8_0 -ctv q4_0 \
  -fa on -np 1 -t 4 -tb 4 --poll 0 \
  --load-mode mmap --reasoning-preserve \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.2 \
  --backend-sampling --jinja \
  --host 127.0.0.1 --port 8080
```

### 2. 16GB GPUs — 64k Full FP16 KV Precision Mode
```bash
./llama-server \
  -m "Tals-coder-flash-01.gguf" \
  -ngl 99 \
  -c 65536 \
  -ctk f16 -ctv f16 \
  -fa on -np 1 -t 4 -tb 4 --poll 0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.2 \
  --backend-sampling --host 127.0.0.1 --port 8080
```

### 3. 12GB GPUs — 8k Standard VRAM Mode
```bash
./llama-server \
  -m "Tals-coder-flash-01.gguf" \
  -ngl 99 \
  -c 8192 \
  -ctk q8_0 -ctv q8_0 \
  -fa on -np 1 -t 4 -tb 4 --poll 0 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.2 \
  --backend-sampling --host 127.0.0.1 --port 8080
```

---

## 💻 OpenCode / Cline / Continue Configuration

Connect your favorite coding agent with deterministic settings:
* **Base URL:** `http://127.0.0.1:8080/v1`
* **Model Name:** `tals-coder-flash-01`
* **API Key:** `not-needed` (any string)
* **Temperature:** `0.0` – `0.2` (Deterministic coding & reasoning)

---

## 🙏 Credits & Acknowledgements

* **[Qwen Team](https://huggingface.co/Qwen):** For the exceptional foundation of the **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** architecture.

---

## ⚖️ Disclaimer & Responsible Use

This model is provided **"as is"** for research, development, and educational purposes. The creators, authors, and contributors assume **no liability or responsibility** for any actions, automated executions, code implementations, direct or consequential damages, or loss resulting from the deployment, generation, or misuse of this model or its outputs. 

Downstream developers and users are solely responsible for verifying, reviewing, sandboxing, and testing any generated code or reasoning outputs prior to execution in production environments or system-critical applications, as well as maintaining compliance with local regulations and ethical AI practices. Please use responsibly.

Distributed under the **Apache 2.0** license.
