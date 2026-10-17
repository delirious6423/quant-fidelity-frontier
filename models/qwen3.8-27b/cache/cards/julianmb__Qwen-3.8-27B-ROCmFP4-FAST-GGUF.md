---
language:
- en
- zh
license: apache-2.0
library_name: gguf
tags:
- qwen
- qwen3
- 27b
- rocm
- vulkan
- strix-halo
- rdna4
- amd
- fp4
- speculative-decoding
- mtp
- turboquant
- ryzen-ai
pipeline_tag: text-generation
inference: false
model_name: Qwen 3.8 27B ROCmFP4 FAST
base_model: Qwen/Qwen3.8-27B
quantized_by: julianmb
model-index:
- name: Qwen 3.8 27B ROCmFP4 FAST
  results:
  - task:
      type: text-generation
      name: Strix Halo MTP Speculative Decoding
    dataset:
      name: Strix Halo LLM Benchmark Suite
      type: synthetic
    metrics:
    - name: Peak Speculative Decode Speed
      type: tokens_per_second
      value: 36.04
    - name: Sustained MTP Decode Speed (K=4)
      type: tokens_per_second
      value: 33.80
    - name: Strict Lossless Greedy MTP Speed
      type: tokens_per_second
      value: 34.82
    - name: Base Unassisted Decode Speed
      type: tokens_per_second
      value: 14.02
    - name: Prompt Evaluation Latency (TTFT)
      type: milliseconds
      value: 439.4
    - name: Hybrid NPU Burst TTFT (Long Prompt)
      type: milliseconds
      value: 870.0
    - name: Effective Bits Per Weight
      type: bpw
      value: 4.26
---

# Qwen 3.8 27B ROCmFPX (GGUF for AMD Radeon GPUs)

[![HaloFPX Server](https://img.shields.io/badge/Server-julianmb%2Fhalofpx-009688?logo=github)](https://github.com/julianmb/halofpx)
[![Qwen 3.8 Project](https://img.shields.io/badge/Qwen_3.8_Project-julianmb%2Fq38rocm-blue?logo=github)](https://github.com/julianmb/q38rocm)
[![Hardware](https://img.shields.io/badge/Hardware-AMD_Strix_Halo_%26_Radeon_GPUs-ED1C24?logo=amd)](https://www.amd.com)
[![Vulkan](https://img.shields.io/badge/Driver-Mesa_RADV_Wave64-FF5722?logo=vulkan)](https://mesa3d.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

This repository contains optimized **ROCmFP4** and **ROCmFP8** GGUF releases of **Qwen 3.8 27B**, custom-engineered for **AMD Strix Halo (Ryzen AI Max+ 395 / Radeon 8060S)** and **AMD Radeon discrete GPUs**.

> 🛠️ **Unified HaloFPX Server:** [github.com/julianmb/halofpx](https://github.com/julianmb/halofpx) (Multi-model zoo, hot-swapping, OpenAI API)  
> 📖 **Qwen 3.8 Dedicated Deep-Dive:** [github.com/julianmb/q38rocm](https://github.com/julianmb/q38rocm) (Standalone optimization report, benchmarks & tuning)

---

## 📦 Available GGUF Quantizations

| File Name | Precision | File Size | Raw Decode | MTP Speculative Decode | Best For |
|---|---|---|---|---|---|
| **`Qwen3.8-27B-ROCmFP4-FAST.gguf`** | **4.26 bpw** | **13.55 GiB** | 14.02 tok/s | 🔥 **30.56 – 36.04 tok/s** | **Maximum Speed (Gold Standard, Fits 16GB+ GPUs)** |
| **`Qwen3.8-27B-ROCmFP8.gguf`** | **8.25 bpw** | **26.25 GiB** | 7.66 tok/s | **18.96 tok/s** | **Maximum Precision (<0.003 PPL loss vs FP16)** |
| **`Qwen3.8-27B-ROCmFP4-STRIX_LEAN.gguf`** | **4.34 bpw** | **13.82 GiB** | 13.90 tok/s | **30.00 – 34.00 tok/s** | FP16 embeddings & norm preservation |
| **`Qwen3.8-27B-Q3_K_S.gguf`** | **3.59 bpw** | **11.40 GiB** | 16.69 tok/s | **20.44 – 26.11 tok/s** | Fastest unassisted decode |
| **`Qwen3.8-27B-Q3_K_M.gguf`** | **3.95 bpw** | **12.56 GiB** | 15.15 tok/s | **25.00 – 28.50 tok/s** | Balanced 3-bit medium quantization |
| **`Qwen3.8-27B-ROCmFP2.gguf`** | **2.69 bpw** | **8.56 GiB** | 12.82 tok/s | **17.50 – 19.00 tok/s** | Ultra-compact memory footprint |

---

## 🔬 Why ROCmFP4 & ROCmFP8 Improve Performance on AMD Hardware

In auto-regressive decoding, generation speed is physically bound by memory bandwidth:
- **Bandwidth Math:** Strix Halo's 256-bit memory controller delivers ~190–200 GB/s sustained read bandwidth. Loading an unquantized FP16 model (54.6 GB) caps generation at **~4.9 tok/s**.
- **8-Bit (ROCmFP8 - 26.25 GB):** Doubles memory streaming throughput to **18.96 tok/s** with MTP speculation while providing **lossless precision (<0.003 PPL loss vs FP16)**.
- **4-Bit (ROCmFP4 - 13.55 GB):** Slashes memory payload by **75.2% vs FP16**, raising unassisted streaming decode to **14.02 tok/s** and MTP speculative throughput to **30.56 – 36.04 tok/s** (with **33.80 tok/s sustained decode at K=4**) with ~99% benchmark retention.

---

## 🧠 Optional: AMD XDNA 2 NPU Acceleration (Qwen 3.8 27B Only)

> ⚠️ **Scope note:** All NPU benchmarks below were **only tested on Qwen 3.8 27B** (dense, ROCmFP4_FAST).

Strix Halo features a **50 TOPS XDNA 2 NPU** (`/dev/accel/accel0`). Extensive empirical benchmarking demonstrates:
- **Sustained Decode Ceiling:** **33.8 tok/s** via embedded MTP (iGPU only) is the practical ceiling. The NPU does *not* improve sustained decode speed.
- **Proven NPU Value:** **1.8× faster first-token TTFT on long prompts (870 ms vs 1587 ms)** via hybrid NPU-burst → iGPU handoff (`scripts/run_pipeline.py`).
- **See [`docs/NPU_INTEGRATION.md`](https://github.com/julianmb/q38rocm/blob/main/docs/NPU_INTEGRATION.md)** for complete setup instructions.

---

## 📊 Performance Benchmarks (AMD Ryzen AI Max+ 395)

Measured on **AMD Ryzen AI Max+ 395 (40 CU Radeon 8060S @ 2.9 GHz, 128 GB 256-bit LPDDR5X, Linux 7.0, Mesa 26.0 RADV)**:

| Optimization Profile | Model Size | Unassisted Decode *(Measured)* | MTP Speculative Decode *(Measured)* | Speedup vs Baseline | TTFT (Prompt Eval) *(Measured)* |
|---|---|---|---|---|---|
| Stock `Q4_K_M` (Baseline) | 15.92 GiB | 12.27 tok/s | N/A | 1.00× | 526.7 ms |
| **`ROCmFP4_FAST` (This Model)** | **13.55 GiB** | **14.02 tok/s** | N/A | 1.14× | 468.3 ms |
| **`ROCmFP4_FAST` + Strict Greedy MTP** | **13.55 GiB** | 14.02 tok/s | **34.82 tok/s** | **2.84×** | 442.8 ms |
| **`ROCmFP4_FAST` + Sustained MTP (K=4)** | **13.55 GiB** | 14.02 tok/s | **33.80 tok/s sustained** | **2.40×** | 510.0 ms |
| **`ROCmFP4_FAST` + MTP (`n6/p0.60`)** | **13.55 GiB** | 14.02 tok/s | **30.56 – 34.82 tok/s** | **2.50× – 2.84×** | 439.4 ms |
| **`ROCmFP4_FAST` + Deep Spec (`n7/p0.35`)** | **13.55 GiB** | 14.02 tok/s | 🔥 **36.04 tok/s** *(JSON/Code)* | 🔥 **2.94×** | 445.8 ms |

---

## 💾 Context Scaling & Memory Footprint

Using **Asymmetric TurboQuant KV cache** (`-ctk q8_0 -ctv turbo4`):

| Context Window | Model Weights | TurboQuant KV Cache | Total RAM Footprint |
|---|---|---|---|
| **8K tokens** | 13.55 GiB | 0.62 GiB | **14.17 GiB** *(Fits 16GB VRAM)* |
| **16K tokens** | 13.55 GiB | 1.23 GiB | **14.78 GiB** *(Fits 16GB VRAM)* |
| **32K tokens** | 13.55 GiB | 2.45 GiB | **16.00 GiB** *(Ideal for 64GB Strix Halo)* |
| **64K tokens** | 13.55 GiB | 4.90 GiB | **18.45 GiB** |
| **128K tokens** | 13.55 GiB | 9.80 GiB | **23.35 GiB** |
| **262K tokens (Full)** | 13.55 GiB | 20.08 GiB | **33.63 GiB** |

---

## 📥 Quick Download

```bash
# Download ROCmFP4_FAST (13.55 GiB)
hf download julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF Qwen3.8-27B-ROCmFP4-FAST.gguf --local-dir .

# Or download ROCmFP8 (Zero-loss 8-bit precision, 26.25 GiB)
hf download julianmb/Qwen-3.8-27B-ROCmFP4-FAST-GGUF Qwen3.8-27B-ROCmFP8.gguf --local-dir .
```

---

## 🚀 How to Run

### Option 1: Via `HaloFPX` (Recommended Multi-Model Server)
```bash
git clone https://github.com/julianmb/halofpx.git
cd halofpx
pip install -e .

halofpx load qwen38-27b
```

### Option 2: Standalone via `q38rocm`
```bash
git clone https://github.com/julianmb/q38rocm.git
cd q38rocm

./quickstart.sh
```

---

## 🔒 Limitations & Safety
- **Custom Backend:** Requires the [ROCmFPX toolchain](https://github.com/charlie12345/ROCmFPX) (pinned build: `e87d53e (213)`).
- **Hardware Target:** Optimized for AMD Strix Halo (`gfx1151`) and AMD discrete Radeon GPUs.
- **Base Alignment:** Inherits base safety characteristics and knowledge capabilities of Qwen 3.8 27B.

---

## 📜 License & Attribution
- **Base Model:** [Qwen 3.8 27B by Alibaba Cloud](https://huggingface.co/Qwen)
- **Quantization & Optimizations:** Apache 2.0 License.
- **Community Research:** Multi-slot concurrency metrics referenced from *MrWidmoreHK* and *kujetic*.

## 📏 Quantization Fidelity (wikitext-2 validation)

Independent perplexity checks (`llama-perplexity`, ROCm0, ctx 512):

| Build | PPL ↓ |
|---|---|
| `ROCmFP4_FAST` (this repo, original build) | 7.205 ± 0.121 |
| `ROCmFP4_FAST` requantized from **unsloth Q8_0** | **7.138 ± 0.120** |

Speed is statistically identical between both builds (13.56 vs 13.64 tg128) — decode is bandwidth-bound. Requantizing any high-precision source (Q8_0/BF16, `--allow-requantize`) into ROCmFP4 presets is a validated recipe; avoid k-quant sources (double quantization).

## 🔬 Batch Scaling (why speculative decoding works here)

`llama-batched-bench`, tg128 per-sequence throughput, Vulkan0:

| Verify batch | tok/s/seq |
|---|---|
| 1 | 13.53 |
| 8 | **55.44** (4.1×) |

Speculative decoding converts memory-bandwidth headroom into tokens: the flat batch-cost curve is what makes MTP's ~2.5× sustained gain possible.
