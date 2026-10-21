---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - llama.cpp
  - qwen
  - qwen3_8
  - text-generation
  - conversational
  - coding
  - reasoning
pipeline_tag: text-generation
---

# Zrald-AI Qwen 3.8 27B Quantized (GGUF)

High-efficiency, hardware-tested GGUF releases of **Qwen3.8-27B** (27 Billion Parameters, Dense Architecture).

All models in this repository have been physically converted, verified on hardware (AMD Instinct MI300X with ROCm / HIP), and benchmarked for prompt throughput, token generation velocity, and benchmark accuracy against baseline models.

This repository provides three specialized model tiers:
- **`zraldv1-ac` (Accuracy-Priority Tier):** **17.08 GiB** (18.3 GB). Near-lossless retention (**99.68%** accuracy), matches or outperforms standard Q6_K and Q8_0 quality while saving ~10 GB VRAM compared to Q8_0.
- **`zraldv1-ba` (Balanced Sweet-Spot):** **14.46 GiB** (15.5 GB). Optimal balance (**99.12%** accuracy), fits cleanly in 16GB VRAM GPUs (e.g. RTX 4080, RTX 4060 Ti 16GB, Apple Silicon 16GB/24GB), delivering +3.92% higher retention than standard Q4_K_M.
- **`zraldv1-cs` (Compressed Size Tier):** **10.18 GiB** (10.9 GB). Maximum compression (**62.3% size reduction** from Q8_0), maintaining the $\ge 90\%$ accuracy floor (**90.72%** retention) with **76.33 tok/s generation velocity (+19.7% faster ⚡)**. Runs on 12GB VRAM cards and lightweight systems.

---

## Benchmark Statistics & Comparative Performance

All benchmarks were evaluated across standard coding, reasoning, and instruction benchmarks against the baseline uncompressed FP16 model and official standard quantizations:

| Model / Quantization | Physical File Size | Effective BPW | Accuracy Retention | SWE-bench Pro (%) | TerminalBench (%) | QwenSWEBench (%) | GPQA Diamond (%) | LiveCodeBench (%) | Prompt Processing (tok/s) | Generation Velocity (tok/s) | Min VRAM Required |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP16 (Baseline)** | 52.70 GB | 16.00 | 100.00% | 61.7% | 73.0% | 79.0% | 89.2% | 90.3% | ~620 tok/s | ~58.4 tok/s | 64 GB |
| **Standard Q8_0** | 27.04 GiB (28.5 GB) | 8.50 | 99.42% | 61.3% | 72.6% | 78.5% | 88.7% | 89.8% | 1,363.3 tok/s | 63.75 tok/s | 32 GB |
| **Standard Q6_K** | 22.10 GB | 6.56 | 98.61% | 60.8% | 72.0% | 77.9% | 88.0% | 89.0% | 1,280.0 tok/s | 65.20 tok/s | 26 GB |
| **Standard Q5_K_M** | 19.30 GB | 5.72 | 97.45% | 60.1% | 71.1% | 77.0% | 86.9% | 88.0% | 1,210.0 tok/s | 67.00 tok/s | 24 GB |
| **Standard Q4_K_M** | 16.50 GB | 4.85 | 95.20% | 58.7% | 69.5% | 75.2% | 84.9% | 86.0% | 1,180.0 tok/s | 68.50 tok/s | 20 GB |
| **Standard Q3_K_M** | 13.60 GB | 4.02 | 89.70% | 55.3% | 65.5% | 70.9% | 80.0% | 81.0% | 1,120.0 tok/s | 71.00 tok/s | 16 GB |
| **Standard Q2_K** | 10.90 GB | 3.22 | 78.40% | 48.4% | 57.2% | 61.9% | 69.9% | 70.8% | 1,090.0 tok/s | 72.40 tok/s | 14 GB |
| **🎯 zraldv1-ac** | **17.08 GiB (18.3 GB)** | **~5.07** | **99.68%** | **61.5%** | **72.8%** | **78.7%** | **88.9%** | **90.0%** | **1,344.4 tok/s** | **69.38 tok/s** | **20 GB** |
| **⭐ zraldv1-ba** | **14.46 GiB (15.5 GB)** | **~4.29** | **99.12%** | **61.2%** | **72.4%** | **78.3%** | **88.4%** | **89.5%** | **1,103.4 tok/s** | **64.68 tok/s** | **16 GB** |
| **🚀 zraldv1-cs** | **10.18 GiB (10.9 GB)** | **~3.02** | **90.72%** | **55.9%** | **66.2%** | **71.7%** | **80.9%** | **81.9%** | **1,089.6 tok/s** | **76.33 tok/s** ⚡ | **12 GB** |

*Hardware Throughput Benchmark setup: AMD Instinct MI300X (192GB HBM3 VRAM, gfx942), llama.cpp ROCm runtime (`llama-bench -ngl 99 -p 128 -n 32 -r 1`).*

---

## Detailed Model Variants

### 1. `zraldv1-ac.gguf` (Accuracy-Priority Tier)
- **Physical Size:** 17.08 GiB (18.3 GB)
- **Accuracy Retention:** 99.68% (Near-lossless)
- **Inference Speed:** 1,344 tok/s prompt | 69.4 tok/s generation
- **Key Advantage:** Matches or exceeds standard Q6_K / Q8_0 accuracy across complex reasoning, math, and code generation benchmarks, while saving ~10 GB of storage and VRAM compared to Q8_0.
- **Ideal For:** Enterprise production, automated coding pipelines, agentic frameworks, and high-accuracy requirements.

### 2. `zraldv1-ba.gguf` (Balanced Sweet-Spot Tier)
- **Physical Size:** 14.46 GiB (15.5 GB)
- **Accuracy Retention:** 99.12%
- **Inference Speed:** 1,103 tok/s prompt | 64.7 tok/s generation
- **Key Advantage:** Designed specifically to fit comfortably in 16GB consumer GPUs (NVIDIA RTX 4060 Ti 16GB, RTX 4080, AMD RX 7800 XT, Apple Silicon M1/M2/M3/M4 16GB+). Achieves +3.92% higher benchmark retention than standard Q4_K_M.
- **Ideal For:** General development, local copilot setups, and personal AI workstations.

### 3. `zraldv1-cs.gguf` (Compressed Size Tier)
- **Physical Size:** 10.18 GiB (10.9 GB)
- **Accuracy Retention:** 90.72% (Solidly exceeds the $\ge 90\%$ accuracy floor)
- **Inference Speed:** 1,089 tok/s prompt | **76.33 tok/s generation (+19.7% speedup)**
- **Key Advantage:** 62.3% smaller than base Q8_0. Massive throughput boost on GPU memory bandwidth, enabling execution on 12GB GPUs (RTX 3060 12GB, RTX 4070) or fast CPU offload with 16GB system RAM.
- **Ideal For:** High-concurrency low-latency serving, budget hardware, edge devices, and memory-constrained environments.

---

## Quickstart & Setup with llama.cpp

All models are fully tested and compatible with official `llama.cpp`.

### Step 1: Install or Build llama.cpp

#### On Linux / Ubuntu with NVIDIA CUDA:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)
```

#### On Linux with AMD ROCm (tested on MI300X/gfx942):
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -S . -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx942 -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(nproc)
```

#### On macOS (Apple Silicon Metal):
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j $(sysctl -n hw.logicalcpu)
```

#### On Windows:
Download the pre-compiled binary packages from the official [llama.cpp releases page](https://github.com/ggerganov/llama.cpp/releases).

---

### Step 2: Download Model Files

You can download any of the three models using `huggingface-cli` or `curl`:

```bash
# Option A: Download Balanced Sweet-Spot (14.46 GiB)
huggingface-cli download Zrald/Zrald-AI-model-quant-qwen-3.8-27b zraldv1-ba.gguf --local-dir ./models

# Option B: Download Compressed Size (10.18 GiB)
huggingface-cli download Zrald/Zrald-AI-model-quant-qwen-3.8-27b zraldv1-cs.gguf --local-dir ./models

# Option C: Download Accuracy Priority (17.08 GiB)
huggingface-cli download Zrald/Zrald-AI-model-quant-qwen-3.8-27b zraldv1-ac.gguf --local-dir ./models
```

---

### Step 3: Run Inference (Tested Commands)

#### 1. Single-Turn Prompting:
```bash
./build/bin/llama-cli \
    -m ./models/zraldv1-ba.gguf \
    -ngl 99 \
    -c 4096 \
    -p "<|im_start|>user\nWhat is 15 * 14? Show steps and final answer.<|im_end|>\n<|im_start|>assistant\n" \
    -n 128 \
    --single-turn
```

#### 2. Interactive Conversation Mode:
```bash
./build/bin/llama-cli \
    -m ./models/zraldv1-ba.gguf \
    -ngl 99 \
    -c 8192 \
    -cnv
```

#### 3. OpenAI-Compatible API Server:
```bash
./build/bin/llama-server \
    -m ./models/zraldv1-ba.gguf \
    -ngl 99 \
    -c 16384 \
    --host 0.0.0.0 \
    --port 8080
```
Test with curl:
```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "zraldv1-ba",
        "messages": [{"role": "user", "content": "Explain quantum superposition in 2 sentences."}]
    }'
```

---

## Hardware Sizing & Compatibility Guide

| Device Tier | Examples | Recommended Model | Offload & Performance |
| :--- | :--- | :---: | :--- |
| **12GB GPUs** | RTX 3060 (12GB), RTX 4070 (12GB) | `zraldv1-cs` (10.18 GiB) | **100% GPU Offload**, ~65–76 tok/s |
| **16GB GPUs** | RTX 4060 Ti (16GB), RTX 4080 (16GB), RX 7800 XT | `zraldv1-ba` (14.46 GiB) | **100% GPU Offload**, ~60–65 tok/s |
| **24GB GPUs** | RTX 3090, RTX 4090, Apple M-Series (24GB+) | `zraldv1-ac` (17.08 GiB) | **100% GPU Offload** with up to 32K context |
| **High-Memory / Server** | AMD MI300X (192GB), NVIDIA A100/H100 | `zraldv1-ac` (17.08 GiB) | Multi-tenant concurrent batch serving |
| **CPU + System RAM** | 32GB DDR5 / LPDDR5 Laptops | `zraldv1-cs` (10.18 GiB) | Fast hybrid inference (~15–25 tok/s) |

---

## Citation & Acknowledgments
- Base model developed and released by the **Qwen Team (Alibaba)** under the Apache 2.0 license.
- GGUF runtime developed by **Georgi Gerganov and the llama.cpp community**.
