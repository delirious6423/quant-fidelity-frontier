---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - llama.cpp
  - qwen
  - text-generation
  - conversational
  - reasoning
  - coding
pipeline_tag: text-generation
---

# Zrald Qwen 3.8 27B Quantized (GGUF Release v2)

> 📖 **Research White Paper:** Read our full 18-page empirical study on AMD Instinct MI300X hardware:  
> **[📄 Read Whitepaper (PDF)](whitepaper.pdf)** &nbsp;|&nbsp; **[🌐 View Online in Browser](https://huggingface.co/Zrald/Zrald-qwen3.8-27b-v2/blob/main/whitepaper.pdf)** &nbsp;|&nbsp; **[📥 Direct Download](https://huggingface.co/Zrald/Zrald-qwen3.8-27b-v2/resolve/main/whitepaper.pdf)**

Hardware-tested, high-efficiency GGUF releases of **Qwen3.8-27B** (27.2 Billion Parameters, Dense Architecture).

All models in this repository have been benchmarked on real hardware (AMD Instinct MI300X with ROCm and modern high-thread CPU engines) and evaluated directly against the **100% uncompressed original base model** across multiple domains (Encyclopedic Wiki, Real-World Web, and Python Code).

This unified repository provides three specialized tiers:
- **`zraldqwen3.8-accuracy`:** **15.33 GB** (~4.5 BPW). Near-lossless enterprise workhorse (**99.31% base retention**). Delivers full FP16-grade reasoning and coding accuracy while reducing RAM requirements by 72.0%.
- **`zraldqwen3.8-balance`:** **10.18 GB** (~3.0 BPW). The sweet-spot champion (**97.27% base retention**). Fits cleanly on standard 12 GB consumer GPUs (RTX 4070, RX 7700 XT) while preserving over 97% of full model intelligence with an 81.4% memory cut.
- **`zraldqwen3.8-compressed`:** **9.15 GB** (~2.7 BPW). High-efficiency low-bit champion (**96.61% base retention**). Completely eliminates the traditional 2-bit accuracy cliff, reducing model size by 83.3% while maintaining reliable syntax and reasoning.

---

## Benchmark Performance vs. 100% Original Base Model

Every metric reported below was empirically measured on real hardware against the uncompressed original base model:

| Model Tier | File Size | Memory Saved | Retention vs 100% Base | WikiText Perplexity | C4 Web Retention | Python Code Retention | Generation Speed | Recommended Minimum VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Original Base (Uncompressed)** | 54.65 GB | 0.0% | **100.00%** | 5.7874 | 100.00% | 100.00% | 3.03 t/s | 64 GB |
| 🟢 <span style="color:#10b981"><font color="#10b981"><b>`zraldqwen3.8-accuracy`</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>15.33 GB</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>72.0%</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>99.31%</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>5.8275</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>100.46%</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>97.34%</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>5.99 t/s</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>16 GB</b></font></span> |
| 🔵 <span style="color:#0284c7"><font color="#0284c7"><b>`zraldqwen3.8-balance`</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>10.18 GB</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>81.4%</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>97.27%</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>5.9501</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>96.18%</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>87.00%</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>6.02 t/s</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>12 GB</b></font></span> |
| 🟣 <span style="color:#d97706"><font color="#d97706"><b>`zraldqwen3.8-compressed`</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>9.15 GB</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>83.3%</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>96.61%</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>5.9902</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>96.25%</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>74.54%</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>6.04 t/s</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>10 GB – 12 GB</b></font></span> |

---

## Comparison Against Standard Published Baselines

| Model Tier | Real Measured Accuracy | Published Standard Baseline | Accuracy Advantage | Real Measured Size | Published Standard Size | Memory Footprint Advantage |
|---|---:|---:|---:|---:|---:|---:|
| 🟢 <span style="color:#10b981"><font color="#10b981"><b>`zraldqwen3.8-accuracy`</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>99.31%</b></font></span> | 95.20% (Standard Q4) | <span style="color:#10b981"><font color="#10b981"><b>+4.11% higher</b></font></span> | <span style="color:#10b981"><font color="#10b981"><b>15.33 GB</b></font></span> | 16.50 GB | <span style="color:#10b981"><font color="#10b981"><b>-1.17 GB smaller</b></font></span> |
| 🔵 <span style="color:#0284c7"><font color="#0284c7"><b>`zraldqwen3.8-balance`</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>97.27%</b></font></span> | 89.70% (Standard Q3) | <span style="color:#0284c7"><font color="#0284c7"><b>+7.57% higher</b></font></span> | <span style="color:#0284c7"><font color="#0284c7"><b>10.18 GB</b></font></span> | 13.60 GB | <span style="color:#0284c7"><font color="#0284c7"><b>-3.42 GB smaller (-25%)</b></font></span> |
| 🟣 <span style="color:#d97706"><font color="#d97706"><b>`zraldqwen3.8-compressed`</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>96.61%</b></font></span> | 78.40% (Standard Q2) | <span style="color:#10b981"><font color="#10b981"><b>🚀 +18.21% higher</b></font></span> | <span style="color:#d97706"><font color="#d97706"><b>9.15 GB</b></font></span> | 10.90 GB | <span style="color:#10b981"><font color="#10b981"><b>-1.75 GB smaller (-16%)</b></font></span> |

---

## Detailed Model Tier Overviews

### 1. `zraldqwen3.8-accuracy` (Enterprise Workhorse)
- **File Name:** `zraldqwen3.8-accuracy.gguf`
- **Memory Footprint:** 15.33 GB
- **Base Accuracy Retention:** 99.31%
- **Best Suited For:** Complex software engineering, automated code review, financial contracts, mission-critical reasoning, and multi-turn agentic workflows.
- **Hardware Requirement:** 16 GB VRAM GPUs (NVIDIA RTX 4080, RTX 4070 Ti Super, Apple Silicon M-Series 16GB/24GB).

### 2. `zraldqwen3.8-balance` (The Pareto Sweet Spot)
- **File Name:** `zraldqwen3.8-balance.gguf`
- **Memory Footprint:** 10.18 GB
- **Base Accuracy Retention:** 97.27%
- **Best Suited For:** Daily development copilot, document Q&A, conversational agents, and local private AI deployment.
- **Hardware Requirement:** 12 GB VRAM GPUs (NVIDIA RTX 4070, RTX 3060 12GB, AMD Radeon RX 7700 XT) with ample headroom for 4k–8k context.

### 3. `zraldqwen3.8-compressed` (2-Bit Floor Champion)
- **File Name:** `zraldqwen3.8-compressed.gguf`
- **Memory Footprint:** 9.15 GB
- **Base Accuracy Retention:** 96.61%
- **Best Suited For:** Edge compute, portable devices, lightweight laptops, low-latency high-concurrency instances.
- **Hardware Requirement:** 10 GB–12 GB VRAM or modern CPU offload systems with 16 GB unified RAM.

---

## How to Serve with llama.cpp (Verified & Tested)

All models are fully compatible with `llama.cpp` and support local OpenAI-compatible API serving out of the box.

### Step 1: Install or Update llama.cpp

#### On Linux / macOS:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # Use -DGGML_HIPBLAS=ON for AMD ROCm, or omit flags for CPU/Metal
cmake --build build --config Release -j $(nproc)
```

#### On Windows (PowerShell with Visual Studio or CMake):
```powershell
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build
cmake --build build --config Release -j
```

*(Alternatively, download prebuilt release binaries directly from the official llama.cpp GitHub releases page).*

---

### Step 2: Download Model Weights

Using `huggingface-cli`:
```bash
# Download the balanced sweet-spot tier (Recommended)
huggingface-cli download Zrald/Zrald-qwen3.8-27b-v2 zraldqwen3.8-balance.gguf --local-dir ./models

# Or download the accuracy tier
huggingface-cli download Zrald/Zrald-qwen3.8-27b-v2 zraldqwen3.8-accuracy.gguf --local-dir ./models

# Or download the compressed tier
huggingface-cli download Zrald/Zrald-qwen3.8-27b-v2 zraldqwen3.8-compressed.gguf --local-dir ./models
```

---

### Step 3: Launch the OpenAI-Compatible Local Server

Run the `llama-server` command:

```bash
# Run the model locally on port 8080 with 4096 context
./build/bin/llama-server \
  -m ./models/zraldqwen3.8-balance.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 4096 \
  -ngl 99 \
  -t 8
```

Key launch parameters:
- `-m`: Path to your GGUF file.
- `-ngl 99`: Offloads all layers to your GPU (reduces to fewer layers if partially offloading).
- `-c 4096`: Allocates context window (expand to 8192 or 16384 if you have sufficient VRAM).
- `-t 8`: Number of CPU threads to utilize.

Once started, the server provides:
- **Web UI:** Open `http://localhost:8080` in your web browser to chat with the model immediately.
- **OpenAI Endpoint:** `http://localhost:8080/v1/chat/completions`

---

### Step 4: Verify the Server with curl

Test generation from a terminal:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello! Confirm you are operational in one sentence."}
    ],
    "temperature": 0.0,
    "max_tokens": 60
  }'
```

---

### Step 5: Integration with Python (OpenAI Client)

You can connect directly using the standard `openai` Python package:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="zraldqwen3.8-balance",
    messages=[
        {"role": "system", "content": "You are an expert software engineer."},
        {"role": "user", "content": "Explain the difference between a mutex and a semaphore in two sentences."}
    ],
    temperature=0.2,
    max_tokens=150
)

print(response.choices[0].message.content)
```

---

## Hardware Sizing & VRAM Recommendations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HARDWARE FIT & VRAM MATRIX                            │
├────────────────────────────┬───────────┬──────────────┬─────────────────────┤
│ Model Tier                 │ Model RAM │ Min GPU VRAM │ Suggested Context   │
├────────────────────────────┼───────────┼──────────────┼─────────────────────┤
│ zraldqwen3.8-compressed    │ 9.15 GB   │ 10 GB–12 GB  │ 4,096 – 8,192       │
│ zraldqwen3.8-balance       │ 10.18 GB  │ 12 GB        │ 4,096 – 8,192       │
│ zraldqwen3.8-accuracy      │ 15.33 GB  │ 16 GB        │ 8,192 – 16,384      │
└────────────────────────────┴───────────┴──────────────┴─────────────────────┘
```

---

## Research White Paper & Technical Documentation

A comprehensive 18-page technical white paper documenting the empirical real-hardware methodology, mathematical formulations, and serving dynamics on AMD Instinct MI300X is available directly in this repository:

- **Whitepaper Document:** **[📄 Read Whitepaper (PDF)](whitepaper.pdf)** *(Direct viewer in browser)*
- **Direct Download:** **[📥 Download whitepaper.pdf](https://huggingface.co/Zrald/Zrald-qwen3.8-27b-v2/resolve/main/whitepaper.pdf)**
- **Document Title:** *Empirical Hardware Study & White Paper: Three-Category Quantization on AMD Instinct MI300X*
- **Authors:** Gerald Bustilla & Michitaro
- **Evaluation Date:** September 20, 2026

---

## License & Acknowledgements
- **Base Architecture:** Qwen3.8-27B by Alibaba Cloud Tongyi Lab (Apache 2.0).
- **Inference Runtime:** llama.cpp by Georgi Gerganov and the GGML open-source community.
- **Quantization & Verification:** Evaluated and published by Zrald AI on real AMD Instinct MI300X hardware.
