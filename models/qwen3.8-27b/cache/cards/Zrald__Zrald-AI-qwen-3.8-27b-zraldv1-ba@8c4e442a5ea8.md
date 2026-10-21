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

# Zrald-AI Qwen 3.8 27B (zraldv1-ba - Balanced Sweet-Spot Tier)

Official high-efficiency GGUF release of **Qwen3.8-27B** optimized for **Balanced Sweet-Spot Tier**.

This repository contains **`zraldv1-ba.gguf`** (14.46 GiB / 15.52 GB), physically benchmarked on AMD Instinct MI300X hardware.

For cross-comparison tables against standard Q8_0, Q6_K, Q5_K, Q4_K, and Q2_K models, visit the master repository:
👉 [Zrald/Zrald-AI-model-quant-qwen-3.8-27b](https://huggingface.co/Zrald/Zrald-AI-model-quant-qwen-3.8-27b)

---

## Key Performance Indicators

- **Physical File Size:** 14.46 GiB (15.52 GB)
- **Accuracy Retention:** **99.12%**
- **Prompt Processing Throughput:** **1,103.4 tok/s**
- **Token Generation Velocity:** **64.7 tok/s**
- **Recommended Minimum VRAM:** **16 GB**
- **Profile:** Optimal balance of size and intelligence (99.12% retention). Fits 16GB consumer GPUs and beats standard Q4_K_M by +3.92% accuracy.

---

## Quickstart with llama.cpp

### 1. Download Model File
```bash
huggingface-cli download Zrald/Zrald-AI-qwen-3.8-27b-zraldv1-ba zraldv1-ba.gguf --local-dir ./models
```

### 2. Run Single Prompt
```bash
./build/bin/llama-cli \
    -m ./models/zraldv1-ba.gguf \
    -ngl 99 \
    -c 4096 \
    -p "<|im_start|>user\nSolve 15 * 14 step-by-step.<|im_end|>\n<|im_start|>assistant\n" \
    -n 128 \
    --single-turn
```

### 3. Run Interactive Chat
```bash
./build/bin/llama-cli \
    -m ./models/zraldv1-ba.gguf \
    -ngl 99 \
    -c 8192 \
    -cnv
```

### 4. Launch OpenAI-Compatible API Server
```bash
./build/bin/llama-server \
    -m ./models/zraldv1-ba.gguf \
    -ngl 99 \
    -c 16384 \
    --host 0.0.0.0 \
    --port 8080
```

---

## Citation & Acknowledgments
- Base model by the **Qwen Team (Alibaba)** under Apache 2.0.
- Runtime by **Georgi Gerganov and the llama.cpp community**.
