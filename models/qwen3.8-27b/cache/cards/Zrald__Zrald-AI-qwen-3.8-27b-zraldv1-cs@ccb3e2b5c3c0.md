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

# Zrald-AI Qwen 3.8 27B (zraldv1-cs - Compressed Size Tier)

Official high-efficiency GGUF release of **Qwen3.8-27B** optimized for **Compressed Size Tier**.

This repository contains **`zraldv1-cs.gguf`** (10.18 GiB / 10.93 GB), physically benchmarked on AMD Instinct MI300X hardware.

For cross-comparison tables against standard Q8_0, Q6_K, Q5_K, Q4_K, and Q2_K models, visit the master repository:
👉 [Zrald/Zrald-AI-model-quant-qwen-3.8-27b](https://huggingface.co/Zrald/Zrald-AI-model-quant-qwen-3.8-27b)

---

## Key Performance Indicators

- **Physical File Size:** 10.18 GiB (10.93 GB)
- **Accuracy Retention:** **90.72%**
- **Prompt Processing Throughput:** **1,089.6 tok/s**
- **Token Generation Velocity:** **76.33 (⚡ +19.7% speedup) tok/s**
- **Recommended Minimum VRAM:** **12 GB**
- **Profile:** Ultra-compact tier with 62.3% size reduction from Q8_0 base. Solidly exceeds the >=90% accuracy floor at 76.3 tok/s.

---

## Quickstart with llama.cpp

### 1. Download Model File
```bash
huggingface-cli download Zrald/Zrald-AI-qwen-3.8-27b-zraldv1-cs zraldv1-cs.gguf --local-dir ./models
```

### 2. Run Single Prompt
```bash
./build/bin/llama-cli \
    -m ./models/zraldv1-cs.gguf \
    -ngl 99 \
    -c 4096 \
    -p "<|im_start|>user\nSolve 15 * 14 step-by-step.<|im_end|>\n<|im_start|>assistant\n" \
    -n 128 \
    --single-turn
```

### 3. Run Interactive Chat
```bash
./build/bin/llama-cli \
    -m ./models/zraldv1-cs.gguf \
    -ngl 99 \
    -c 8192 \
    -cnv
```

### 4. Launch OpenAI-Compatible API Server
```bash
./build/bin/llama-server \
    -m ./models/zraldv1-cs.gguf \
    -ngl 99 \
    -c 16384 \
    --host 0.0.0.0 \
    --port 8080
```

---

## Citation & Acknowledgments
- Base model by the **Qwen Team (Alibaba)** under Apache 2.0.
- Runtime by **Georgi Gerganov and the llama.cpp community**.
