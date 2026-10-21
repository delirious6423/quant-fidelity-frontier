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

# Zrald-AI Qwen 3.8 27B (zraldv1-ac - Accuracy Priority Tier)

Official high-efficiency GGUF release of **Qwen3.8-27B** optimized for **Accuracy Priority Tier**.

This repository contains **`zraldv1-ac.gguf`** (17.08 GiB / 18.34 GB), physically benchmarked on AMD Instinct MI300X hardware.

For cross-comparison tables against standard Q8_0, Q6_K, Q5_K, Q4_K, and Q2_K models, visit the master repository:
👉 [Zrald/Zrald-AI-model-quant-qwen-3.8-27b](https://huggingface.co/Zrald/Zrald-AI-model-quant-qwen-3.8-27b)

---

## Key Performance Indicators

- **Physical File Size:** 17.08 GiB (18.34 GB)
- **Accuracy Retention:** **99.68%**
- **Prompt Processing Throughput:** **1,344.4 tok/s**
- **Token Generation Velocity:** **69.4 tok/s**
- **Recommended Minimum VRAM:** **20 GB**
- **Profile:** Highest precision retention (99.68%). Matches or beats Q6_K and Q8_0 in benchmark accuracy while saving ~10 GB VRAM.

---

## Quickstart with llama.cpp

### 1. Download Model File
```bash
huggingface-cli download Zrald/Zrald-AI-qwen-3.8-27b-zraldv1-ac zraldv1-ac.gguf --local-dir ./models
```

### 2. Run Single Prompt
```bash
./build/bin/llama-cli \
    -m ./models/zraldv1-ac.gguf \
    -ngl 99 \
    -c 4096 \
    -p "<|im_start|>user\nSolve 15 * 14 step-by-step.<|im_end|>\n<|im_start|>assistant\n" \
    -n 128 \
    --single-turn
```

### 3. Run Interactive Chat
```bash
./build/bin/llama-cli \
    -m ./models/zraldv1-ac.gguf \
    -ngl 99 \
    -c 8192 \
    -cnv
```

### 4. Launch OpenAI-Compatible API Server
```bash
./build/bin/llama-server \
    -m ./models/zraldv1-ac.gguf \
    -ngl 99 \
    -c 16384 \
    --host 0.0.0.0 \
    --port 8080
```

---

## Citation & Acknowledgments
- Base model by the **Qwen Team (Alibaba)** under Apache 2.0.
- Runtime by **Georgi Gerganov and the llama.cpp community**.
