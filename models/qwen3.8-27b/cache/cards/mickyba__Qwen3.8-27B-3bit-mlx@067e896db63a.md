---
language:
- en
- zh
license: apache-2.0
tags:
- mlx
- 3bit
- vision-language
- qwen
- quantization
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
---

# Qwen 3.8 27B — 3-Bit MLX Model

This repository contains a **3-bit quantized MLX version** of the `Qwen/Qwen3.8-27B` multimodal (vision-language) model, optimized for Apple Silicon via [MLX](https://github.com/ml-explore/mlx) and [`mlx-vlm`](https://github.com/ml-explore/mlx-examples/tree/main/mlx_vlm).

---

## 📊 Overview & Model Summary

| Attribute | Details |
| :--- | :--- |
| **Base Model** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Architecture** | Vision-Language Dense Model (27 Billion Parameters) |
| **Quantization** | 3-bit uniform quantization (`--q-bits 3`) |
| **Format** | MLX Native (`.safetensors`) |
| **Estimated VRAM** | **~11–12 GB Unified Memory** (vs. ~54 GB in FP16 / ~15 GB in 4-bit) |
| **Primary Target** | Apple Silicon Macs (M1/M2/M3/M4 with 16GB+ Unified Memory) |

---

## 💡 Why 3-Bit?

Uniform 2-bit quantization often degrades weights to the point of producing corrupt output ("garbage/word salad"). 

3-bit quantization provides an optimal middle ground for memory-constrained systems:
- **Significantly higher coherence & reasoning capabilities** compared to 2-bit.
- **Lower memory footprint (~11–12 GB VRAM)** compared to standard 4-bit (~15 GB VRAM), fitting comfortably on 16GB RAM Macs with system overhead to spare.

---

## 🚀 Quickstart & Usage Guide

### 1. Prerequisites & Installation

Ensure you have Python 3.10+ and the required MLX multimodal packages installed:

```bash
pip install -U mlx-vlm mlx
```

### 2. Run Generation via CLI

To test the model with a text prompt directly from your terminal:

```bash
python3 -m mlx_vlm.generate \
  --model <YOUR_HF_USERNAME>/Qwen3.8-27B-3bit-mlx \
  --prompt "Hello! Write a clean Python script to reverse a string."
```

For multimodal/vision inputs:

```bash
python3 -m mlx_vlm.generate \
  --model <YOUR_HF_USERNAME>/Qwen3.8-27B-3bit-mlx \
  --image https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/ai2d-demo.jpg \
  --prompt "Describe what you see in this image."
```

---

## 💻 Integration with VS Code & Local Clients

Serve this model locally using an **OpenAI-compatible HTTP endpoint** to connect with extensions like **Continue.dev**, **Roo Code**, or **Cline**.

### Step 1: Start the Server

```bash
python3 -m mlx_vlm.server \
  --model <YOUR_HF_USERNAME>/Qwen3.8-27B-3bit-mlx \
  --port 8000
```

### Step 2: Configure Continue.dev (`config.json`)

Add the following provider entry to your `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Qwen 3.8 27B (3-bit MLX)",
      "provider": "openai",
      "model": "default",
      "apiBase": "http://127.0.0.1:8000/v1",
      "apiKey": "sk-local"
    }
  ]
}
```

---

## ⚡ Precision & Performance Comparison

| Precision / Quant | Disk Size | Min VRAM | Coherence / Output Quality |
| :--- | :--- | :--- | :--- |
| **FP16 (Base)** | ~54 GB | ~60 GB | Baseline (100%) |
| **4-Bit MLX** | ~15 GB | ~16 GB | High (~95% quality retention) |
| **3-Bit MLX (This Repo)** | **~11 GB** | **~12 GB** | **Moderate-High (Usable logic, fits 16GB Macs)** |
| **2-Bit MLX** | ~7.5 GB | ~8 GB | Low (Prone to output artifacts / garbage) |

---

## 🛠️ Reproduction / How This Model Was Created

Converted using `mlx_vlm.convert` on Apple Silicon:

```bash
# Convert base model to 3-bit MLX format
python3 -m mlx_vlm.convert \
  --hf-path Qwen/Qwen3.8-27B \
  -q \
  --q-bits 3 \
  --mlx-path ~/models/Qwen3.8-27B-3bit-mlx

# Push to Hugging Face Hub
hf repo create Qwen3.8-27B-3bit-mlx --type model
hf upload <YOUR_HF_USERNAME>/Qwen3.8-27B-3bit-mlx ~/models/Qwen3.8-27B-3bit-mlx .
```

---

## 📜 License & Acknowledgments

- **Base Model License:** [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) (inherited from Qwen family)
- **Framework:** Built with Apple's [MLX](https://github.com/ml-explore/mlx) framework.

