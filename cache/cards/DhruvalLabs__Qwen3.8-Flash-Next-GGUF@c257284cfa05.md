---
license: other
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - qwen3.8
  - mixture-of-experts
language:
  - en
---

<div align="center">

# Qwen3.8-Flash-Next — GGUF Quantizations

[![Original Model](https://img.shields.io/badge/Original-Qwen_Qwen3.8--Flash--Next-blue)](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
[![quant-kit](https://img.shields.io/badge/Made_with-quant--kit-green)](https://github.com/DhruvalPtl/quant-kit)

**Quantized GGUF versions of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**

Works with **[llama.cpp](https://github.com/ggerganov/llama.cpp)** · **[Ollama](https://ollama.ai)** · **[LM Studio](https://lmstudio.ai)** 

</div>

---

## ⚡ The Model Architecture

**Qwen3.8-Flash-Next** is an incredibly powerful 125B parameter Mixture-of-Experts (MoE) model featuring a massive 51B N-gram embedding table. Despite its massive total size, its sparse architecture only activates **~6B parameters per token**, making it blazingly fast during inference!

*This GGUF was quantized to `Q4_K_M` to preserve maximum intelligence while shrinking the massive 360GB original size down to a highly efficient 111GB footprint.*

---

## 📦 Available Files (Sharded)

Because this model is so large, the `Q4_K_M` GGUF has been **split into 3 shards** to bypass the 50GB file size limit. 

| Filename | Size | RAM Required | Quant | Quality |
|---|---|---|---|---|
| `Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf` | 45.0 GB | - | `Q4_K_M` | ⭐⭐⭐⭐ |
| `Qwen3.8-Flash-Next-Q4_K_M-00002-of-00003.gguf` | 45.0 GB | - | `Q4_K_M` | ⭐⭐⭐⭐ |
| `Qwen3.8-Flash-Next-Q4_K_M-00003-of-00003.gguf` | 21.0 GB | - | `Q4_K_M` | ⭐⭐⭐⭐ |

> ⚠️ **IMPORTANT:** You must download all 3 files into the same folder! When loading the model in `llama.cpp` or `Ollama`, simply point the program to the first file (`...00001-of-00003.gguf`) and it will automatically detect and stitch together the rest of the shards during inference. 

*(**System RAM Warning**: You will need at least 128GB of Unified Memory (Mac) or System RAM/VRAM to run this model without severe swapping).*

---

## 🚀 How to Use

### llama.cpp CLI
Make sure all 3 shards are in the same directory.
```bash
./llama-cli \
  -m Qwen3.8-Flash-Next-Q4_K_M-00001-of-00003.gguf \
  -p "Explain the benefits of sparse Mixture-of-Experts architectures." \
  --conversation \
  -n 1024
