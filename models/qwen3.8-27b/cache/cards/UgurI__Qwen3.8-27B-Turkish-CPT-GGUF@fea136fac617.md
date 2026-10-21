---
language:
- tr
- en
license: cc-by-nc-sa-4.0
base_model: Qwen/Qwen3.8-27B
tags:
- turkish
- continued-pretraining
- gguf
- q4_k_m
- qwen3.8
- ollama
pipeline_tag: text-generation
---

# Qwen3.8-27B-Turkish-CPT-GGUF (Q4_K_M)

GGUF quantizations for **Qwen3.8-27B-Turkish-CPT**, adapted for high-quality Turkish comprehension, reasoning, and instruction following.

## Model Description

- **Base Model:** `Qwen/Qwen3.8-27B` (Text-only causal LM backbone)
- **Quantization:** `Q4_K_M` (15.41 GB, 4.92 BPW)
- **Context Length:** Up to 262,144 tokens
- **Compatibility:** Ollama, LM Studio, llama.cpp, Jan, OpenWebUI

## Available Quantizations

| File | Quant Method | File Size | Description |
|---|:---:|:---:|---|
| `Qwen3.8-27B-Turkish-CPT-Q4_K_M.gguf` | `Q4_K_M` | **15.41 GB** | Recommended balance of speed, memory (~18 GB VRAM/RAM), and Turkish accuracy |

## Benchmark & Exam Performance

| Metric | Base Model (Qwen3.8-27B) | Turkish-CPT-GGUF | Improvement |
|---|---:|---:|---:|
| **Turkish Validation Perplexity** | 7.5838 | **6.9360** | **-8.55%** |
| **Turkish Test Perplexity** | 8.0534 | **7.3303** | **-8.98%** |
| **TurkishMMLU (90-Q Stratified)** | 60.00% | **75.56%** | **+15.56%** |
| **Belebele Reading Comprehension (TR)** | 83.33% | **90.00%** | **+6.67%** |
| **Belebele Reading Comprehension (EN)** | 96.67% | **96.67%** | **100% Retained** |

## Quick Start with Ollama

1. Download the `Qwen3.8-27B-Turkish-CPT-Q4_K_M.gguf` and `Modelfile` files.
2. In your terminal, run:
   ```bash
   ollama create qwen3.8-turkish -f Modelfile
   ollama run qwen3.8-turkish
   ```

## Quick Start with LM Studio / llama.cpp

1. Place `Qwen3.8-27B-Turkish-CPT-Q4_K_M.gguf` into your LM Studio models directory.
2. Or run directly with llama-cli:
   ```bash
   llama-cli -m Qwen3.8-27B-Turkish-CPT-Q4_K_M.gguf -p "Türkiye'de yenilenebilir enerji kaynaklarının önemi nedir?" -n 256
   ```

## Citation & Attribution

- **Training Corpus:** VNGRS-AI (`vngrs-ai/vngrs-web-corpus`)
- **Base Model:** Qwen Team (`Qwen/Qwen3.8-27B`)
- **Author:** Ugur Inanc
