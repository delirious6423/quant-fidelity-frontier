---
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - quantized
---

# Qwen3.8-27B-DR-2.50bpw-GGUF

Quantized GGUF weights for `Qwen/Qwen3.8-27B`.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen3.8-27B` |
| **Format** | `GGUF` (`Q2_K` / `Q3_K` / `Q4_K`) |
| **File Size** | `14.38 GB` |
| **Context Length** | `32,768` |

## Evaluation Results

### Reasoning Benchmark

| Benchmark | Pass@1 | Pass@2 |
| :--- | :--- | :--- |
| **AIME 2025** | **83.3%** (`25 / 30`) | **93.3%** (`28 / 30`) |

> **Evaluation & Recommended Decoding Settings (`Qwen3.8-27B` Reasoning Mode:xhigh):**
> * **Sampler Parameters:** `temperature = 0.6`, `top_p = 0.95`, `top_k = 20`, `min_p = 0.0`, `presence_penalty = 0.6`
> * **Chat Template:** Native Qwen3.8 thinking template (`enable_thinking = True`, default `xhigh` reasoning effort)
> * **Context / Token Budget:** `42,000` thinking tokens (`+512` post-`</think>` answer summary tokens; `45K+` total context window recommended)

### Perplexity, KL Divergence & Token Agreement
| Evaluation Suite | Perplexity (PPL) *(vs. `BF16`)* | NLL *(vs. `BF16`)* | KL Divergence (KLD) | Top-1 Agreement | Top-5 Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WikiText-2 (`2048` ctx)** | **7.5964** *(`6.5210`)* | `2.0277` *(`1.8750`)* | `0.1942` | `82.04%` | `98.00%` |
| **WikiText-2 Full Test Split (`145 × 2048`, `296,815` tokens)** | **8.3371** *(`7.2740`)* | `2.1207` *(`1.9843`)* | — | — | — |
| **WikiText-2 (`1024` ctx)** | **8.5715** *(`7.0795`)* | `2.1484` *(`1.9572`)* | `0.2125` | `80.69%` | `97.41%` |
| **Extended Reasoning / Math / Code (`Ext`)** | **17.3456** *(`14.4714`)* | `2.8533` *(`2.6722`)* | `0.2967` | `77.19%` | `96.78%` |
| **Structured Generation (`SG`)** | **2.9534** *(`2.5256`)* | `1.0830` *(`0.9265`)* | `0.2165` | `83.04%` | `98.09%` |
