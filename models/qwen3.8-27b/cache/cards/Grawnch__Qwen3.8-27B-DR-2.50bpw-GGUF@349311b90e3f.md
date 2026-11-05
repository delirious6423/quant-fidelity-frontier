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
| **Effective Bitrate** | `2.50 bpw` |
| **File Size** | `9.68 GiB` (`9.67 GiB` in VRAM / `10.39 GB` decimal) |
| **Context Length** | `262,144` |

## Evaluation Results

### Reasoning Benchmark
| Benchmark | Score |
| :--- | :--- |
| **AIME 2025** | **93.3%** (`28 / 30`) |

### Perplexity, KL Divergence & Token Agreement
| Evaluation Suite | Perplexity (PPL) *(vs. `BF16`)* | NLL *(vs. `BF16`)* | KL Divergence (KLD) | Top-1 Agreement | Top-5 Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WikiText-2 (`2048` ctx)** | **7.6316** *(`6.5210`)* | `2.0323` *(`1.8750`)* | `0.1980` | `81.89%` | `97.96%` |
| **WikiText-2 Full Test Split (`145 × 2048`, `296,815` tokens)** | **8.3754** *(`7.2740`)* | `2.1253` *(`1.9843`)* | — | — | — |
| **WikiText-2 (`1024` ctx)** | **8.6111** *(`7.0795`)* | `2.1530` *(`1.9572`)* | `0.2163` | `80.54%` | `97.37%` |
| **Extended Reasoning / Math / Code (`Ext`)** | **17.4249** *(`14.4714`)* | `2.8579` *(`2.6722`)* | `0.3005` | `77.04%` | `96.74%` |
| **Structured Generation (`SG`)** | **2.9671** *(`2.5256`)* | `1.0876` *(`0.9265`)* | `0.2203` | `82.90%` | `98.05%` |
