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
| Benchmark | Score |
| :--- | :--- |
| **AIME 2025** | **93.3%** (`28 / 30`) |

### Perplexity, KL Divergence & Token Agreement
| Evaluation Suite | Perplexity (PPL) *(vs. `BF16`)* | NLL *(vs. `BF16`)* | KL Divergence (KLD) | Top-1 Agreement | Top-5 Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **WikiText-2 (`2048` ctx)** | **7.5964** *(`6.5210`)* | `2.0277` *(`1.8750`)* | `0.1942` | `82.04%` | `98.00%` |
| **WikiText-2 Full Test Split (`145 × 2048`, `296,815` tokens)** | **8.3371** *(`7.2740`)* | `2.1207` *(`1.9843`)* | — | — | — |
| **WikiText-2 (`1024` ctx)** | **8.5715** *(`7.0795`)* | `2.1484` *(`1.9572`)* | `0.2125` | `80.69%` | `97.41%` |
| **Extended Reasoning / Math / Code (`Ext`)** | **17.3456** *(`14.4714`)* | `2.8533` *(`2.6722`)* | `0.2967` | `77.19%` | `96.78%` |
| **Structured Generation (`SG`)** | **2.9534** *(`2.5256`)* | `1.0830` *(`0.9265`)* | `0.2165` | `83.04%` | `98.09%` |
