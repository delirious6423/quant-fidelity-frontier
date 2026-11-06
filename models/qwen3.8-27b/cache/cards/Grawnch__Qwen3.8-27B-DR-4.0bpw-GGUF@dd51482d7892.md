---
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - quantized
  - llama.cpp
---

# Qwen3.8-27B-DR-4.0bpw-v3-GGUF

Stock GGUF 4-bit mixed-precision quantization (`v3`) of `Qwen/Qwen3.8-27B` (`17.38 GB`, `5.00 bpw` effective including `Q6_K` embeddings, LM head, and `F32` norms/recurrent gates). Runs natively on standard mainline `llama.cpp` across CUDA, Metal, and Vulkan with zero custom kernels or lookup tables.

## Model Summary

| Property | Value |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen3.8-27B` |
| **Format** | Stock `GGUF` (`Q4_K` / `Q5_K` / `Q6_K` / `F32`) |
| **File Name** | `Qwen3.8-27B-DR-4.0bpw-v3.gguf` |
| **File Size** | `17.38 GB` (`16.18 GiB` / `17,377,598,592` bytes) |
| **Effective Bitrate** | `5.0013 bpw` (`(17,377,598,592 * 8) / 27,796,869,120`) |
| **SHA256** | `41fce0cc7e07fa27d6b7c9b6a5d15197d610193afd1d3b2bd45e263b2787a662` |
| **Context Length** | `262,144` |

## Perplexity, Logit KL Divergence & Token Agreement vs. Uncompressed `BF16`

| Evaluation Suite | Tokens | `BF16` PPL | `DR-4.0bpw-v3` PPL | `ΔPPL` vs. `BF16` | `BF16` NLL | `DR-4.0bpw-v3` NLL | Mean KLD | Top-1 Agreement | Top-5 Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`WT2_2048`** (`16 x 2048`) | `32,752` | `6.7357` | **`6.7149`** | **`-0.0208`** | `1.907415` | **`1.904325`** | **`0.022374`** | **`95.60%`** | **`99.88%`** |
| **`WT2_Full`** (`145 x 2048`) | `296,815` | `6.8970` | **`6.8925`** | **`-0.0045`** | `1.931083` | **`1.930430`** | — | — | — |
| **`WT2_1024`** (`4 x 1024`) | `4,092` | `7.3284` | **`7.2989`** | **`-0.0295`** | `1.991762` | **`1.987720`** | **`0.021618`** | **`95.75%`** | **`99.95%`** |
| **`Ext`** (Reasoning / Code) | `342` | `15.0114` | **`15.1147`** | **`+0.1033`** | `2.708812` | **`2.715666`** | **`0.011953`** | **`95.03%`** | **`100.00%`** |
| **`SG`** (`<think>` Traces) | `2,040` | `5.5672` | **`5.5497`** | **`-0.0175`** | `1.716892` | **`1.713740`** | **`0.004062`** | **`97.84%`** | **`100.00%`** |

## Downstream Reasoning Benchmark (`AIME 2025`)

| Benchmark | Problems | Metric | Score |
| :--- | :---: | :---: | :---: |
| **`AIME 2025`** (AIME I & II, `Pass@1`) | `30` | Exact Match | **`28 / 30 (93.33%)`** |
