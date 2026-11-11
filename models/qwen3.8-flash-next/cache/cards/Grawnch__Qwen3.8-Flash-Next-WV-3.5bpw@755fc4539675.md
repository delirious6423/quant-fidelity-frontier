---
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- qwen3.8
- moe
- quantized
license: apache-2.0
---

# Qwen3.8-Flash-Next-WV-3.5bpw

Non-linear **WV (`3.5 bpw`)** 2-shard quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` Total Parameters / `48` Layers / `512` Routed + `1` Shared Expert / Layer-1 N-Gram PLE).

## Shards & Memory Footprint

| File | Size | Placement | Contents |
| :--- | :---: | :---: | :--- |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00001-of-00002.gguf` | `55.12 GB` (`51.33 GiB`) | GPU VRAM | Token embeddings, all `48` transformer / GDN / MoE layers (`1,223` tensors), & output head |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00002-of-00002.gguf` | `28.80 GB` (`26.82 GiB`) | Host RAM (`mmap`) | Layer-1 N-Gram Per-Layer Embedding table (`Q4_0`, `1` tensor) |

## Held-Out Logit & Perplexity Evaluation (`248,320` Vocab)

| Evaluation Suite | `BF16` PPL | `WV-3.5bpw` PPL (`ΔPPL`) | Mean `KLD` | Same `Top-1` | Final `CosSim` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Self-Gen Math/Code` (`<think>`)** | `5.434` | **`5.474` (`+0.040`)** | **`0.09757`** | **`88.28%`** | **`0.956945`** |
| **`External Held-Out` (`2048`)** | `11.443` | **`10.212` (`-1.232`)** | **`0.18083`** | **`84.12%`** | **`0.948237`** |
| **`WikiText-2` (`4x2048`)** | `2.148` | **`2.399` (`+0.251`)** | **`0.31614`** | **`85.50%`** | **`0.893036`** |
| **`WikiText-2` (`4x1024`)** | `2.301` | **`2.586` (`+0.285`)** | **`0.33541`** | **`84.63%`** | **`0.891150`** |

> **Runtime Note:** Routed expert tensors (`ffn_*_exps.weight`) use the lossless `WV` non-linear codebook + dynamic expert-tier + low-rank container format and require a `WV`-compatible decoder (such as **Strata**).
