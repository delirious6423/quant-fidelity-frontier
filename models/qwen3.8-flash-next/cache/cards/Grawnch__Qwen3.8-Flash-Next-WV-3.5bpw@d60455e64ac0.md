---
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
- gguf
- qwen3.8
- moe
- quantized
license: apache-2.0
---

# Qwen3.8-Flash-Next-WV-3.5bpw

Non-linear **`WV` (`3.5 bpw`)** 2-shard quantization of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` Total Parameters / `48` Hybrid GDN + QSA Layers / `512` Routed + `1` Shared Expert / `4`-Stream mHC Hyper-Connections / Layer-1 N-Gram PLE).

## Model Family

| Repository | Format & Target Runtime | GPU VRAM | Host RAM (PLE) | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw)** | 2-Shard `WV` Non-Linear `GGUF` (`3.5 bpw`) | `55.12 GB` | `28.80 GB` (`mmap`) | Master lossless `WV` container (`ΔPPL +0.040` on `<think>`, `0.09757` KLD) |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120)** | Pre-swizzled `sm_120` `safetensors` + Host PLE | `66.70 GB` (`Mode 0`)<br>`45.25 GB` (`Mode 1`) | `28.80 GB` (`0 GB` VRAM) | **Up to `2,500.27 tok/s`** on 1× RTX PRO 6000; `< 6s` load; `80K` sparse `QSA` indexer |
| **[`Qwen3.8-Flash-Next-3.5bpw-GGUF`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF)** | 2-Shard Standard `llama.cpp` `GGUF` (`3.5 bpw`) | `55.05 GB` | `28.80 GB` (`mmap`) | 100% upstream `ggml` (`Q3_K`/`Q4_0`/`Q2_0`/`Q6_K`) drop-in for `llama.cpp` |

---

## Shards & Memory Footprint

| File | Size | Placement | Contents |
| :--- | :---: | :---: | :--- |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00001-of-00002.gguf` | `55.12 GB` (`51.33 GiB`) | GPU VRAM | Token embeddings, all `48` transformer / GDN / MoE layers (`1,223` tensors), & output head |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00002-of-00002.gguf` | `28.80 GB` (`26.82 GiB`) | Host RAM (`mmap`) | Layer-1 N-Gram Per-Layer Embedding table (`Q4_0`, `1` tensor) |
| `benchmarks/qfn_wave_v4_z3_eval_summary.json` | `120.5 KB` | Reference | Full 48-layer teacher-vs-quantized KLD/PPL/cosine evaluation logs |
| `benchmarks/aime25_temp10_64k_results.json` | `11.4 KB` | Reference | Per-problem `64K` AIME 2025 logs (`23/30 = 76.67%`, AIME II `14/15 = 93.33%`) |

---

## Held-Out 48-Layer Logit & Perplexity Evaluation (`248,320` Vocab)

| Evaluation Suite | `BF16` PPL | `WV-3.5bpw` PPL (`ΔPPL`) | Mean `KLD` | Same `Top-1` | Final `CosSim` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Self-Gen Math/Code` (`<think>`)** | `5.434` | **`5.474` (`+0.040`)** | **`0.09757`** | **`88.28%`** | **`0.956945`** |
| **`External Held-Out` (`2048`)** | `11.443` | **`10.212` (`-1.232`)** | **`0.18083`** | **`84.12%`** | **`0.948237`** |
| **`WikiText-2` (`4×2048`)** | `2.148` | **`2.399` (`+0.251`)** | **`0.31614`** | **`85.50%`** | **`0.893036`** |
| **`WikiText-2` (`4×1024`)** | `2.301` | **`2.586` (`+0.285`)** | **`0.33541`** | **`84.63%`** | **`0.891150`** |

---

## Full `64K`-Context AIME 2025 Benchmark

| Decoding Configuration | Max Tokens / Problem | AIME 2025 I (`15`) | AIME 2025 II (`15`) | Overall (`30`) |
| :--- | :---: | :---: | :---: | :---: |
| **Sampling (`temp=1.0, top_p=0.95, top_k=20`)** | `65,536` (`64K`) | `9 / 15` (`60.00%`) | **`14 / 15` (`93.33%`)** | **`23 / 30` (`76.67%`)** |
| **Deterministic Greedy (`temp=0.0`)** | `36,864` (`36K`) | `10 / 15` (`66.67%`) | `12 / 15` (`80.00%`) | **`22 / 30` (`73.33%`)** |

> **Runtime Note:** Routed expert tensors (`ffn_*_exps.weight`) use the lossless `WV` non-linear codebook + dynamic expert-tier + Rank-8 FP16 SVD container format and require a `WV`-compatible decoder (such as **Strata** or the native Blackwell runtime paired with [`Qwen3.8-Flash-Next-WV-3.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120)). For stock upstream `llama.cpp`, use [`Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF).
