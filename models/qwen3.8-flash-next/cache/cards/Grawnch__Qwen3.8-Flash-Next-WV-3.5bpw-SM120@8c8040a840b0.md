---
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
- safetensors
- gguf
- qwen3.8
- moe
- quantized
- blackwell
- sm120
- speculative-decoding
license: apache-2.0
---

> [!WARNING]
> **Superseded:** A newer checkpoint is recommended: [Grawnch/Qwen3.8-Flash-Next-WV-2.96bpw-SM120](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-2.96bpw-SM120) — WV-v6 2.96 bpw SM120 (self-contained with unified v22.5.0 wheel & DFlash2).
> Kept here for benchmark reproducibility and historical reference.

# Qwen3.8-Flash-Next-WV-3.5bpw-SM120 (`v19`)

Pre-packed native **Blackwell (`sm_120`) `3.5 bpw`** checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid layers: `36` Gated DeltaNet `GDN` + `12` Qwen Sparse Attention `QSA`, `512` routed + `1` shared expert, `4`-stream mHC Hyper-Connections, and Layer-1 `28.80 GB` Zero-Copy N-Gram PLE).

Achieves `0.001533` Math/Code KLD vs master [`Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw) (`99.023%` Top-1 logit agreement, `0.999465` 48-layer hidden cosine similarity) and **`30 / 30 (100.00%)` Pass@1 back-to-back (`60 / 60` total)** on **`MathArena/aime_2025`** at **`128K` (`131,072`) tokens** (`B = 30`, `temp = 1.0, top_p = 0.95, top_k = 20`) — including **`30 / 30` with `presence_penalty = 0.0`**.

For the compact `2.7 bpw` checkpoint (`51.17 GiB` static VRAM, `1.76M` KV token capacity), see [`Grawnch/Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120).

## Model Family

| Repository | Format & Target Runtime | Static GPU VRAM | Host RAM (PLE) | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120)** | `v19` `Mode 0` `sm_120` `safetensors` (`3.5 bpw`) | `67.49 GiB` | `28.80 GB` (`0 GB` VRAM) | **`30 / 30` (`100.00%`)** `128K` AIME 2025 (`pp = 0.0`); `0.00153` KLD vs `WV`; `1,281.3 tok/s` (`B=30`) |
| **[`Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120)** | `v19` `Mode 2` `sm_120` `safetensors` (`2.7 bpw`) | `51.17 GiB` | `28.80 GB` (`0 GB` VRAM) | `1.76M` KV tok; `1,258.9 tok/s` (`B=30`) |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw)** | 2-Shard `WV` `GGUF` (`3.5 bpw`) | `55.12 GB` | `28.80 GB` (`mmap`) | Master `WV` container (`ΔPPL +0.040` on `<think>`, `0.09757` KLD vs `BF16`) |
| **[`Qwen3.8-Flash-Next-3.5bpw-GGUF`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF)** | 2-Shard Standard `llama.cpp` `GGUF` (`3.5 bpw`) | `55.05 GB` | `28.80 GB` (`mmap`) | 100% upstream `ggml` (`Q3_K`/`Q4_0`/`Q2_0`/`Q6_K`) drop-in for `llama.cpp` |

---

## `v19` Verification: `MathArena/aime_2025` (`128K` = `131,072` Tokens, `B = 30`)

Evaluated on **1× NVIDIA RTX PRO 6000 Blackwell** (`101.97 GB` GDDR7, `1.79 TB/s`) with official RL sampling (`temp = 1.0, top_p = 0.95, top_k = 20, min_p = 0.0`, `seed = 42`, `1,096,320` paged KV tokens):

| Evaluation Pass | Decoding Policy | Overall Pass@1 | AIME 2025 I | AIME 2025 II | Truncated / Loops | Avg Tokens (`Max`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`v19` Pass A (Strict Vanilla)** | **`presence_penalty = 0.0`, `mask_eos_in_think = False`** | **`30 / 30` (`100.00%`)** | **`15 / 15` (`100%`)** | **`15 / 15` (`100%`)** | **`0` / `0`** | `25,241.3` (`89,807`) |
| **`v19` Pass B (EOS-Guard + PP 0.2)** | `presence_penalty = 0.2`, `mask_eos_in_think = True` | **`30 / 30` (`100.00%`)** | **`15 / 15` (`100%`)** | **`15 / 15` (`100%`)** | **`0` / `0`** | `25,849.6` (`124,807`) |
| *(Prev `v18` `Mode 0`, `64K`)* | *Strict Vanilla (`pp = 0.0`)* | *`22 / 30` (`73.33%`)* | *`11 / 15`* | *`11 / 15`* | *`8` / `4`* | *`29,638.9` (`65,536`)* |

---

## `v19` Parity vs Master `WV-3.5bpw` & Synchronized Throughput

| Metric / Workload | `v19 Mode 0` (`3.5 bpw`) | `v19 Mode 2` (`2.7 bpw`) | Notes |
| :--- | :---: | :---: | :--- |
| **Math/Code On-Policy Mean KLD vs `WV-3.5bpw`** | **`0.001533`** (`29.3×` lower vs `v18`) | **`0.039365`** *(beats `v18 Mode 0`!)* | `99.023%` Top-1 / `100.0%` Top-5 (`Mode 0`) |
| **48-Layer Hidden Cosine Similarity** | **`0.999465`** (`min 0.998659`) | **`0.994985`** | All 48 layers |
| **Static GPU VRAM (`b_max = 32`) / Auto KV Pool** | **`67.49 GiB`** / `1,072,640` KV tok | **`51.17 GiB`** / `1,763,968` KV tok | `92.03 GiB` total pool (`28.80 GB` PLE in host RAM) |
| **Pure `M = 1` Batched Greedy (`B = 1 / 16 / 30`)** | `103.1 / 857.5 / 1,281.3 tok/s` | `102.6 / 849.0 / 1,258.9 tok/s` | `0 MB` drafter |
| **Pure `M = 1` Batched Sampled (`B = 1 / 16 / 30`)** | `103.1 / 671.4 / 970.4 tok/s` | `102.5 / 673.2 / 982.5 tok/s` | `temp = 1.0, top_k = 20, top_p = 0.95` |
| **Single-Stream `B = 1` Speculative Decode** | `204.19 tok/s` (`τ = 5.14`) | **`262.17 tok/s`** (`τ = 6.35`) | Exact speculative verification |

---

## Repository Contents

| File / Path | Size | Placement | Description |
| :--- | :---: | :---: | :--- |
| `globals.safetensors` | `1.99 GB` | GPU VRAM | Token embeddings, final norm/projections, `output.weight` (`248,320 × 2560`), & PLE projections |
| `layers_00_05.safetensors` .. `layers_42_47.safetensors` | `8 × 7.99 GB` (`63.92 GB`) | GPU VRAM | `48` quantized `Mode 0` hybrid MoE layers (`6` layers per shard) |
| `qsa_indexer.safetensors` | `39.34 MB` | GPU VRAM | `12` QSA layers' indexer weights for `128K`-context Top-2048 sparse attention |
| `dflash2_swz4_calibrated.safetensors` | `241.8 MB` | Optional GPU | Calibrated `DFlash2` speculative drafter weights |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00002-of-00002.gguf` | `28.80 GB` | Host Pinned RAM (`0 GB` VRAM) | Layer-1 N-Gram Per-Layer Embedding table |
| `sm120_manifest.json` | `9.2 KB` | Host | `SM120` shard & runtime manifest (`mode = 0`) |
| `v16_submodular_tier_maps_h96.pt` & `vq4_codebooks.pt` | `23.5 KB` | GPU | Runtime lookup tables |
| `config.json`, `generation_config.json`, `tokenizer*` | `12.8 MB` | Host | Hugging Face config & `248,320`-vocab tokenizer metadata |
| `benchmarks/*.json` | `~350 KB` | Reference | `v19` Gate 1 KLD parity, Gate 2 throughput, & Gate 3 `128K` `30/30` AIME 2025 logs |

### `v19.1` Runtime Benchmark Update (`2026-11-13`)
- **Gate 1 (`benchmarks/qfn_v19_1_gate1_kld_parity.json`):** `gate1_passed = true` — `100.0%` exact Top-20 index match (`0.0` max logit diff at `pp=0.0` & `pp=0.2`), `0.016581` `1x8192` sparse-QSA handover KLD (`100.0%` Top-5, `0.989442` hidden cosine on `Mode 0`; `0.30722` on `Mode 2`).
- **Gate 2 (`benchmarks/qfn_v19_1_gate2_throughput.json`, 1x RTX PRO 6000 Blackwell):**
  - `Mode 0 (3.5 bpw)`: `108.08 tok/s` (`+4.85%`) `B=1` greedy, `108.54 tok/s` (`+5.33%`) `B=1` sampled, `1,306.78 tok/s` `B=30` greedy, `982.41 tok/s` `B=30` sampled, **`213.38 tok/s` (`+4.50%`) `B=1` speculative**.
  - `Mode 2 (2.7 bpw)`: `107.25 tok/s` (`+4.58%`) `B=1` greedy, `107.74 tok/s` (`+5.11%`) `B=1` sampled, **`1,312.10 tok/s` (`+4.23%`) `B=30` greedy**, `985.62 tok/s` `B=30` sampled, **`274.39 tok/s` (`+4.66%`) `B=1` speculative**.
