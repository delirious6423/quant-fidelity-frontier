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

# Qwen3.8-Flash-Next-WV-3.5bpw-SM120

Pre-packed native **Blackwell (`sm_120`) INT8 Tensor Core (`m16n8k32`) + Wave-LUT (`SWZ3_G32` / `SWZ4_G32`) + Rank-8 FP16 SVD + `80K` Sparse `QSA` Indexer** checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid layers: `36` Gated DeltaNet `GDN` + `12` Qwen Sparse Attention `QSA`, `512` routed + `1` shared expert, `4`-stream mHC Hyper-Connections, and Layer-1 `28.80 GB` Zero-Copy N-Gram PLE).

Derived losslessly from [`Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw) (`Mode 0: WV-Exact`, `3.85 bpw` effective GPU footprint) with `vq4_codebooks.pt` included for optional `2.50 bpw` 4D Wave-Lattice VQ (`45.25 GB` VRAM) transcoding.

## Model Family

| Repository | Format & Target Runtime | GPU VRAM | Host RAM (PLE) | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120)** | Pre-swizzled `sm_120` `safetensors` + Host PLE | `66.70 GB` (`Mode 0`)<br>`45.25 GB` (`Mode 1`) | `28.80 GB` (`0 GB` VRAM) | **Up to `2,500.27 tok/s`** on 1× RTX PRO 6000; `< 6s` load; `80K` sparse `QSA` indexer |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw)** | 2-Shard `WV` Non-Linear `GGUF` (`3.5 bpw`) | `55.12 GB` | `28.80 GB` (`mmap`) | Master lossless `WV` container (`ΔPPL +0.040` on `<think>`, `0.09757` KLD) |
| **[`Qwen3.8-Flash-Next-3.5bpw-GGUF`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF)** | 2-Shard Standard `llama.cpp` `GGUF` (`3.5 bpw`) | `55.05 GB` | `28.80 GB` (`mmap`) | 100% upstream `ggml` (`Q3_K`/`Q4_0`/`Q2_0`/`Q6_K`) drop-in for `llama.cpp` |

---

## Single-GPU Throughput (`1× NVIDIA RTX PRO 6000 Blackwell`, `101.97 GB` GDDR7, `1.79 TB/s`)

All `v15.1` batched figures report `100%` verified target-model pre-EOS tokens (`non_eos_tok_per_s == raw_hw_tok_per_s`) alongside strict per-slot budget-capped effective throughput:

| Engine / Workload | Hardware & BW | Batch (`B`) | Speculation / Horizon | Verified Throughput | Strict Capped (`tok/s`) | vs. `llama.cpp` |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: |
| **`llama.cpp` (`3.5bpw GGUF`)** | RTX PRO 6000 (`1.79 TB/s`) | `1` | None (`M=1`) | `~95 tok/s` | `~95 tok/s` | `1.00×` |
| **`SGLang` Official `FP8`** | NVIDIA GB300 (`8.00 TB/s` HBM3e) | `1` | MTP / Eagle | `344 – 384 tok/s` | `344 – 384 tok/s` | `3.62× – 4.04×` |
| **`WV-SM120` (`Mode 0`, Standalone)** | RTX PRO 6000 (`1.79 TB/s`) | `1` | `M=1` / `N-Gram` (`0 MB`) | `143.0 / 184.2 tok/s` | `143.0 / 184.2 tok/s` | `1.51× / 1.94×` |
| **`WV-SM120` (`Mode 0: WV-Exact`)** | RTX PRO 6000 (`1.79 TB/s`) | `1` | `DFlash2` (`k=8..12` / `Tree-16`) | **`403.7 – 432.1 tok/s`** | **`403.7 – 432.1 tok/s`** | **`4.25× – 4.55×`** |
| **`WV-SM120` (`Mode 1: VQ4-Turbo`)** | RTX PRO 6000 (`1.79 TB/s`) | `1` | `DFlash2` (`45.25 GB` VRAM) | **`400.5 – 564.8 tok/s`** | **`400.5 – 564.8 tok/s`** | **`4.22× – 5.95×`** |
| **`WV-SM120` (`v15.1` Pure `M=1`)** | RTX PRO 6000 (`1.79 TB/s`) | `32` | None (`27.84 ms/step`) | **`1,149.43 tok/s`** | **`1,149.43 tok/s`** | **`12.10×`** |
| **`WV-SM120` (`v15.1` `cuda_sm120_code`)** | RTX PRO 6000 (`1.79 TB/s`) | `32` | `DFlash2` (`k=6`, `M=192`, `τ=3.40`) | **`1,656.23 tok/s`** | `1,380.40 tok/s` | **`17.43×`** |
| **`WV-SM120` (`v15.1` `aime2025_math`)** | RTX PRO 6000 (`1.79 TB/s`) | `32` | `DFlash2` (`k=6`, `M=192`, `τ=4.58`) | **`2,180.41 tok/s`** | `1,729.25 tok/s` (`B=28`) | **`22.95×`** |
| **`WV-SM120` (`v15.1` `structured_json`)** | RTX PRO 6000 (`1.79 TB/s`) | `32` | `DFlash2` (`k=8`, `M=256`, `τ=5.75`) | **`2,500.27 tok/s`** | `2,229.31 tok/s` (`k=6`) | **`26.32×`** |

---

## Full `64K`-Context AIME 2025 Benchmark (`Mode 0: WV-Exact`)

Evaluated end-to-end across all `30` official **AIME 2025 (I & II)** problems concurrently (`B = 30`) on a single RTX PRO 6000 using the `80K`-token paged `QSA` KV pool and hardware Top-2048 block-sparse `QSA` indexer (`qsa_indexer.safetensors`):

| Decoding Configuration | Max Tokens / Problem | AIME 2025 I (`15`) | AIME 2025 II (`15`) | Overall (`30`) | Total Tokens & Wall Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sampling (`temp=1.0, top_p=0.95, top_k=20`)** | `65,536` (`64K`) | `9 / 15` (`60.00%`) | **`14 / 15` (`93.33%`)** | **`23 / 30` (`76.67%`)** | `465,134` tok in `9.55 min` (`997.4 tok/s` peak) |
| **Deterministic Greedy (`temp=0.0`)** | `36,864` (`36K`) | `10 / 15` (`66.67%`) | `12 / 15` (`80.00%`) | **`22 / 30` (`73.33%`)** | `472,390` tok in `10.17 min` (`1,106.0 tok/s` peak) |

---

## Held-Out 48-Layer Logit & Perplexity Evaluation (`248,320` Vocab)

| Evaluation Suite | `BF16` PPL | `WV-3.5bpw` PPL (`ΔPPL`) | Mean `KLD` | Same `Top-1` | Final `CosSim` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Self-Gen Math/Code` (`<think>`)** | `5.434` | **`5.474` (`+0.040`)** | **`0.09757`** | **`88.28%`** | **`0.956945`** |
| **`External Held-Out` (`2048`)** | `11.443` | **`10.212` (`-1.232`)** | **`0.18083`** | **`84.12%`** | **`0.948237`** |
| **`WikiText-2` (`4×2048`)** | `2.148` | **`2.399` (`+0.251`)** | **`0.31614`** | **`85.50%`** | **`0.893036`** |

---

## Repository Contents

| File / Path | Size | Placement | Description |
| :--- | :---: | :---: | :--- |
| `globals.safetensors` | `1.66 GB` | GPU VRAM | Token embeddings, final mHC norms/projections, `SWZ4_G32` LM head, and Layer-1 PLE query/key/conv weights |
| `layers_00_05.safetensors` .. `layers_42_47.safetensors` | `8 × 7.71 GB` (`61.69 GB`) | GPU VRAM | Pre-swizzled `sm_120` `SWZ3_G32` / `SWZ4_G32` / `Q8-MMA` + Rank-8 FP16 SVD tensors for all `48` layers |
| `qsa_indexer.safetensors` | `39.34 MB` | GPU VRAM | `12` QSA layers' `idx_qk` (`BF16 [640, 2560]`) + `idx_qn` / `idx_kn` (`FP32 [128]`) for `80K`-context Top-2048 sparse attention |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00002-of-00002.gguf` | `28.80 GB` | Host Pinned RAM (`0 GB` VRAM) | Layer-1 N-Gram Per-Layer Embedding table (`cudaHostRegisterMapped` zero-copy PCIe gather) |
| `sm120_manifest.json` | `7.7 KB` | Host | Layer topology, GDN/QSA routing flags, DFlash2 tap indices, and Rank-8 SVD expert IDs |
| `vq4_codebooks.pt` | `10.1 KB` | GPU Shared Mem | Calibrated `256 × 4` 4D Wave-Lattice VQ codebooks (`gu` & `dn`) for `Mode 1 (2.50 bpw, 45.25 GB)` |
| `config.json`, `generation_config.json`, `tokenizer*` | `12.8 MB` | Host | Hugging Face config & `248,320`-vocab tokenizer metadata |
| `benchmarks/qfn_v15_1_batched_spec_results.json` | `49.2 KB` | Reference | Full `102`-row `v15.1` `B=1..32` batched speculative + `M=1` benchmark logs (`2,500.27 tok/s` peak) |
| `benchmarks/aime25_temp10_64k_results.json` | `11.4 KB` | Reference | Per-problem `64K` AIME 2025 logs at `temp=1.0` (`23/30 = 76.67%`, AIME II `14/15 = 93.33%`) |
| `benchmarks/aime25_greedy_36k_results.json` | `11.4 KB` | Reference | Per-problem `36K` deterministic greedy AIME 2025 logs (`22/30 = 73.33%`) |
| `benchmarks/bench_mode0_v12b_single_stream.json` | `16.0 KB` | Reference | Single-stream (`B=1`) `Mode 0: WV-Exact` 6-task benchmark & generated outputs |
| `benchmarks/bench_mode1_vq4_single_stream.json` | `15.9 KB` | Reference | Single-stream (`B=1`) `Mode 1: VQ4-Turbo` (`45.25 GB` VRAM) 6-task benchmark & generated outputs |
