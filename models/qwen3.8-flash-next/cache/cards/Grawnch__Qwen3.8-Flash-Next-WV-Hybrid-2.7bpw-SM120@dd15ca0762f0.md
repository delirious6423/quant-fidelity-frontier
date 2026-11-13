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

# Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120 (`v19 Lossless-Spine`)

Pre-transcoded native **Blackwell (`sm_120`) `Mode 2` Submodular Hybrid (`2.70 bpw`, `51.17 GiB` static VRAM)** checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid layers: `36` Gated DeltaNet `GDN` + `12` Qwen Sparse Attention `QSA`, `512` routed + `1` shared expert, `4`-stream mHC Hyper-Connections, and Layer-1 `28.80 GB` Zero-Copy N-Gram PLE).

Upgraded in **`v19 ("Lossless-Spine SM120")`** with the exact **`SWZ8_G32` dense QSA/GDN spine, `SWZ8_G32` shared experts (`Q8_0`-exact), `SWZ8_G32` `248,320`-vocab LM head (preserving Step 4.3 `</think>` surgery), exact `BF16` GDN `(w_a, w_b)` recurrent gates, `g=32` `Q8_0` mHC projections, 43-expert Rank-8 FP16 SVD, and `128K` Sparse `QSA` Indexer**, combined with `H = 96` submodular Hot `SWZ3/4` Wave-LUT experts + `416` Cold `VQ4` 4D Wave-Lattice experts per layer:

- **Higher Fidelity Than Previous `v18` `3.5 bpw` at `-14.4 GiB` Lower VRAM:** Achieves **`0.039365` Math/Code KLD** (`95.90%` Top-1 logit agreement, `0.994985` 48-layer hidden-state cosine similarity) vs master [`WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw), outperforming `v18 Mode 0` (`0.044850` KLD, `65.58 GiB`) in just **`51.17 GiB`** static GPU VRAM.
- **`1.76M`-Token KV Pool on 1× RTX PRO 6000 (`101.97 GB`):** Leaves `40.86 GiB` free for `13,781` paged KV blocks (**`1,763,968` KV tokens**), hitting **`262.17 tok/s`** single-stream `B = 1` `DFlash2` (`τ = 6.35`), **`1,258.88 tok/s`** (`B = 30` greedy), and **`982.48 tok/s`** (`B = 30` sampled).

For full lossless `3.50 bpw` `Mode 0` (`30 / 30 = 100.00%` on `128K` `MathArena/aime_2025` with `presence_penalty = 0.0`, `0.001533` KLD), see [`Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120).

## Model Family

| Repository | Format & Target Runtime | Static GPU VRAM | Host RAM (PLE) | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120)** | `v19` Lossless-Spine `Mode 2` Hybrid `sm_120` `safetensors` (`2.7 bpw`) | `51.17 GiB` | `28.80 GB` (`0 GB` VRAM) | `0.03937` Math/Code KLD; `1.76M` KV tok; **`262.2 tok/s`** (`B=1` `DFlash2`), **`1,258.9 tok/s`** (`B=30`) |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120)** | `v19` Lossless-Spine `Mode 0` `sm_120` `safetensors` (`3.5 bpw`) | `67.49 GiB` | `28.80 GB` (`0 GB` VRAM) | **`30 / 30` (`100.00%`)** `128K` AIME 2025 (`pp = 0.0`); `0.00153` KLD vs `WV`; `1,281.3 tok/s` (`B=30`) |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw)** | 2-Shard `WV` Non-Linear `GGUF` (`3.5 bpw`) | `55.12 GB` | `28.80 GB` (`mmap`) | Master lossless `WV` container (`ΔPPL +0.040` on `<think>`, `0.09757` KLD vs `BF16`) |
| **[`Qwen3.8-Flash-Next-3.5bpw-GGUF`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF)** | 2-Shard Standard `llama.cpp` `GGUF` (`3.5 bpw`) | `55.05 GB` | `28.80 GB` (`mmap`) | 100% upstream `ggml` (`Q3_K`/`Q4_0`/`Q2_0`/`Q6_K`) drop-in for `llama.cpp` |

---

## `v19 Mode 2` Parity & Single-GPU Throughput (`1× NVIDIA RTX PRO 6000 Blackwell`)

| Metric / Workload | `v19 Mode 2` (`2.7 bpw` Hybrid) | `v19 Mode 0` (`3.5 bpw` Exact) | Notes |
| :--- | :---: | :---: | :--- |
| **Math/Code On-Policy Mean KLD vs `WV-3.5bpw`** | **`0.039365`** (`95.90%` Top-1) | `0.001533` (`99.02%` Top-1) | `99.85%` Top-5 logit agreement (`Mode 2`) |
| **48-Layer `TF_ROUTER` Hidden Cosine Similarity** | **`0.994985`** (`min 0.992209`) | `0.999465` (`min 0.998659`) | Identical `SWZ8_G32` spine + `BF16` GDN gates |
| **Static GPU VRAM (`b_max = 32`) / Auto KV Pool** | **`51.17 GiB`** / **`1,763,968` KV tok** | `67.49 GiB` / `1,072,640` KV tok | `-16.32 GiB` VRAM, `+64.5%` larger KV pool |
| **Pure `M = 1` Batched Greedy (`B = 1 / 16 / 30`)** | `102.6 / 849.0 / 1,258.9 tok/s` | `103.1 / 857.5 / 1,281.3 tok/s` | `burst_k = 8` `CUDAGraph`, `0.0s` transcode |
| **Pure `M = 1` Batched Sampled (`B = 1 / 16 / 30`)** | **`102.5 / 673.2 / 982.5 tok/s`** | `103.1 / 671.4 / 970.4 tok/s` | `temp = 1.0, top_k = 20, top_p = 0.95` |
| **Single-Stream `B = 1` `DFlash2` (`HybridTree-16`)** | **`262.17 tok/s`** (`τ = 6.35`) | `204.19 tok/s` (`τ = 5.14`) | `+28.4%` faster `B=1` speculative verify vs `Mode 0` |

---

## Repository Contents

| File / Path | Size | Placement | Description |
| :--- | :---: | :---: | :--- |
| `globals.safetensors` | `1.99 GB` | GPU VRAM | Token embeddings, `g=32` `Q8_0` final mHC, exact `SWZ8_G32` `output.weight` (`248,320 × 2560`), & `SWZ8_G32` PLE projections |
| `layers_00_05.safetensors` .. `layers_42_47.safetensors` | `8 × 5.76 GB` (`46.04 GB`) | GPU VRAM | Native `v19` `Mode 2` shards (`SWZ8_G32` spine/shared-expert + `BF16` GDN gates + `96` Hot `SWZ3/4` + `416` Cold `VQ4` + 43-expert Rank-8 SVD) |
| `qsa_indexer.safetensors` | `39.34 MB` | GPU VRAM | `12` QSA layers' `idx_qk` + `idx_qn` / `idx_kn` for `128K`-context Top-2048 sparse attention |
| `dflash2_swz4_calibrated.safetensors` | `241.8 MB` | Optional GPU | Calibrated `SWZ4_G32` `DFlash2` 20-matrix speculative drafter |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00002-of-00002.gguf` | `28.80 GB` | Host Pinned RAM (`0 GB` VRAM) | Layer-1 N-Gram Per-Layer Embedding table (`cudaHostRegisterMapped` zero-copy PCIe gather) |
| `sm120_manifest.json` | `9.2 KB` | Host | Native `v19_lossless_spine` `mode = 2` (`0.0s` load-time transcoding) manifest |
| `v16_submodular_tier_maps_h96.pt` & `vq4_codebooks.pt` | `23.5 KB` | GPU | Submodular `H=96` tier maps & `256 × 4` 4D Wave-Lattice VQ codebooks |
| `config.json`, `generation_config.json`, `tokenizer*` | `12.8 MB` | Host | Hugging Face config & `248,320`-vocab tokenizer metadata |
| `benchmarks/*.json` | `~305 KB` | Reference | `v19` Gate 1 KLD parity & Gate 2 throughput benchmark logs |
