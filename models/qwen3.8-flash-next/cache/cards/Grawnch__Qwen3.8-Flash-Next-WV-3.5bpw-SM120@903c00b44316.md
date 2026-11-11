---
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- safetensors
- gguf
- qwen3.8
- moe
- quantized
- blackwell
- sm120
license: apache-2.0
---

# Qwen3.8-Flash-Next-WV-3.5bpw-SM120

Pre-packed native **Blackwell (`sm_120`) INT8 Tensor Core (`m16n8k32`) + Wave-LUT (`SWZ3_G32` / `SWZ4_G32`) + Rank-8 FP16 SVD** checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid layers: `36` Gated DeltaNet `GDN` + `12` Qwen Sparse Attention `QSA`, `512` routed + `1` shared expert, `4`-stream mHC Hyper-Connections, and Layer-1 `28.80 GB` Zero-Copy N-Gram PLE).

Derived losslessly from [`Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw) (`Mode 0: WV-Exact`, `3.85 bpw` effective GPU footprint) with `vq4_codebooks.pt` included for optional `2.50 bpw` 4D Wave-Lattice VQ (`45.25 GB` VRAM) transcoding.

---

## Validated Single-GPU Throughput (`NVIDIA RTX PRO 6000 Blackwell`, `1.79 TB/s` GDDR7)

| Engine / Runtime | Hardware & Memory BW | Allocated VRAM | Speculation Mode | Sustained Decode (`tok/s`) | Speedup vs `llama.cpp` |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **`llama.cpp` (Baseline)** | RTX PRO 6000 (`1.79 TB/s` GDDR7) | `~56 GB` | None (`M=1`) | `~95 tok/s` | `1.00x` |
| **`SGLang` Official FP8** | NVIDIA GB300 (`8.00 TB/s` HBM3e) | `~180 GB` | MTP / Eagle | `344 – 384 tok/s` | `3.62x – 4.04x` |
| **`WV-SM120` (`Mode 0: WV-Exact`)** | RTX PRO 6000 (`1.79 TB/s` GDDR7) | **`66.70 GB`** (+ `0 GB` Host PLE) | **Standalone `M=1` Greedy** | **`136.4 – 143.0 tok/s`** | **`1.44x – 1.51x`** |
| **`WV-SM120` (`Mode 0: WV-Exact`)** | RTX PRO 6000 (`1.79 TB/s` GDDR7) | **`66.70 GB`** (+ `0 GB` Host PLE) | **Standalone `N-Gram` (`0 MB`)** | **`166.3 – 184.2 tok/s`** | **`1.75x – 1.94x`** |
| **`WV-SM120` (`Mode 0: WV-Exact`)** | RTX PRO 6000 (`1.79 TB/s` GDDR7) | **`66.70 GB`** (+ `1.21 GB` `DFlash2`) | **`DFlash2` (`b=8` / `Tree-16`)** | **`412.0 – 432.1 tok/s`** *(Math/CUDA/JSON)*<br>`272.1 – 336.3 tok/s` *(C++20/Py)* | **`4.34x – 4.55x`** |
| **`WV-SM120` (`Mode 1: VQ4-Turbo`)** | RTX PRO 6000 (`1.79 TB/s` GDDR7) | **`45.25 GB`** (+ `1.21 GB` `DFlash2`) | **`DFlash2` (`b=8` / `Tree-16`)** | **`400.5 – 564.8 tok/s`** *(Math/CUDA/JSON)*<br>`305.8 – 324.5 tok/s` *(C++20/Py)* | **`4.22x – 5.95x`** |

- **Long-Horizon `2K`-Context Stability (`Mode 0: WV-Exact`):** Sustains **`413.6 tok/s` ($\tau = 8.05$ accepted tokens/round)** across a continuous `1,700`-token AIME Diophantine proof and **`405.5 tok/s` ($\tau = 7.87$)** through full `<|im_end|>` completion on vectorized `float4` CUDA kernel refactoring.
- **Raw 48-Layer `180B` `CUDAGraph` Step Latency (`v12b`):**
  - `Mode 0 (3.85 bpw)`: `M=1`: **`6.380 ms`** | `M=2`: `7.736 ms` | `M=4`: `9.061 ms` | `M=8`: **`11.181 ms`** | `M=12`: `12.953 ms` | `M=16`: **`14.211 ms`**
  - `Mode 1 (2.50 bpw)`: `M=1`: **`6.224 ms`** | `M=2`: `7.394 ms` | `M=4`: `8.399 ms` | `M=8`: **`10.040 ms`** | `M=12`: `11.582 ms` | `M=16`: **`12.718 ms`**

---

## Held-Out Logit & Perplexity Evaluation (`248,320` Vocab)

| Evaluation Suite | `BF16` PPL | `WV-3.5bpw` PPL (`ΔPPL`) | Mean `KLD` | Same `Top-1` | Final `CosSim` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`Self-Gen Math/Code` (`<think>`)** | `5.434` | **`5.474` (`+0.040`)** | **`0.09757`** | **`88.28%`** | **`0.956945`** |
| **`External Held-Out` (`2048`)** | `11.443` | **`10.212` (`-1.232`)** | **`0.18083`** | **`84.12%`** | **`0.948237`** |
| **`WikiText-2` (`4x2048`)** | `2.148` | **`2.399` (`+0.251`)** | **`0.31614`** | **`85.50%`** | **`0.893036`** |

---

## Repository Contents

| File(s) | Size | Placement | Description |
| :--- | :---: | :---: | :--- |
| `globals.safetensors` | `~1.3 GB` | GPU VRAM | Token embeddings, final mHC norms/projections, `SWZ4_G32` LM head, and Layer-1 PLE query/key/conv weights |
| `layers_00_05.safetensors` .. `layers_42_47.safetensors` | `8 x ~7.71 GB` (`61.68 GB`) | GPU VRAM | Pre-swizzled `sm_120` `SWZ3_G32` / `SWZ4_G32` / `Q8-MMA` + Rank-8 FP16 SVD tensors for all `48` layers |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00002-of-00002.gguf` | `28.80 GB` | Host Pinned RAM (`0 GB` VRAM) | Layer-1 N-Gram Per-Layer Embedding table (`cudaHostRegisterMapped` zero-copy PCIe gather) |
| `sm120_manifest.json` | `< 10 KB` | Host | Layer topology, GDN/QSA routing flags, DFlash tap indices, and Rank-8 SVD expert IDs |
| `vq4_codebooks.pt` | `10 KB` | GPU Shared Mem | Calibrated `256 x 4` 4D Wave-Lattice VQ codebooks (`gu` & `dn`) for `Mode 1 (2.50 bpw, 45.25 GB)` |
| `bench_mode0_v12b_results.json`, `bench_mode1_vq4_results.json` | `40 KB` | Reference | Full 6-task synchronized hardware benchmark logs & generated outputs |
