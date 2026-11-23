---
base_model: Qwen/Qwen3.8-Flash-Next
license: apache-2.0
library_name: transformers
quantized_by: nchapman
tags:
- gptq
- hadamard
- quip
- w2a16
- w4a16
- mixed-precision
- moe
- ple
- rco
- compressed-tensors
- vllm
---

# Qwen3.8-Flash-Next-W2.6A16-HGPTQ-RCO

Mixed 2-bit / 4-bit Hadamard + GPTQ (`W2.6A16`, group size 128, `H_128` Walsh–Hadamard rotation, symmetric mid-riser `code_offset=0.5` 2-bit grid) and 4-bit `Custom-LUT4` Layer-1 PLE quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) built with Riemannian Constrained Optimization (`HGPTQ-RCO`). It compresses the 335.3 GiB (360 GB) BF16 checkpoint down to **71.67 GiB** total (`41.51 GiB` pure text backbone; 4.68× smaller than BF16, saving 21.17 GiB vs. pure `W4A16`) and brings the 48-layer MoE trunk + Native MTP + Vision down to **44.84 GiB** so it fits on a single **48 GB GPU** (with the 26.82 GiB 4-bit PLE table `mmap`'d in system RAM) or 100% in VRAM on a single **80 GB GPU**.

- **VRAM & GPU fit:** `44.84 GiB` for the 48-layer MoE trunk + Native MTP + Vision tower (`41.51 GiB` pure text backbone; ~3.1 GiB headroom on a 48 GB GPU with Layer-1 PLE `mmap`'d in RAM) and `71.67 GiB` total including the `26.82 GiB` 4-bit PLE table (~8.3 GiB headroom all-in on an 80 GB GPU; for 96 GB+ setups, see [`Qwen3.8-Flash-Next-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ) at `92.84 GiB`).
- **Quantization recipe:** Keeps 100% of Attention/GDN (156 linears), Shared Experts (144 linears), Native MTP (1,543 linears), and Vision (110 linears) in `W4_g128` (`H_128`), `lm_head` in `W8_g64`, and Layer-1 PLE in 4-bit `Custom-LUT4` with $W_{kv}$ MMSE compensation, while LOO KL sensitivity allocates the 15 highest-sensitivity routed-expert layers (7,680 experts: layers `9, 17, 22, 23, 28..31, 34, 35, 37..39, 46, 47`) to `W4_g128` and the remaining 33 routed-expert layers (16,896 experts) to `W2_g128` (`code_offset=0.5`), polished with end-to-end FP32-master Norm/GDN/HC/PLE KL QAT.
- **Fidelity vs. BF16:** `0.1091` weighted KL (`0.1691` mean KL, `−6.60%` PPL) on `probe-v4`, `0.1961` mean KL (`0.2157` weighted KL, `−23.74%` PPL) on the `probe-guard` holdout, `95.13%` / `82.90%` isolated/E2E MTP top-1 agreement, and `96.36%` / `92.72%` isolated/E2E vision top-1 agreement (`97.04%` / `93.33%` on `vision-guard-v2`).

## Quality vs. BF16 Base Model (`Qwen/Qwen3.8-Flash-Next`)

Teacher-forced evaluation comparing this checkpoint directly against the unquantized BF16 base model and the `92.84 GiB` pure 4-bit [`Qwen3.8-Flash-Next-W4A16-HGPTQ`](https://huggingface.co/nchapman/Qwen3.8-Flash-Next-W4A16-HGPTQ) checkpoint:

| Evaluation | BF16 Base (`Qwen/Qwen3.8-Flash-Next`) | `W4A16-HGPTQ` (`4.00 bpw` experts) | **`W2.6A16-HGPTQ-RCO` (`2.625 bpw` experts)** | Notes |
|---|---:|---:|---:|---|
| **Total Safetensors Size** | `335.28 GiB` (`360,000,192,888 B`) | `92.84 GiB` (`99,690,665,368 B`) | **`71.67 GiB` (`76,950,776,792 B`)** | **`4.68×` compression** (`41.51 GiB` pure text backbone; `−263.61 GiB` vs. BF16, `−21.17 GiB` vs. W4A16) |
| **Trunk + MTP + Vision Size** (excl. `mmap` PLE) | `239.91 GiB` (`257,599,900,928 B`) | `66.02 GiB` (`70,890,439,312 B`) | **`44.84 GiB` (`48,150,550,736 B`)** | Fits on a **48 GB GPU** (`26.82 GiB` 4-bit PLE `mmap`'d in host RAM) |
| **Text Perplexity** (378 prompts, 852k tokens) ↓ | `3.6085` | `3.5683` (`−1.11%`) | **`3.3704` (`−6.60%`)** | Math (`0.0599` KL), chat (`0.0623` KL), science (`0.0870` KL), IF (`0.0987` KL) |
| **Text Divergence vs. BF16** (weighted KL / token mean KL) ↓ | `0.0000` | `0.0421` / `0.0798` | **`0.1091` / `0.1691`** | Full-vocabulary FP32 KL divergence (`−21.9%` wKL vs. V1) |
| **General & Multilingual Holdout Perplexity** (76 prompts, 23k tokens) ↓ | `24.2483` | `22.8992` (`−5.56%`) | **`18.4912` (`−23.74%`)** | Token mean KL `0.1961` (weighted KL `0.2157`, `no_robots` KL `0.1802`; `−37.7%` mean KL / `−53.3%` wKL vs. V1) |
| **Native MTP (Speculative Drafter)** (12,339 positions) | `100.0%` Top-1 (`NLL 2.6008`) | `95.41%` Top-1 (`KL 0.0184`) | **`95.13%` Top-1 (`NLL 2.6104`)** | Isolated MTP KL `0.0208` (end-to-end Top-1 `82.90%`, KL `0.3238`) |
| **Vision (VQA, Charts, Documents & OCR)** (96 images, 302 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `0.9906` cos sim · `97.35%` Top-1 | **`0.9898` cos sim · `96.36%` Top-1** | Isolated ViT KL `0.0206` (end-to-end Top-1 `92.72%`, KL `0.1364`, `−38.0%` vs. V1) |
| **Vision (UI / ScreenSpot & Math OCR)** (48 images, 135 tokens) | `1.0000` cos sim · `100.0%` Top-1 | `0.9930` cos sim · `99.26%` Top-1 | **`0.9926` cos sim · `97.04%` Top-1** | Isolated ViT KL `0.0175` (end-to-end Top-1 `93.33%`, KL `0.1095`) |

## What Is Quantized (`HGPTQ-RCO` 48 GB Trunk / 80 GB Total Allocation)

- **Strictly Standard Bit-Widths (`{2, 4, 8}` bits, `group_size=128`, `H_128`)**:
  - **Always-Active Backbone (`100%` in `W4_g128`, `300` linears)**: All `156` hybrid GatedDeltaNet + QSA-Indexed-Attention projections (`linear_attn`, `self_attn`) and all `144` shared expert projections (`mlp.shared_expert.{gate_proj,up_proj,down_proj}`) across all 48 layers stay in `W4_g128` (`head_dim=128`, MSE scale search + 2-pass H-CD).
  - **Routed MoE Experts (`2.625 bpw` average across `73,728` linears, solved via Leave-One-Out End-to-End KL Sensitivity)**:
    - **`W4_g128` (`15` layers = `7,680` experts = `23,040` linears)**: Layers `[9, 17, 22, 23, 28, 29, 30, 31, 34, 35, 37, 38, 39, 46, 47]`.
    - **`W2_g128` (`33` layers = `16,896` experts = `50,688` linears)**: Remaining 33 layers quantized with symmetric mid-riser 2-bit Hadamard Cholesky GPTQ (`code_offset=0.5`, codes `{-1.5, -0.5, +0.5, +1.5}`) + 2-pass Hessian Coordinate Descent.
  - **Untied Output Head (`W8_g64`, `1` linear)**: `lm_head` (`248320 × 2560`) quantized with unrotated 8-bit Cholesky GPTQ + MSE scale search (`group_size=64`).
  - **Layer-1 N-Gram Per-Layer Embedding (`51.2B` parameters → `26.822 GiB` across `model-ple-000..007.safetensors`)**: Kept in 4-bit `Custom-LUT4` (`group_size=32`, `90 B/row`, `45 rows/4 KiB page`) with $\operatorname{diag}(H_m)$-weighted MSE scale search + `W_kv`-HGPTQ + 2-pass H-CD + closed-form FP16 group scale refinement + population $W_{kv}$ MMSE compensation on `layers.1.ple.{key_proj,value_proj}`.
  - **Native MTP (`1,543` linears in `W4_g128`, `1.289 GiB`)**: `4` `self_attn` projections + `3` `mlp.shared_expert` projections + `1,536` routed MoE expert projections (`head_dim=128`, `group_size=128`).
  - **Vision (`110` linears in `W4_g128`, `0.220 GiB`)**: All 27 ViT blocks + 2 spatial merger projections (`visual.blocks.*.mlp.linear_fc2` uses per-channel GPTQ + `head_dim=16` Hadamard rotation `v_vision_fc2`; all other Vision linears use `group_size=128` + `head_dim=128`).
  - **End-to-End FP32-Master Norm, GDN, Hyper-Connection & PLE KL QAT**: Post-allocation teacher-forced KL + hidden-MSE QAT over all 4-stream `hyper_connection` norms and injectors, QK norms, GatedDeltaNet norms/gates, `mlp.shared_expert_gate`, and Layer-1 `ple.{norm_key,norm_query,norm_conv,conv1d}`.
- **Serving note (`vLLM`)**: Serve with `VLLM_WARP_QUANT_W2_MARLIN=1` (verified on `vLLM 0.34+` with `--max-num-seqs 256`) for full-speed CUDA-graph Marlin/Humming fused MoE kernels, or `--enforce-eager` for minimum true 2-bit VRAM (`44.84 GiB` trunk + `26.82 GiB` `mmap`'d PLE).

## Shard Layout & Memory Tiering (`71.666 GiB` total / `44.844 GiB` Trunk+MTP+Vision)

| Component | Shards | Size | Format | Recommended Placement (`48 GB GPU` vs `80 GB GPU`) |
|---|---:|---:|---|---|
| Embed + LM Head (`W8_g64`) + Mixer | `model-embed-head.safetensors` | `1.795 GiB` | BF16 + `W8_g64` | GPU VRAM resident |
| 48 Target Layers (Backbone + `15` `W4` + `33` `W2` MoE layers) | `model-layer-00..47.safetensors` | `41.510 GiB` | `W4_g128` (`15` layers + attn/shared) + `W2_g128` (`33` layers) | GPU VRAM resident on `48 GB` / `80 GB` GPUs |
| Layer-1 PLE N-Gram Table (`128` shards, `51.2B` params) | `model-ple-000..007.safetensors` | `26.822 GiB` | 4-bit `Custom-LUT4` (`g32`, `90 B/row`) | `mmap` in host RAM (`48 GB GPU`) or HBM resident (`80 GB GPU`) |
| Native MTP Layer (`512` experts + Shared MLP + QSA Attn) | `model-mtp.safetensors` | `1.289 GiB` | W4A16-HGPTQ (`g128-h128`) | GPU VRAM resident |
| Vision Tower (`27` blocks + merger) | `model-vision.safetensors` | `0.220 GiB` | W4A16-HGPTQ (`g128-h128` / `channel-h16`) | GPU VRAM resident |

## Static Expert Hot-Cache Rankings (`runtime/static_hot_cache_rankings.json`)

Both `runtime/static_hot_cache_rankings.json` and root `static_hot_cache_rankings.json` map layer indices `"0"` through `"47"` to global expert IDs (`0..511`) ordered from **most-frequently routed (hottest) to least-frequently routed**. Full per-layer activation counts, routing weight mass, and `mtp.0` rankings are provided in `runtime/expert_routing_profile.json` (`9,804,460` expert assignments per target layer):

| Layer | Precision | Active Experts | Top-5 Hottest Expert IDs | `top_32` Hit Rate | `top_64` Hit Rate | `top_88` Hit Rate | `top_128` Hit Rate | `top_256` Hit Rate |
|---|---|---:|---|---:|---:|---:|---:|---:|
| Layer `0` | `W2_g64` | `508 / 512` | `[470, 306, 280, 279, 266]` | `13.82%` | `23.88%` | `30.82%` | `41.33%` | `69.83%` |
| Layer `1` | `W2_g64` | `511 / 512` | `[130, 308, 191, 320, 58]` | `14.61%` | `25.19%` | `32.27%` | `43.07%` | `71.83%` |
| Layer `12` | `W2_g64` | `512 / 512` | `[34, 139, 63, 272, 166]` | `15.65%` | `26.82%` | `34.27%` | `45.18%` | `73.27%` |
| Layer `23` | `W2_g64` | `512 / 512` | `[499, 361, 242, 183, 84]` | `16.65%` | `27.92%` | `34.87%` | `45.02%` | `70.63%` |
| Layer `35` | `W2_g64` | `512 / 512` | `[189, 479, 18, 372, 199]` | `22.03%` | `34.88%` | `42.43%` | `53.01%` | `77.90%` |
| Layer `47` | `W4_g64` | `510 / 512` | `[242, 398, 157, 110, 127]` | `31.81%` | `46.18%` | `54.51%` | `66.00%` | `88.52%` |
| `mtp.0` | `W4_g64` | `512 / 512` | `[439, 433, 410, 257, 451]` | `26.68%` | `39.88%` | `47.57%` | `58.41%` | `81.31%` |

## Measured `warp34` `128K`-Context (`--kv-pages 8192`) Throughput & VRAM on `1×` and `2× RTX 3090 (24 GB)`

Across all 48 layers (`12` `W4_g64` layers at `1.992 MiB/expert` + `36` `W2_g64` layers at `1.055 MiB/expert`), each `+1` hot expert slot per rank costs **`61.875 MiB`** (`-35.3%` vs. `95.625 MiB/slot` on pure `W4A16`), while the `26.82 GiB` 4-bit PLE table is `mmap`'d in system RAM and 100% of the Native MTP layer's local experts (`256 / 256` per rank on `TP=2`) reside in GPU VRAM:

| Hardware (`TP`, Per-Rank `k_hot`) | VRAM Experts (`Total / 512`) | `128K` `tq4` + `MTP` VRAM / GPU (`Free`) | Prefill (`512` tok) | Decode (`no MTP`) | Decode (`--speculative mtp --mtp-k 3`) | Mean Tokens / Round (`ITL`) |
|---|---|---:|---:|---:|---:|---:|
| **`1× RTX 3090 24 GB`** (`TP=1`, `k_hot=96/512`) | `96 / 512` (`18.75%`) | `19,489 MiB` (`4,638 MiB` free, `20` GDN slots) | `278.8 tok/s` | **`26.10 tok/s`** | `24.04 tok/s` | `3.94` (`38.31 ms` std) |
| **`2× RTX 3090 24 GB`** (`TP=2`, `k_hot=128/256`) | `256 / 512` (`50.00%`) | `18,112 MiB` (`6,000 MiB` free, `18` GDN slots) | `403.5 tok/s` | `50.32 tok/s` | **`73.23 tok/s`** | `3.94` (`13.65 ms`) |
| **`2× RTX 3090 24 GB`** (`TP=2`, default `k_hot=160/256` LoRA+Desktop) | `320 / 512` (`62.50%`) | `21,256 MiB` (`2,856 MiB` free, `18` GDN slots) | `502.3 tok/s` | `59.69 tok/s` | **`94.97 tok/s`** | `3.94` (`10.53 ms`) |
| **`2× RTX 3090 24 GB`** (`TP=2`, `k_hot=192/256` `75%` VRAM) | `384 / 512` (`75.00%`) | `23,728 MiB` (`384 MiB` free, `18` GDN slots) | `707.6 tok/s` | `71.37 tok/s` | **`124.17 tok/s`** | `3.94` (`8.05 ms`) |
| **`2× RTX 3090 24 GB`** (`TP=2`, `k_hot=208/256` `81.25%` VRAM) | `416 / 512` (`81.25%`) | `23,728 MiB` (`384 MiB` free, `18` GDN slots) | **`787.5 tok/s`** | **`74.02 tok/s`** | **`131.55 tok/s`** | `3.94` (**`7.60 ms`**) |
