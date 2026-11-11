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

# Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120

Pre-transcoded **Zero-Overhead (`0.0s` load transcode) Submodular Hybrid `Mode 2` (`2.69 bpw` effective MoE, `-16.64 GB` VRAM)** Blackwell (`sm_120`) checkpoint of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`180B` total parameters, `48` hybrid layers: `36` Gated DeltaNet `GDN` + `12` Qwen Sparse Attention `QSA`, `512` routed + `1` shared expert, `4`-stream mHC Hyper-Connections, and Layer-1 `28.80 GB` Zero-Copy N-Gram PLE).

Each layer stores **`96` Hot Experts** (`~70–75%` of routed token probability mass on math, code, and structured JSON) in lossless **`Mode 0` (`SWZ3_G32` 3.5 bpw INT8 Tensor Core)** and **`416` Cold Tail Experts** in **`Mode 2` (`2.50 bpw` 4D Wave-Lattice `VQ4`)**, selected via empirical submodular activation-weighted KL minimization (`0.001782` forward KL vs `Mode 0`, `97.39%` Top-1 match, `0.999907` logit cosine).

## Model Family

| Repository | Format & Target Runtime | GPU VRAM | Host RAM (PLE) | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-Hybrid-2.7bpw-SM120)** | Pre-transcoded `Mode 2` Hybrid `sm_120` `safetensors` (`2.69 bpw`) | **`50.06 GB`** | `28.80 GB` (`0 GB` VRAM) | **`3,068.49 tok/s`** (`3,027.69` exact router!) on 1× RTX PRO 6000; `0.0s` transcode; `-16.64 GB` VRAM |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw-SM120`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw-SM120)** | Pre-swizzled `Mode 0` `sm_120` `safetensors` (`3.5 bpw`) | `66.70 GB` | `28.80 GB` (`0 GB` VRAM) | **`2,676.97 tok/s`** lossless `Mode 0`; `64K` AIME 2025 `23/30` (`76.67%`, AIME II `14/15`) |
| **[`Qwen3.8-Flash-Next-WV-3.5bpw`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-WV-3.5bpw)** | 2-Shard `WV` Non-Linear `GGUF` (`3.5 bpw`) | `55.12 GB` | `28.80 GB` (`mmap`) | Master lossless `WV` container (`ΔPPL +0.040` on `<think>`, `0.09757` KLD) |
| **[`Qwen3.8-Flash-Next-3.5bpw-GGUF`](https://huggingface.co/Grawnch/Qwen3.8-Flash-Next-3.5bpw-GGUF)** | 2-Shard Standard `llama.cpp` `GGUF` (`3.5 bpw`) | `55.05 GB` | `28.80 GB` (`mmap`) | 100% upstream `ggml` (`Q3_K`/`Q4_0`/`Q2_0`/`Q6_K`) drop-in for `llama.cpp` |

---

## Single-GPU `v17` Throughput (`1× NVIDIA RTX PRO 6000 Blackwell`, `101.97 GB` GDDR7, `1.79 TB/s`)

Measured end-to-end across all `48` layers (`180B` parameters) with `v17` Split-KV Flash-Block-Diffusion (`2.11×` faster draft step at `B=48`: `11.19 ms → 5.30 ms`), Fused In-CTA Top-16 Drafter Head (`98.51%` Top-16 overlap), Hessian-Cholesky `row_lut_i8` `lm_head` calibration (`-44.45%` KL vs exact `Q6_K`: `0.006959 → 0.003866`), and `Expected-τ` Bellman DP (`non_eos_tok_per_s == raw_hw_tok_per_s` on every row):

| Workload / Benchmark | Batch (`B`) | Horizon (`k`) | Router / Head / Selector | Verified Throughput | Strict Capped (`tok/s`) | Mean `τ` (`ms/rnd`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`structured_json` (Peak Verified)** | `48` | `8` | `min_w = 0.015` | **`3,068.49 tok/s`** | `2,319.66 tok/s` | `5.410` (`84.63 ms`) |
| **`structured_json` (Exact Router + Peak Capped)** | `48` | `8 / 6` | **Exact (`min_w = 0.0`)** | **`3,027.69 / 3,023.91 tok/s`** | **`2,575.91 tok/s`** (`k=6`) | `5.319` (`84.32 / 70.77 ms`) |
| **`aime2025_math` (Peak Verified)** | `48` | `6` | `min_w = 0.015` + Calib Head | **`2,811.39 tok/s`** | `2,170.30 tok/s` | `4.643` (`79.27 ms`) |
| **`aime2025_math` (Exact Router + Peak Capped)** | `48` | `6` | **Exact (`0.0`) + `Expected-τ` DP** | **`2,775.76 tok/s`** | **`2,223.05 tok/s`** | `4.578` (`79.17 ms`) |
| **`cuda_sm120_code` (Peak Verified)** | `48` | `6` | `min_w = 0.015` | **`2,309.08 tok/s`** | `1,666.98 tok/s` | `3.745` (`77.84 ms`) |
| **`cuda_sm120_code` (Exact Top-10 Router)** | `48` | `6` | **Exact (`min_w = 0.0`)** | **`2,206.91 / 2,197.15 tok/s`** | **`1,727.75 tok/s`** | `3.576` (`77.77 ms`) |
| **`M = 1` Fused Burst (Zero Drafter)** | `48` | `1` | Exact (`min_w = 0.0`) | **`1,400.10 tok/s`** | **`1,400.10 tok/s`** | `1.000` (`34.28 ms`) |
| **`B = 1` Single-Stream (`DFlash2`)** | `1` | `8..12` | Exact (`min_w = 0.0`) | **`428.86 tok/s`** *(Math)* / **`257.61`** *(`SWZ4` DFlash2 Code)* | **`428.86 / 252.66 tok/s`** | `4.26 – 6.11` |

---

## Repository Contents

| File / Path | Size | Placement | Description |
| :--- | :---: | :---: | :--- |
| `globals.safetensors` | `1.66 GB` | GPU VRAM | Token embeddings, final mHC norms/projections, default `SWZ4_G32` LM head (`3,068.5 tok/s`), and Layer-1 PLE query/key/conv weights |
| `layers_00_05.safetensors` .. `layers_42_47.safetensors` | `8 × 5.54 GB` (`44.31 GB`) | GPU VRAM | Native pre-transcoded `Mode 2` Hybrid shards (`96` Hot `SWZ3_G32` + `416` Cold `VQ4` per layer; `-17.38 GB` shard bytes vs `Mode 0`) |
| `lm_head_calibrated_swz4.safetensors` | `366.2 MB` | Optional GPU | `v17` Hessian-Cholesky + Per-Row `row_lut_i8` calibrated `output.weight` (`KL 0.006959 → 0.003866` vs exact `Q6_K`, `96.88%` Top-1, **`2,811.4 tok/s` AIME math**) |
| `dflash2_swz4_calibrated.safetensors` | `241.8 MB` | Optional GPU | `v17` Hessian-Cholesky calibrated `SWZ4_G32` `DFlash2` 20-matrix drafter (`3.52×` smaller than `852 MB` `BF16`; `+7.9%` faster at `B=1`) |
| `qsa_indexer.safetensors` | `39.34 MB` | GPU VRAM | `12` QSA layers' `idx_qk` + `idx_qn` / `idx_kn` for `80K`-context Top-2048 sparse attention |
| `Qwen3.8-Flash-Next-WV-3.5bpw-00002-of-00002.gguf` | `28.80 GB` | Host Pinned RAM (`0 GB` VRAM) | Layer-1 N-Gram Per-Layer Embedding table (`cudaHostRegisterMapped` zero-copy PCIe gather) |
| `sm120_manifest.json` | `7.8 KB` | Host | Native `mode=2` (`num_hybrid_tensors=7`) layer manifest (`0.0s` load-time transcoding, `59.2s` cold-start load) |
| `v16_submodular_tier_maps_h96.pt` | `13.4 KB` | GPU (`int32`) | Submodular `H=96` Hot / `416` Cold expert tier maps across all `48` layers |
| `vq4_codebooks.pt` | `10.1 KB` | GPU Shared Mem | Calibrated `256 × 4` 4D Wave-Lattice VQ codebooks (`gu` & `dn`) |
| `config.json`, `generation_config.json`, `tokenizer*` | `12.8 MB` | Host | Hugging Face config & `248,320`-vocab tokenizer metadata |
| `benchmarks/*.json` | `288 KB` | Reference | `v17` (**`3,068.49 tok/s`**), `v16.1` (`2,974.38 tok/s`), `v15.1` (`2,500.27 tok/s`), and `64K` AIME 2025 (`23/30`) benchmark logs |
