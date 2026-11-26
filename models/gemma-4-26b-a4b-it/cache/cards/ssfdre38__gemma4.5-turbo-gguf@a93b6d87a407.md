---
license: apache-2.0
base_model:
  - google/gemma-4-E2B-it
  - google/gemma-4-E4B-it
  - google/gemma-4-26B-A4B-it
  - google/gemma-4-31B-it
tags:
  - gemma4
  - gemma4.5
  - gguf
  - iq4_xs
  - quarot
  - hadamard
  - multimodal
  - vision
  - cpu-inference
  - ollama
  - llama-cpp
  - haven-cpp
  - tpu
  - trillium
  - cuda
  - nvidia-l4
---

# Gemma 4.5 Turbo (`ssfdre38/gemma4.5-turbo`)

**Algebraically optimized Gemma 4 models built from unquantized BF16 source weights via [`haven-bake`](https://github.com/ssfdre38/haven-cpp) — lower perplexity, smaller footprint, faster CPU inference, verified across NVIDIA L4 24GB GPU (100% VRAM-resident for all 4 sizes) and Google Cloud TPU v6e Trillium (up to 18,512 tok/s), with full multimodal vision + thinking preserved.**

> **Zero retraining. Pure exact linear algebra baked directly into BF16 weights before `IQ4_XS` quantization.**

## Quick Start (Ollama — Multimodal Vision + Thinking)

All Ollama tags use the universal stock-compatible `--mode stock` Gemma 4.5 weights merged with the full multimodal vision encoder (`mmproj`), so they run out of the box on stock **Ollama**, **`llama.cpp`**, and **`haven-cpp`**:

```bash
ollama run ssfdre38/gemma4.5-turbo          # e4b (5.63 GiB, recommended, multimodal)
ollama run ssfdre38/gemma4.5-turbo:e2b      # 3.99 GiB, 8 GB+ RAM (multimodal)
ollama run ssfdre38/gemma4.5-turbo:e4b      # 5.63 GiB, 10 GB+ RAM (multimodal)
ollama run ssfdre38/gemma4.5-turbo:26b      # 14.21 GiB, 26B-A4B MoE (multimodal)
ollama run ssfdre38/gemma4.5-turbo:31b      # 16.82 GiB, 31B Dense flagship (multimodal)
```

Vision works out of the box:
```bash
ollama run ssfdre38/gemma4.5-turbo "describe this image" /path/to/image.jpg
```

---

## What Changed in Gemma 4.5 Turbo?

Gemma 4 introduces several architectural quirks that waste memory or create severe 4-bit quantization cliffs when quantized naively:
- **Dead placeholder shared-KV tensors:** E2B (`shared_kv_layers=20`) and E4B (`shared_kv_layers=18`) ship 60 and 54 unused `attn_k`, `attn_v`, and `attn_k_norm` tensors on shared-KV layers.
- **Unfolded MoE router & expert scales:** 26B-A4B ships separate `ffn_gate_inp.scale` and `ffn_down_exps.scale` multipliers.
- **Attention & activation outlier cliffs:** Learned RMSNorm gammas reach `450x–900x`, causing severe clipping in 4-bit `IQ4_XS` weight blocks.

Our open-source C++20 AVX2 optimizer **`haven-bake`** resolves these at BF16 bake time *prior* to `llama-quantize IQ4_XS`:

1. **Universal Stock Edition (`gemma4.5-*-iq4xs.gguf` & `*-mmproj.gguf` — 100% Ollama & `llama.cpp` compatible):**
   - Strips dead placeholder shared-KV tensors (`60` tensors / `37.8 MB` BF16 on E2B; `54` tensors / `33.6 MB` IQ4_XS on E4B).
   - Folds MoE `ffn_gate_inp.scale` and `ffn_down_exps.scale` cleanly into weights (`26B-A4B`).
   - Bakes **zero-cost QuaRot R2 head-dimension Hadamard rotation (`V → O`)** on `E4B` (`24/24` KV-source layers), `26B-A4B` (`25/30` SWA layers, automatically skipping `5` `k_eq_v` layers), and `31B` (`50/60` SWA layers, automatically skipping `10` `k_eq_v` layers). Because Gemma 4's V-RMSNorm is parameter-free and `H` is orthogonal, `V → O` cancels identically at runtime with **0 extra operations** while smoothing 4-bit quantization noise. *(On E2B's single-KV-head geometry, `--no-r2` proved strictly optimal for stock perplexity.)*
   - Merges the full multimodal vision encoder (`merge_g45_mmproj.py`) without re-quantizing or disturbing LM weights.

2. **Native `haven-cpp` Turbo Edition (`gemma4.5-*-haven-turbo-iq4xs.gguf`):**
   - Everything in Stock, PLUS **norm-preserving orthogonal Block-256 / Block-64 Fast Walsh-Hadamard Transform (FWHT)** baked into `Q/K/V`, `FFN Gate/Up`, `FFN Down`, and `MoE` projections (`gemma4.turbo.*` metadata auto-dispatched by [`haven-cpp`](https://github.com/ssfdre38/haven-cpp) AVX2 CPU and OpenXLA PJRT TPU kernels).
   - Flattens both weight-block crest factors and `450x–900x` runtime activation cliffs by `16x`, slashing KL divergence vs unquantized BF16 by **19% to 44%** and aligning natively with TPU v6e Hadamard W8A8 INT8 quantization!

---

## Verified Benchmarks (CPU-Only: Intel Xeon E-2236, 6C/12T AVX2, No GPU)

### 1. Stock `llama.cpp` Perplexity (`llama-perplexity`, `calibration_data.txt`, `n_ctx=512`, `t=12`)

Every single model size achieves **equal or strictly lower perplexity** than original `gemma4-turbo` with **zero runtime overhead**:

| Model | Original `gemma4-turbo` PPL | **Gemma 4.5 Stock PPL** | Improvement | Text-Only GGUF | Multimodal (`mmproj`) GGUF | vs Stock Ollama |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E2B** | `8.6825` | **`8.6825`** *(0.0000 loss, 60 dead KV stripped)* | Faster (`25.5s` vs `26.8s`), `-11 MiB` RAM | `3.07 GiB` | **`3.99 GiB`** | `-45%` *(was 7.2 GB)* |
| **E4B** | `6.2341` | **`5.9685`** *(54 dead KV stripped + 24/24 R2)* | **`-0.266` PPL better** (all 3 chunks) | `4.71 GiB` | **`5.63 GiB`** | `-41%` *(was 9.6 GB)* |
| **26B-A4B MoE** | `33.9742` | **`33.0918`** *(folded scales + 25/30 SWA R2)* | **`-0.882` PPL better** | `13.10 GiB` | **`14.21 GiB`** | `-16%` *(was 17 GB)* |
| **31B Dense** | `11.7080` | **`10.9936`** *(50/60 SWA R2)* | **`-0.714` PPL better** (both chunks) | `15.70 GiB` | **`16.82 GiB`** | `-11%` *(was 19 GB)* |

### 2. Live Ollama CPU Throughput (`POST /api/chat`, Intel Xeon E-2236, 6C/12T, No GPU)

Verified end-to-end on stock Ollama (`think: false`, `num_ctx: 2048`, exact clean stop):

| Ollama Tag | Size | Prefill (`tok/s`) | Decode (`tok/s`) | Status |
| :--- | :---: | :---: | :---: | :---: |
| `ssfdre38/gemma4.5-turbo:e2b` | `4.3 GB` | **`116.91 tok/s`** | **`18.89 tok/s`** | Verified |
| `ssfdre38/gemma4.5-turbo:e4b` (`:latest`) | `6.0 GB` | **`61.65 tok/s`** | **`9.93 tok/s`** | Verified |
| `ssfdre38/gemma4.5-turbo:26b` | `15 GB` | **`55.29 tok/s`** | **`11.63 tok/s`** | Verified |
| `ssfdre38/gemma4.5-turbo:31b` | `18 GB` | **`8.81 tok/s`** | **`1.80 tok/s`** | Verified |

### 3. Native `haven-cpp` Turbo vs Unquantized BF16 Ground Truth

Measured against the original unquantized BF16 masters on CPU (`haven-cpp` AVX2 L2-tiled integer GEMM + fused FWHT):

| Model (`haven-turbo`) | Metric vs Unquantized BF16 | Original `gemma4-turbo` | **Gemma 4.5 Native Turbo** | Gain | Prefill (`S=274`, 12T) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **E2B Turbo** | `KL(bf16 \|\| q)` | `1.4460` | **`0.8139`** | **`-43.7%` less KL** | **`108.7 tok/s`** *(Stock: `113.2 tok/s`, TTFT `2.42s`)* |
| **E2B Turbo** | `forward_batch` Cosine vs BF16 | `0.8536` | **`0.9818`** | **`8.0x` less error** | Decode: `16.1 tok/s` |
| **E4B Turbo** | `KL(bf16 \|\| q)` | `0.8067` | **`0.6523`** | **`-19.1%` less KL** | **`58.64 tok/s`** *(TTFT `4.67s`)* |
| **E4B Turbo** | `forward` Cosine / Top-1 Agree | `0.9904` / `62.5%` | **`0.9983` / `72.9%`** | **`5.6x` less error / `+10.4%` Top-1** | Decode: `9.37 tok/s` |
| **26B-A4B Turbo** | `seq` vs `batch` Cosine / Top-1 | — | **`0.9998` / Exact `100==100`** | Clean English + FWHT | **`58.70 tok/s`** *(TTFT `4.67s`, `5.07x` speedup)* |
| **31B Dense Turbo** | `seq` vs `batch` Cosine / Top-1 | `0.9991` *(Stock)* | **`0.9998` / Exact `100==100`** | `max_abs_diff` cut `1.46 → 0.83` (`-43%`) | **`7.57 tok/s`** *(Decode: `1.69 tok/s` vs `1.62` Stock)* |

### 4. Native `haven-cpp` Pure C++20 Cloud TPU v6e-4 (Trillium) Silicon Benchmarks

Verified across all four Gemma 4.5 Turbo models on **4x Google Cloud TPU v6e-4 Trillium** (128 GB HBM3, 2x2 Optical ICI Torus, zero Python runtime via OpenXLA PJRT C ABI in [`haven-cpp`](https://github.com/ssfdre38/haven-cpp) commit `d0a4077`):

| Model (`haven-turbo`) | Layers / HBM3 Footprint (BF16 / INT8) | `B=1` Interactive Decode | vs Xeon CPU | `S=128` MXU Prefill | Peak Cloud Batching | 262k Logit Parity vs F32 CPU | Top-1 Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **E2B Turbo** | 35L + Pleias (`5.4 / 2.9 GB`) | **`406.3 tok/s`** *(1-chip: `251.4`)* | **19.2x** | **`43,858 tok/s`** (`2.92 ms`) | **`18,512.4 tok/s`** (`B=128` BF16) | **`0.999991 – 1.000000`** | **11/11 EXACT** |
| **E4B Turbo** | 42L + Pleias (`9.3 / 5.0 GB`) | **`343.2 tok/s`** *(1-chip: `184.7`)* | **31.8x** | **`30,415 tok/s`** (`4.21 ms`) | **`16,101.4 tok/s`** (`B=128` INT8) | **`0.999979`** | **11/11 EXACT** |
| **26B-A4B MoE Turbo** | 30L 128E Top-8 (`47.1 / 23.7 GB`) | **`201.6 tok/s`** *(BF16: `180.2`)* | **15.2x** | **`14,334 tok/s`** (`8.93 ms`) | **`9,886.9 tok/s`** (`B=256` W8A8) | **`0.999992`** *(W8A8: **`0.999925`**)* | **8/8 EXACT** |
| **31B Dense Turbo** | 60L Dense (`57.2 / 28.7 GB`) | **`110.8 tok/s`** *(BF16: `66.8`)* | **65.6x** | **`9,637 tok/s`** (`13.28 ms`) | **`2,673.1 tok/s`** (`B=64` INT8) | **`0.999921`** *(INT8: **`0.998900`**)* | **7/7 EXACT** |

> All 37/37 TPU configurations across all 4 models achieved **100% exact Top-1 token agreement** against F32 CPU ground truth (`0.9989 – 1.0000` full 262,144-vocab cosine similarity).

### 5. Google Cloud 1x NVIDIA L4 24GB GPU Benchmarks (`72W` TDP, Ada Lovelace `sm_89`, CUDA 13.0)

Verified on serverless Google Cloud Run **1x NVIDIA L4 24GB** (`23,034 MiB` VRAM, `300 GB/s` GDDR6, `llama.cpp` CUDA `9c003a9`, `-ngl 99 -fa 1`, `r=3` repetitions — **all 4 models fit 100% inside 24 GB VRAM with zero CPU offload**):

| Model (`IQ4_XS - 4.25 bpw`) | Size | Active Params | L4 `pp128` Prefill | L4 `pp512` Prefill | L4 `tg128` Decode | vs Xeon CPU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Gemma 4.5 E2B** | `3.06 GiB` | `4.63 B` | `8,726.3 tok/s` | **`17,232.8 tok/s`** | **`125.80 tok/s`** | **6.7x** |
| **Gemma 4.5 E4B** | `4.70 GiB` | `7.46 B` | `4,749.6 tok/s` | **`7,504.4 tok/s`** | **`72.05 tok/s`** | **7.3x** |
| **Gemma 4.5 26B-A4B MoE** | `13.08 GiB` | `~4 B / 25.23 B` | `1,654.7 tok/s` | **`3,159.2 tok/s`** | **`83.04 tok/s`** | **7.1x** |
| **Gemma 4.5 31B Dense** | `15.69 GiB` | `30.70 B` | `733.3 tok/s` | **`848.3 tok/s`** | **`14.28 tok/s`** *(80.2% GDDR6 BW)* | **7.9x** |

> **Unmodified `haven-cpp` Pure C++20 on NVIDIA CUDA (`xla_cuda_plugin.so`, PJRT `0.70`, zero Python runtime):** Running [`haven-cpp`](https://github.com/ssfdre38/haven-cpp) (`d0a4077`) directly against OpenXLA's CUDA PJRT plugin on the same NVIDIA L4 verified **6/6 async multi-slot parity on E2B Turbo (`0.999974 – 1.000000` 262k-vocab cosine vs F32 CPU, exact Top-1 `6372 == 6372`)** and **`3,332 tok/s`** BF16 `S=128` prefill.

---

## Included Files on Hugging Face (`ssfdre38/gemma4.5-turbo-gguf`)

### Universal GGUFs (Ollama / `llama.cpp` / LM Studio / `haven-cpp`)
- **Multimodal (Vision + Text + Thinking):**
  - `gemma4.5-e2b-iq4xs-mmproj.gguf` (`3.99 GiB`)
  - `gemma4.5-e4b-iq4xs-mmproj.gguf` (`5.63 GiB`)
  - `gemma4.5-26b-iq4xs-mmproj.gguf` (`14.21 GiB`)
  - `gemma4.5-31b-iq4xs-mmproj.gguf` (`16.82 GiB`)
- **Text-Only (Ultra-lean, no vision tower):**
  - `gemma4.5-e2b-iq4xs.gguf` (`3.07 GiB`)
  - `gemma4.5-e4b-iq4xs.gguf` (`4.71 GiB`)
  - `gemma4.5-26b-iq4xs.gguf` (`13.10 GiB`)
  - `gemma4.5-31b-iq4xs.gguf` (`15.70 GiB`)

### Native `haven-cpp` Block-FWHT Turbo GGUFs
> Requires [`haven-cpp`](https://github.com/ssfdre38/haven-cpp) (commit `d0a4077` or later for full CPU AVX2 + Cloud TPU v6e support across all 4 sizes) which automatically detects `gemma4.turbo.*` metadata and runs fused AVX2 FWHT / TPU StableHLO kernels. Do **not** load `*-haven-turbo-*` files in stock `llama.cpp`/Ollama (use the universal files above instead).

- `gemma4.5-e2b-haven-turbo-iq4xs.gguf` (`3.07 GiB`)
- `gemma4.5-e4b-haven-turbo-iq4xs.gguf` (`4.71 GiB`)
- `gemma4.5-26b-haven-turbo-iq4xs.gguf` (`13.10 GiB`)
- `gemma4.5-31b-haven-turbo-iq4xs.gguf` (`15.70 GiB`)

---

## Reproducibility

Built with open-source tooling:
- **Architecture Optimizer & TPU Runtime:** `haven-bake` & `haven-cpp` ([ssfdre38/haven-cpp](https://github.com/ssfdre38/haven-cpp))
- **Quantizer:** `llama.cpp` b9050 `llama-quantize` (`IQ4_XS`)
- **Multimodal Merger:** `merge_g45_mmproj.py` ([ssfdre38/gemma4-turbo](https://github.com/ssfdre38/gemma4-turbo))

## License

Derived from Google's Gemma 4 IT model family (`google/gemma-4-{E2B,E4B,26B-A4B,31B}-it`) under the **Apache 2.0 License** and [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
Optimization, baking, and benchmarking by [ssfdre38](https://ollama.com/ssfdre38).
