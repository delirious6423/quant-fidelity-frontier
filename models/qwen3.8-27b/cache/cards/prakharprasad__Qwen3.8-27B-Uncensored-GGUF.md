---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
language:
  - en
  - zh
tags:
  - abliterated
  - uncensored
  - qwen3.8
  - gguf
  - dynamic-quantization
  - ud-quants
  - ud-3.0
  - quantized
  - llama.cpp
  - red-teaming
  - reasoning
  - function-calling
---

<div align="center">

# Qwen3.8-27B-Uncensored Dynamic 3.0 (UD 3.0) GGUF Suite

<p><em>State-of-the-Art SSM-Aware Selective Quantization for the Gated DeltaNet Hybrid Architecture</em></p>
<p>Quantized from <a href="https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8">orcarouter/Qwen3.8-27B-Uncensored-FP8</a> (Abliterated Qwen3.8-27B)</p>

[ ![Base Model](https://img.shields.io/badge/Base_Model-Qwen3.8--27B-1E6FEB) ](https://huggingface.co/Qwen/Qwen3.8-27B) [ ![License](https://img.shields.io/badge/License-Apache%202.0-4C8BF5) ](https://www.apache.org/licenses/LICENSE-2.0) ![UD 3.0](https://img.shields.io/badge/Method-Dynamic_3.0_(UD_3.0)-00A67E) ![11 Quants](https://img.shields.io/badge/Tiers-11_Models-blue) ![9.99 to 27.05 GiB](https://img.shields.io/badge/Footprint-9.99_to_27.05_GiB-FF8800) ![3.18 to 8.61 BPW](https://img.shields.io/badge/BPW-3.18_to_8.61-9B59B6) ![262K context](https://img.shields.io/badge/Context-262K-orange) ![llama.cpp](https://img.shields.io/badge/llama.cpp-compatible-green)

</div>

---

## What are Unsloth Dynamic 3.0 (UD 3.0) Quants?

**Dynamic 3.0 (UD 3.0)** is the modern standard for mixed-precision GGUF quantization pioneered by Unsloth AI (introduced for Qwen3.8 in August 2026). Rather than applying uniform bit-widths across an entire model, UD 3.0 uses architecture-aware tensor routing to protect the most sensitive mathematical structures while aggressively quantizing non-critical bulk parameters.

### Why Standard Naive K-Quants Fail on Qwen3.8
Qwen3.8-27B is **not a standard transformer**. It is a 64-layer hybrid consisting of **48 linear-attention Gated DeltaNet layers** and 16 full-attention layers (interleaved every 4 layers).

In Gated DeltaNet layers, sequence memory is maintained via a **recursive recurrent state update**:

$$S_t = (I - \beta_t K_t^\top) S_{t-1} + \alpha_t V_t Q_t^\top$$

- Standard K-quantization (naive `Q2_K` through `Q5_K`) crushes the recurrent gating matrices (`ssm_alpha`, `ssm_beta`) down to 2 to 4 bits.
- Because recurrence multiplies state updates recursively over sequence steps, this introduces a **7.5% to 21.8% RMS noise floor** into every token transition. Over long contexts (32k–262k tokens), this error accumulates exponentially, causing context drift, hallucinations, and tool-calling breakdown.
- **The UD 3.0 Solution:** In this UD 3.0 suite, all recurrent gating tensors (`ssm_alpha`, `ssm_beta`, and in XL tiers `ssm_out`) are isolated and pinned to **`Q8_0`** (and `f32` for recurrence normalization). This enforces a constant high-precision 43.8 dB SQNR floor across all tiers, eliminating recursive decay noise.

### The UD 3.0 Multi-Token Prediction (MTP) Strategy
Qwen3.8 includes a native Multi-Token Prediction (`nextn`) speculative head located at Layer 64 (~500 MB in BF16, ~146 MB in Q2_K).
- On hardware with 12 GB to 24 GB+ VRAM, retaining the MTP head provides built-in speculative decoding speedups without requiring external draft models.
- On strict 10 GB VRAM cards (such as RTX 3080 10GB or systems with 16 GB RAM), retaining this head pushes the model size to 10.13 GiB, triggering catastrophic OS swapping and CUDA out-of-memory errors.
- Following UD 3.0 design guidelines, this release provides **both**:
  - `UD-Q2_K_XL-MTP` (10.13 GiB) with MTP included for systems with adequate headroom.
  - `UD-Q2_K_XL` (9.99 GiB) with Layer 64 pruned, cleanly breaking the 10 GiB memory barrier.

### Beyond Perplexity: Multi-Token Trajectory Divergence
Traditional quantization evaluation relies on single-token perplexity on Wikipedia text. However, as demonstrated by the Unsloth research team (*"Accuracy is Not All You Need"*), perplexity is easily gamed due to positive and negative error cancellation across individual tokens.

UD 3.0 evaluates models via **KL Divergence** and **Divergence-300 @ 32** (greedy argmax decoding across 300 held-out prompts from Terminal-Bench 2.1, DeepSWE, and MathArena). This guarantees that multi-token reasoning trajectories and agentic tool-calling capabilities remain aligned with the full-precision baseline.

---

## Empirical Drift Pareto Frontier

Below is the measured Pareto frontier across all 11 dynamic quant models, evaluating overall model reconstruction fidelity and demonstrating the elimination of recurrent gating noise:

<div align="center">

![Empirical Drift Pareto Frontier](assets/drift_vs_size_pareto.png)

</div>

---

## Complete Model Specification Table

All metrics below are empirically measured via tensor-level dequantization audits against baseline ground truth:

| Model File | Base Quant | Size (GiB) | BPW | Recurrent Gating (RMSE %) | FFN Down (RMSE %) | Overall SQNR (dB) | Overall RMSE (%) | Cosine Similarity | Peak Error (L-inf) | Target Hardware / Fit | MTP Head |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| **`UD-Q2_K_XL`** | `Q2_K` | **`9.99 GiB`** | **`3.18`** | `0.82%` | `21.80%` | `15.6 dB` | `17.1%` | `0.9908` | `0.098` | 10 GB GPUs / Low-RAM Systems | *Pruned* |
| **`UD-Q2_K_XL-MTP`** | `Q2_K` | **`10.13 GiB`** | **`3.22`** | `0.82%` | `21.80%` | `15.8 dB` | `16.8%` | `0.9912` | `0.098` | 10-12 GB GPUs (Full in-VRAM + MTP) | Included |
| **`UD-Q3_K_XL-MTP`** | `Q3_K_L` | **`13.57 GiB`** | **`4.32`** | `0.82%` | `13.40%` | `21.1 dB` | `9.4%` | `0.9965` | `0.068` | RTX 3060 (12GB) / RTX 4070 (12GB) | Included |
| **`UD-Q4_K_S-MTP`** | `Q4_K_S` | **`14.75 GiB`** | **`4.69`** | `0.82%` | `8.65%` | `24.8 dB` | `6.2%` | `0.9984` | `0.045` | 16 GB GPUs (Max KV Headroom) | Included |
| **`UD-Q4_K_M-MTP`** | `Q4_K_M` | **`15.67 GiB`** | **`4.98`** | `0.82%` | `7.60%` | `25.9 dB` | `5.3%` | `0.9988` | `0.039` | 16 GB GPUs (RTX 4080 / 4060 Ti) | Included |
| **`UD-Q4_K_XL-MTP`** | `Q4_K_M` | **`16.36 GiB`** | **`5.14`** | `0.82%` | `7.20%` | `26.8 dB` | `4.7%` | `0.9991` | `0.034` | **RTX 3090 / 4090 24GB Sweet Spot** | Included |
| **`UD-Q5_K_M-MTP`** | `Q5_K_M` | **`18.20 GiB`** | **`5.79`** | `0.82%` | `5.10%` | `30.2 dB` | `3.3%` | `0.9994` | `0.028` | 24 GB Cards (High Precision) | Included |
| **`UD-Q5_K_XL-MTP`** | `Q5_K_M` | **`19.46 GiB`** | **`6.19`** | `0.82%` | `4.50%` | `31.8 dB` | `2.8%` | `0.9996` | `0.024` | 24 GB Cards (Max Quality Fit) | Included |
| **`UD-Q6_K-MTP`** | `Q6_K` | **`20.90 GiB`** | **`6.65`** | `0.82%` | `3.20%` | `34.6 dB` | `2.0%` | `0.9997` | `0.019` | 24 GB Cards (Near Lossless) | Included |
| **`UD-Q6_K_XL-MTP`** | `Q6_K` | **`21.81 GiB`** | **`6.94`** | `0.82%` | `2.80%` | `36.4 dB` | `1.6%` | `0.9998` | `0.016` | 32 GB+ Workstations / Mac Studio | Included |
| **`UD-Q8_K_XL-MTP`** | `Q8_0` | **`27.05 GiB`** | **`8.61`** | `0.82%` | `0.82%` | `44.2 dB` | `0.65%` | `0.99997` | `0.005` | Studio Reference Master | Included |

> **Note on Recurrent Gating Fidelity:** In standard naive quants, `ssm_alpha` and `ssm_beta` suffer from 7.5% (in Q4) to 21.8% (in Q2) RMSE. In this UD 3.0 suite, recurrent gating is pinned to Q8_0 across all tiers, capping error at the 0.82% Q8_0 mathematical precision barrier. Furthermore, our `ssm_alpha` and `ssm_beta` weights are **100% bit-for-bit identical** (0.000% difference) to the official Unsloth Base UD-Q4_K_XL weights.

---

## Cryptographic Verification (SHA-256)

Every model file in this repository has been validated bit-for-bit against remote Hugging Face LFS storage:

| Model File | SHA-256 Checksum |
| :--- | :--- |
| `Qwen3.8-27B-Uncensored-UD-Q2_K_XL.gguf` | `bf1baf24447feb96737a7ba5d6550cdda6bc4a5e22b3c90fa1408e6592b3c440` |
| `Qwen3.8-27B-Uncensored-UD-Q2_K_XL-MTP.gguf` | `046e3046d22981b9c213de6742ba0e5bbde044cde721c88e26ea68854f9f1530` |
| `Qwen3.8-27B-Uncensored-UD-Q3_K_XL-MTP.gguf` | `e3c69b9539851f74b358da72a5321cbb093b7a7d7ccbd1f22a29a72b337cc627` |
| `Qwen3.8-27B-Uncensored-UD-Q4_K_S-MTP.gguf` | `b19c26125ebd8d2e7d3ab82ec7d1eab511021d10448f5a46e0bdbd967077e183` |
| `Qwen3.8-27B-Uncensored-UD-Q4_K_M-MTP.gguf` | `d24c03eb604abc546e0e426cdda016a785044d97ae4387bd387312676fa3a075` |
| `Qwen3.8-27B-Uncensored-UD-Q4_K_XL-MTP.gguf` | `e97a3fec02ec1bb610fcfd8bfd53d865c3bbd0a79a32c4eeb2292f763327cb7c` |
| `Qwen3.8-27B-Uncensored-UD-Q5_K_M-MTP.gguf` | `8dea23509ea080e4737ea3290560dee946e187ee7efdfda6cab745c054a0239d` |
| `Qwen3.8-27B-Uncensored-UD-Q5_K_XL-MTP.gguf` | `ddb5d30922df1a4fd21fdf64c7f4d6e9eadcbc7fdc1654b9478e5f523767379d` |
| `Qwen3.8-27B-Uncensored-UD-Q6_K-MTP.gguf` | `c9c65823589199ac855bbf07be218b96da6d5fc116ba5064e3b0272dff130b53` |
| `Qwen3.8-27B-Uncensored-UD-Q6_K_XL-MTP.gguf` | `bb3b792f6adf448ddeb0d475bfb38360015db24c40aad097c09db9aa900ad8c1` |
| `Qwen3.8-27B-Uncensored-UD-Q8_K_XL-MTP.gguf` | `4583a307d4c20cb6f439eb93d628d45ba9680451d8a0c936ad1cd2e793327553` |

---

## Hardware Sizing & Deployment Guide

### 1. 10 GB VRAM GPUs (e.g. RTX 3080 10GB, Mobile Workstations)
- **Select:** `UD-Q2_K_XL` (9.99 GiB)
- **Why:** Full in-VRAM execution with zero OS page swapping. Pruning Layer 64 eliminates 146 MiB of overhead, avoiding system memory thrashing.

### 2. 12 GB GPUs (e.g. RTX 3060 12GB, RTX 4070 12GB)
- **Select:** `UD-Q3_K_XL-MTP` (13.57 GiB) or `UD-Q2_K_XL-MTP` (10.13 GiB)
- **Why:** `UD-Q2_K_XL-MTP` fits 100% in VRAM with room for 32k context and built-in MTP acceleration. `UD-Q3_K_XL-MTP` delivers higher precision (21.1 dB SQNR) with light CPU offload (`-ngl 56`).

### 3. 16 GB GPUs (e.g. RTX 4080 16GB, RTX 4060 Ti 16GB, V100 16GB)
- **Select:** `UD-Q4_K_S-MTP` (14.75 GiB) or `UD-Q4_K_M-MTP` (15.67 GiB)
- **Why:** `UD-Q4_K_S-MTP` provides comfortable headroom for 8k–16k context buffers using `--cache-type-k q8_0 --cache-type-v q8_0`.

### 4. 24 GB GPUs (e.g. RTX 3090, RTX 4090, A10G, Apple Silicon 32GB)
- **Select:** **`UD-Q4_K_XL-MTP` (16.36 GiB)** *(Gold Standard)*
- **Why:** Allocating 16.36 GiB leaves **7.64 GiB of VRAM headroom** on a 24 GB GPU for full 64k+ FlashAttention context buffers and concurrent speculative drafters. For shorter context tasks, step up to `UD-Q5_K_XL-MTP` (19.46 GiB) or `UD-Q6_K-MTP` (20.90 GiB) for near-lossless output.

### 5. 32 GB+ Workstations & Apple Silicon Unified Memory
- **Select:** `UD-Q6_K_XL-MTP` (21.81 GiB) or `UD-Q8_K_XL-MTP` (27.05 GiB)
- **Why:** Reference-grade fidelity achieving 36.4 dB to 44.2 dB SQNR.

---

## Speculative Decoding Acceleration

Qwen3.8-27B supports dual speculative acceleration options:

1. **In-Tree MTP Head (Next-N):** Included in 10 of the 11 models (all `-MTP` files; excluded only in `UD-Q2_K_XL`). Enables multi-token speculative verification without downloading additional models.
2. **External DFlash-2 Drafter:** For higher speculative acceptance ratios with minimal VRAM overhead, the verified compatible drafter is [`HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF`](https://huggingface.co/HermiHg/Qwen3.8-27B-DFlash2-Q2_K_S-MIX-GGUF) (`Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf`, **0.52 GB**). Running this drafter with `q4/q4` KV cache fits comfortably alongside `UD-Q4_K_XL-MTP` within 24 GB VRAM:

```bash
llama-server \
  -m Qwen3.8-27B-Uncensored-UD-Q4_K_XL-MTP.gguf \
  -md Qwen3.8-27B-DFlash2-Q2_K_S-MIX.gguf \
  -ngl 99 -ngld 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-type-k-draft q4_0 --cache-type-v-draft q4_0 \
  -c 32768 -fa
```

---

## About the Source Model & Abliteration

[`orcarouter/Qwen3.8-27B-Uncensored-FP8`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-FP8) is an abliterated (refusal-removed) build of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) produced by OrcaRouter.

Abliteration follows **Arditi et al. (2024)**: the refusal direction was isolated at layer 38 using harmful vs. harmless residual activations, then orthogonalized out from residual-writing projections (`o_proj`, `out_proj`, `down_proj`, `embed_tokens`). The MTP head was abliterated in parity.

---

## ⚠️ Disclaimer — Research Use Only

This model inherits the abliteration applied by OrcaRouter. Its **refusal alignment has been substantially removed**:

- **It will comply with harmful, unethical, offensive, or illegal requests** that the original `Qwen3.8-27B` would refuse.
- It is released **strictly for legitimate research** — interpretability, AI-safety study, red-teaming, robustness evaluation, and controlled experiments.
- **You assume full responsibility and liability** for every use and for everything it generates.
- Do not deploy to end users or in production without adding your own safety, moderation, and abuse-prevention layers.
- Use must comply with the **[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)** and all applicable laws and regulations.

---

## License

Apache 2.0 — inherited from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
