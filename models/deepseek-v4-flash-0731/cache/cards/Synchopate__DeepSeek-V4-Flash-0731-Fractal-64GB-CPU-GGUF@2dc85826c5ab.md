---
license: mit
library_name: gguf
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - gguf
  - deepseek-v4
  - mixture-of-experts
  - cpu-inference
  - quantization
  - llama-cpp
  - claude-code
  - avx512
  - local-llm
pipeline_tag: text-generation
quantized_by: Synchopate
---

# 🐋 DeepSeek-V4-Flash-0731-Fractal-64GB-CPU-GGUF (284.33B Parameters in 56.56 GiB)

### Break the 64 GB RAM Wall: Run the Full 284.33B DeepSeek-V4-Flash Frontier MoE at 256K Context on a Standard 64 GB DDR5 CPU Workstation (`57.61 GiB VmRSS`, 0 GPUs, 0 Swap)

🔗 **Engine, AVX-512BW/AVX2 Kernels & One-Click Agent CLI (`./modchip claude`):** [**https://github.com/synchopate/llama-modchip**](https://github.com/synchopate/llama-modchip)

---

## ⚡ Why This GGUF Exists (The 64 GB Linux OOM Wall)

Standard 1-bit/2-bit GGUF quantizations (`IQ1_S` / `IQ2_XXS`) of **284.33B DeepSeek-V4-Flash** (61 layers, 256 routed experts, top-8 active, MLA + Engram + MQA8 Sparse Indexer) occupy **76.87 GiB on disk**. On a 64 GB Linux workstation (~62.8 GiB usable physical RAM), loading a 76.87 GiB file either triggers the Linux OOM killer or forces `16+ GiB` of weights to stream from NVMe SSD swap/page-cache on every single token, collapsing throughput by **93% (`133 tok/s -> 9.6 tok/s`)**.

Using the open-source **`llama-modchip v2.4.0`** transcoder ([`scripts/transcode_dsv4.c`](https://github.com/synchopate/llama-modchip/blob/main/scripts/transcode_dsv4.c)), this model shrinks from **`76.87 GiB` down to `56.56 GiB` (`60,729,886,080` bytes, `-20.31 GiB` / `-26.4%`)**, allowing 100% of the 284.33B weights AND the 256K MLA KV-cache to reside in physical DDR5 RAM (**`57.61 GiB VmRSS`**, `6.39 GiB` free OS headroom, `0 bytes` of swap):

| Metric | Standard `IQ1_S` / `IQ2_XXS` Baseline | **`DeepSeek-V4-Flash-Fractal-64GB` (`v2.4.0`)** | Improvement |
| :--- | :--- | :--- | :--- |
| **GGUF File Size on Disk** | `76.87 GiB` *(OOM on 64 GB RAM)* | **`56.56 GiB`** (`60,729,886,080` B) | **`-20.31 GiB` (`-26.4%`)** |
| **Live Process `VmRSS` (`256K` Context)** | `> 64 GiB` *(16 GiB SSD thrashing)* | **`57.61 GiB`** (`60,408,048` kB, `0` Swap) | **100% Resident in 64 GB RAM** |
| **Pure CPU Prefill (`pp=403..580`, 16c Zen 5)** | `25.05 tok/s` | **`43.16 – 44.36 tok/s`** | **`+77.1%` (4-Level Cache Tiling)** |
| **Pure CPU Decode (`tg`, 16c Zen 5)** | `6.41 tok/s` | **`9.06 – 10.74 tok/s`** (`11.5–14.7` w/ PVS) | **`+55% – +129%`** |
| **Claude Code Turn-1 Warm Restore (`2,940` tok)** | `65,800 ms` *(Cold Prompt Eval)* | **`13.06 ms`** (`/dev/shm` Slot-0 KV Restore) | **`5,038x Faster`** |

---

## 🔬 How `56.56 GiB` Was Achieved Without Perplexity Collapse

1. **Analytical `SwiGLU-H2` Second-Order Hessian Cross-Projection Compression (`ffn_up_exps` & `ffn_gate_exps` $\rightarrow$ `IQ1_S` `1.56 bpw`, `-15.49 GiB`):**
   In a SwiGLU FFN $y = \text{SiLU}(W_g x) \odot (W_u x)$, quantization error in row $r$ of $W_{\text{up}}$ is amplified by the second moment of the coupled gate activation $\mathbb{E}[\text{SiLU}(z_{g,r})^2] = \frac{7}{20}\|W_g^{(r,:)}\|_2^2 \sigma_x^2$, while error in row $r$ of $W_{\text{gate}}$ is amplified by $\mathbb{E}[\text{SiLU}'(z_{g,r})^2] \cdot \|W_u^{(r,:)}\|_2^2 \approx \frac{31}{100}\|W_u^{(r,:)}\|_2^2 \sigma_x^2$. Weighting the 1-bit `IQ1_S` super-block optimization by these exact closed-form cross-projection row norms preserves mathematical and coding accuracy without requiring external calibration datasets.
2. **Fractal-64 (`GGML_TYPE_Q2_0`, `3.75 bpw`) MLA & Sparse Indexer Transcoding (`-4.82 GiB`):**
   Transcodes `blk.*.attn_k_b.weight` (`3.70 GiB` $\rightarrow$ `1.62 GiB`), `blk.*.indexer.wk.weight`, and `blk.*.indexer.weights_proj.weight` from `Q5_K` into 64-weight `Q2_0` blocks (`30 bytes / 64 weights`, fine-grained `FP16` scale + `FP16` zero-point per 64 weights via a 24-iteration Lloyd-Max solver).
3. **Dense Reasoning Backbone Preserved at High Precision (`Q5_K` / `Q6_K` / `F32`):**
   All dense MLA query/value projections (`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_v_b`, `attn_wo`), shared experts (`ffn_*_shexp`), dense layers (`blk.0..2`), and token embeddings remain untouched at `Q5_K` / `Q6_K` / `F32`.

---

## 🚀 Quickstart with `llama-modchip`

### 1. Check if your CPU & RAM can run this 284B model in 2 seconds: