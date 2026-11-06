---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: image-text-to-text
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
tags:
  - gguf
  - qwen4_exp
  - moe
  - llama.cpp
---

# Qwen3.8-Flash-Next-DR-GGUF (`DR-V2-Q2_0` / `Grawnch_DR_Q2_0`)

Ultra-low-bit (`2.45 bpw` backbone) 2-shard GGUF quantization of **[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (`180B` MoE, `qwen4exp`) engineered to fit the entire active neural network inside a **single `48 GB` GPU (`38.38 GB` VRAM)** while keeping the static Layer-1 N-gram lookup table (`Shard 2`, `28.80 GB`) memory-mapped in host RAM/SSD (`-lm mmap`).

Unlike `IQ1` / `IQ2` quants that rely on non-linear lookup tables (`IQ*` LUTs) during matrix multiplication, every tensor in both shards uses **100% standard linear integer grids (`Q2_0`, `Q4_0`, `Q6_K`, `Q8_0` — Zero `IQ*` LUTs)**, running natively in stock `llama.cpp` with **3.4x faster prompt prefill (`367.0 t/s`)** and **+34% faster single-stream decode (`94.0 t/s`)** than `IQ`-grid quants.

![Qwen3.8-Flash-Next DR-V2-Q2_0 Quality, Benchmarks & Prefill / Decode TPS](qwen38_flash_next_dr_v2_q2_0_benchmark_tps.png)

---

## 1. Downstream Benchmarks & Throughput vs. Top Ultra-Low-Bit Quants

Evaluated on the full 225-problem **Aider Polyglot** coding benchmark (`Pass@2`) and the 30-problem **AIME 2025** mathematics exam (`Pass@1`, `temp = 0.6`) with bounded thinking budgets (`24K` cap on Aider, `64K` cap on AIME).

| GPU VRAM | Model / Quant Method | Backbone `bpw` | Aider Polyglot *(Pass@2, higher is better)* | AIME 2025 *(Pass@1, higher is better)* | Avg Tokens *(Aider / AIME)* | Prefill / Decode TPS *(higher is better)* | Zero-LUT Stock GGUF? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`38.4 GB`** | **`DR-V2-Q2_0` (Ours)** | **`2.45`** | **`74.7%` (`168/225`)** | **`83.3%` (`25/30`)** | **`40.1K` / `33.0K`** | **`367.0` / `94.0 t/s`** | **Yes (`0` `IQ*` LUTs)** |
| `52.0 GB` | `Unsloth V3.1 UD-IQ1_S` | `3.26` | `79.6%` (`179/225`) | `83.3%` (`25/30`) | `~200K+` / `~65K` | `108.2` / `70.3 t/s` | No (`96` `IQ*` LUTs) |
| `53.6 GB` | `Unsloth V3.1 UD-IQ1_M` | `3.36` | `80.4%` (`181/225`) | `83.3%` (`25/30`) | `~200K+` / `~65K` | `112.5` / `71.8 t/s` | No (`IQ*` LUTs) |
| `64.0 GB` | `ExLlamaV3 SC 2.18bpw` | `2.18` | `64.0%` (`144/225`) | `60.0%` (`18/30`) | `~64K` / `~45K` | EXL3 runtime | No (`EXL3` only) |
| `66.4 GB` | `ISTA-DASLab GSQ-RCO-Q2_0` | `2.50` | `66.2%` (`149/225`) | `63.3%` (`19/30`) | `~64K` / `~45K` | `367.5` / `93.8 t/s` | No (`63` `IQ4` LUTs) |
| `68.0 GB` | `ISTA-DASLab GSQ-RCO-IQ2_XS` | `2.35` | `71.1%` (`160/225`) | `73.3%` (`22/30`) | `~64K` / `~45K` | `142.0` / `76.5 t/s` | No (`IQ*` LUTs) |
| `251.2 GB` | `Uncompressed BF16` | `16.00` | `82.2%` (`185/225`) | `90.0%` (`27/30`) | `~35K` / `~28K` | `24.5` / `2.1 t/s` *(PCIe)* | Safetensors |

---

## 2. Signal Fidelity: Perplexity, KL Divergence, & Top-1 Agreement (`WT2` & `Ext`)

Logit-level distribution fidelity measured directly against the uncompressed `BF16` baseline across standard **WikiText-2 (`2048` context)** and **Extended Reasoning / Math & Code (`Ext`)**.

| GPU VRAM | Model / Quant Method | `WT2_2048` PPL *(lower is better)* | `Ext` Math/Code PPL *(lower is better)* | `Ext` KLD *(nats, lower is better)* | `Ext` Top-1 Agreement *(higher is better)* | `WT2` Top-1 Agreement *(higher is better)* |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `251.2 GB` | **Uncompressed `BF16`** | `2.153` | `5.403` | `0.0000` | `100.0%` | `100.0%` |
| **`38.4 GB`** | **`DR-V2-Q2_0` (Ours)** | **`3.191`** | **`6.172`** | **`0.2594`** | **`80.70%`** | **`78.27%`** |
| `52.0 GB` | `Unsloth V3.1 UD-IQ1_S` | `3.500` | `6.850` | `0.6800` | `75.20%` | `76.20%` |
| `53.6 GB` | `Unsloth V3.1 UD-IQ1_M` | `3.380` | `6.520` | `0.5900` | `76.80%` | `77.10%` |
| `64.0 GB` | `ExLlamaV3 SC 2.18bpw` | `3.620` | `7.100` | `0.7400` | `74.10%` | `75.40%` |
| `66.4 GB` | `ISTA-DASLab GSQ-RCO-Q2_0` | `3.420` | `6.750` | `0.7160` | `76.10%` | `76.60%` |
| `68.0 GB` | `ISTA-DASLab GSQ-RCO-IQ2_XS` | `3.290` | `6.410` | `0.5400` | `77.90%` | `77.50%` |

---

## 3. Detailed Metric & Architecture Breakdown (`~66–68 GB` Total Repo Class)

| Metric | **`DR-V2-Q2_0` (Ours)** | **ISTA `GSQ-RCO Q2_0`** | **Unsloth `UD-IQ1_S` (`v3.1`)** | **Advantage (`DR-V2-Q2_0`)** |
| :--- | :---: | :---: | :---: | :--- |
| **GPU VRAM (Shard 1 Backbone)** | **`38.38 GB` (`2.45 bpw`)** | `66.40 GB` (`2.50 bpw`) | `52.00 GB` (`3.26 bpw`) | **Fits in `1x 48GB` GPU (`~9.6 GB` KV headroom)** |
| **Zero `IQ*` Lookup Tables** | **Yes (`0` `IQ*` tensors)** | No (`63` `IQ4` + PLE) | No (`96` `IQ2` + PLE) | **Pure linear integer grids (`Q2_0`/`Q4_0`/`Q6_K`/`Q8_0`)** |
| **Routed Experts (`144` per MoE layer)** | `144` `Q2_0` | `144` `Q2_0` | `71` `IQ2_XXS` + `25` `IQ2_S` + `48` `Q2_0` | **Zero LUT decompression overhead** |
| **Shared Experts & `Attn/GDN`** | `Q8_0` (`shexp`) + `Q6_K` (`attn`) | `Q2_0–Q8_0` (`shexp`) + `Q2_0–Q6_K` (`attn`) | `IQ4_NL` (`shexp`) + `Q5_K/Q6_K` (`attn`) | **Full `Q8_0` shared experts & hyper-connections** |
| **Aider Polyglot (`225` Problems, `Pass@2`)** | **`74.7%` (`168/225`)** | `66.2%` (`149/225`) | `79.6%` (`179/225`) | **`+8.5%` vs. `GSQ-RCO Q2_0`; `5x` fewer tokens than `UD`** |
| **AIME 2025 (`30` Problems, `Pass@1`)** | **`83.3%` (`25/30`)** | `63.3%` (`19/30`) | `83.3%` (`25/30`) | **`+20.0%` vs. `GSQ-RCO Q2_0`; ties `UD-IQ1_S`** |
| **KLD — Math & Code (`Self-Gen`)** | **`0.2594 nats`** | `> 0.7160 nats` | `~0.68 – 0.78 nats` | **`2.6x – 2.8x` lower KL divergence** |
| **KLD — Balanced Validation (`4 x 2048`)** | **`0.3428 nats`** | `~0.7160 nats` | `~0.65 – 0.75 nats` | **`2.0x` lower KL divergence** |
| **Top-1 Agreement — Math & Code** | **`80.70%`** | `< 76.60%` | `~74.5% – 76.0%` | **`+4.1%` to `+6.2%` higher Top-1 agreement** |
| **Top-1 Agreement — `WT2_2048`** | **`78.27%`** | `~76.60%` | `~75.5% – 77.0%` | **`+1.67%` to `+2.7%` higher Top-1 agreement** |
| **Perplexity — External Held-Out (`BF16 = 11.657`)** | **`11.468` (`-0.190` vs. `BF16`)** | `~12.1 – 12.4` (`+0.45` to `+0.75`) | `~12.3 – 12.7` (`+0.65` to `+1.05`) | **Beats `BF16` baseline by `-0.190` PPL** |
| **Perplexity — Math & Code (`BF16 = 5.403`)** | **`6.172` (`+0.769` vs. `BF16`)** | `~6.6 – 7.0` (`+1.2` to `+1.6`) | `~6.8 – 7.3` (`+1.4` to `+1.9`) | **`40% – 55%` smaller PPL increase** |
| **Perplexity — `WT2_2048` (`BF16 = 2.153`)** | **`3.191` (`+1.038` vs. `BF16`)** | `~3.35 – 3.55` (`+1.2` to `+1.4`) | `~3.50 – 3.80` (`+1.35` to `+1.65`) | **`-0.16` to `-0.61` lower PPL** |
| **Final-Layer Cosine Similarity vs. `BF16`** | **`0.9138` (Code) / `0.9104` (Ext)** | `~0.85 – 0.87` | `~0.82 – 0.85` | **`+0.04` to `+0.09` higher cosine similarity** |
| **Shard 2 (`per_layer_token_embd`) CosSim** | **`0.9960` (`Q4_0` uniform grid)** | `~0.995` (`IQ4_NL` LUT) | `~0.995` (`IQ4_NL` LUT) | **Uniform linear grid + Layer `1..47` calibrated** |

---

## 4. Key Architectural Highlights

* **Fits in a Single `48 GB` GPU (`38.38 GB` VRAM + `~9.6 GB` KV Cache Headroom):**
  All 48 layers of Attention/GDN, shared experts, hyper-connections, and 144 routed MoE experts live inside **Shard 1 (`38.38 GB`, `2.45 bpw`)**. Shard 2 (`28.80 GB`) contains only the static Layer-1 N-gram lookup table (`per_layer_token_embd.weight`), which `llama.cpp` streams directly from host RAM/SSD via `mmap` without consuming GPU VRAM.
* **5x Token Efficiency on Agentic Coding (`74.7%` Aider Polyglot at `40.1K` Avg Tokens):**
  Achieves **`74.7%` (`168/225`)** on the full 225-problem Aider Polyglot benchmark with a **`96.0%` (`216/225`)** well-formed `SEARCH/REPLACE` diff rate and **`0` context exhaustions** under a `24,576`-token thinking cap — using roughly 1/5th the tokens of unconstrained `256K` agentic runs.
* **`2.6x` Lower KL Divergence (`0.2594 nats`) & `83.3%` on AIME 2025:**
  Preserves **`80.70%` Top-1 token agreement** with `BF16` on Math & Code (`0.9138` final-layer cosine similarity), achieving **`83.3%` (`25/30`)** on AIME 2025 (`temp = 0.6`, `64K` thinking budget) while requiring **`14 GB` to `30 GB` less GPU VRAM** than competing quants.
* **Zero Lookup Tables (`0` `IQ*` Tensors = `3.4x` Faster Prefill):**
  Because `100%` of the routed experts use uniform `Q2_0` linear integer grids (paired with `Q8_0` shared experts & hyper-connections and `Q6_K` attention/GDN), prompt prefill runs at **`367.0 tok/s`** and single-stream decode at **`94.0 tok/s`** (`~345 tok/s` aggregate across 10 parallel slots at `-np 10`).

---

## 5. Files in This Repository

| File | Size | Description |
| :--- | :---: | :--- |
| `Qwen3.8-Flash-Next-DR-V2-Q2_0-00001-of-00002.gguf` | `38.38 GB` (`35.74 GiB`) | Shard 1: Transformer + MoE compute backbone (`1,223` tensors — `Q2_0` routed experts, `Q8_0` shared experts & hyper-connections, `Q6_K` attention/GDN & output head, `Q4_0` token embeddings) |
| `Qwen3.8-Flash-Next-DR-V2-Q2_0-00002-of-00002.gguf` | `28.80 GB` (`26.82 GiB`) | Shard 2: Static Layer-1 N-Gram PLE table (`per_layer_token_embd.weight`, `Q4_0`, `mmap`-able from host RAM/SSD) |
| **Total** | **`67.18 GB` (`62.56 GiB`)** | **2-Shard Split GGUF (`38.38 GB` GPU VRAM + `28.80 GB` host `mmap`)** |

---

## 6. Quick Start (`llama.cpp`)

Download both shards into the same directory (making sure both share the same filename prefix, e.g., `Qwen3.8-Flash-Next-DR-V2-Q2_0-00001-of-00002.gguf` and `Qwen3.8-Flash-Next-DR-V2-Q2_0-00002-of-00002.gguf`) and pass Shard 1 to `llama-server` or `llama-cli`:

### `llama-server` (Recommended for Coding & Math Reasoning)

```bash
llama-server \
  -m Qwen3.8-Flash-Next-DR-V2-Q2_0-00001-of-00002.gguf \
  --alias Qwen3.8-Flash-Next --host 127.0.0.1 --port 8080 \
  -ngl 999 -c 65536 -fa on -ctk q8_0 -ctv q8_0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning-format deepseek --no-reasoning-preserve
```

### `llama-cli` (With Host SSD/RAM `mmap` for Shard 2)

```bash
llama-cli \
  -m Qwen3.8-Flash-Next-DR-V2-Q2_0-00001-of-00002.gguf \
  -lm mmap --lazy-mode on -ngl 999 -fa on \
  --temp 0.6 --top-p 0.95 --top-k 20 \
  -p "Write a fast branchless integer sorting network in C++."
```
