---
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - quantized
  - llama.cpp
  - qwen3.8
  - math
  - reasoning
---

# Qwen3.8-27B-DR-4.0bpw-v4-GGUF

Stock GGUF 4-bit mixed-precision quantization (`v4`) of `Qwen/Qwen3.8-27B` (`17.38 GB`, `4.00 bpw` layer body / `5.00 bpw` total file including `Q6_K` embeddings, `Q6_K` untied LM head, and `F32` norms & Gated DeltaNet recurrent gates).

Runs natively out-of-the-box on standard mainline `llama.cpp` (`llama-cli`, `llama-server`) across CUDA, Metal, and Vulkan with **zero custom kernels or non-linear lookup tables (`IQ`)**.

![Executive 4-Panel Benchmark Summary](chart4_executive_4panel_hero.png)

---

## Key Highlights

- **`29 / 30 (96.67%)` `Pass@1` on `AIME 2025` (`MathArena/aime_2025`):**
  - **AIME 2025 Paper I (`#01–#15`):** `14 / 15 (93.33%)`
  - **AIME 2025 Paper II (`#16–#30`):** **`15 / 15 (100.00%)`** (perfect sweep)
  - **`30 / 30` (`100%`) natural completions (`finish=stop`)** at `64k` thinking budget (`--reasoning-budget 65536`), averaging `25,636` completion tokens per problem across `769,079` total generated tokens.
- **Beats Uncompressed `BF16` Perplexity Across All 5 Evaluation Suites:**
  - **`WT2_2048` (`32,752` tok):** **`6.6935`** vs. `6.7357` (`ΔPPL = -0.0422`)
  - **`WT2_Full` (`296,815` tok):** **`6.8929`** vs. `6.8970` (`ΔPPL = -0.0041`)
  - **`WT2_1024` (`4,092` tok):** **`7.2796`** vs. `7.3284` (`ΔPPL = -0.0488`, **`KLD = 0.016827`**)
  - **`Ext` (Reasoning / Code / Math, `342` tok):** **`14.8780`** vs. `15.0114` (`ΔPPL = -0.1334`, **`96.20%` Top-1 / `100.00%` Top-5**)
  - **`SG` (`<think>` Traces, `2,040` tok):** **`5.5135`** vs. `5.5672` (`ΔPPL = -0.0537`, **`KLD = 0.004624`**, **`97.65%` Top-1 / `100.00%` Top-5**)

---

## Model Summary

| Property | Value |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen3.8-27B` |
| **Format** | Stock `GGUF` (`Q4_K` / `Q5_K` / `Q6_K` / `F32`) |
| **File Name** | `Qwen3.8-27B-DR-4.0bpw-v4.gguf` |
| **File Size** | `17.38 GB` (`16.18 GiB` / `17,377,598,592` bytes) |
| **Layer Body Bitrate** | `4.00 bpw` (`64` hybrid Gated DeltaNet + Full GQA layers) |
| **Total File Effective Bitrate** | `5.0013 bpw` (`(17,377,598,592 * 8) / 27,796,869,120`) |
| **SHA256** | `15844db2aa724f3612a81863d362c9120cb51fe6db86421e9c75b90d87a6ca3f` |
| **Native Context Length** | `262,144` |

---

## Perplexity, Logit KL Divergence & Token Agreement vs. Uncompressed `BF16`

| Evaluation Suite | Tokens | `BF16` PPL | `DR-4.0bpw-v4` PPL | `ΔPPL` vs. `BF16` | `BF16` NLL | `DR-4.0bpw-v4` NLL | Mean KLD | Top-1 Agreement | Top-5 Agreement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`WT2_2048`** (`16 x 2048`) | `32,752` | `6.7357` | **`6.6935`** | **`-0.0422`** | `1.907415` | **`1.901144`** | **`0.023006`** | **`95.54%`** | **`99.86%`** |
| **`WT2_Full`** (`145 x 2048`) | `296,815` | `6.8970` | **`6.8929`** | **`-0.0041`** | `1.931083` | **`1.930492`** | — | — | — |
| **`WT2_1024`** (`4 x 1024`) | `4,092` | `7.3284` | **`7.2796`** | **`-0.0488`** | `1.991762` | **`1.985078`** | **`0.016827`** | **`95.77%`** | **`99.98%`** |
| **`Ext`** (Reasoning / Code) | `342` | `15.0114` | **`14.8780`** | **`-0.1334`** | `2.708812` | **`2.699882`** | **`0.013113`** | **`96.20%`** | **`100.00%`** |
| **`SG`** (`<think>` Traces) | `2,040` | `5.5672` | **`5.5135`** | **`-0.0537`** | `1.716892` | **`1.707202`** | **`0.004624`** | **`97.65%`** | **`100.00%`** |

![Perplexity Change vs BF16](chart2_delta_ppl_diverging.png)

---

## Downstream Olympiad Reasoning (`AIME 2025`)

Evaluated on all 30 problems of `MathArena/aime_2025` (AIME 2025 I & II) using stock `llama-server` with `64k` thinking budget (`--reasoning-budget 65536`, `max_tokens = 67584`), `-ctk q8_0 -ctv q8_0 -fa on`, and `--temp 0.60 --top-p 0.95 --top-k 20 --min-p 0.03`:

| Benchmark Split | Problems | `Pass@1` Score | Avg Completion Tokens | Natural Stop Rate (`finish=stop`) |
| :--- | :---: | :---: | :---: | :---: |
| **`AIME 2025 Paper I` (`#01–#15`)** | `15` | **`14 / 15 (93.33%)`** | `27,457.9` | `15 / 15 (100%)` |
| **`AIME 2025 Paper II` (`#16–#30`)** | `15` | **`15 / 15 (100.00%)`** | `23,814.1` | `15 / 15 (100%)` |
| **`AIME 2025 Combined` (`#01–#30`)** | `30` | **`29 / 30 (96.67%)`** | `25,636.0` | **`30 / 30 (100%)`** |

Full per-problem logs and token counts are included in [`aime25_v4_64k_np10_results.json`](aime25_v4_64k_np10_results.json).

---

## Recommended `llama.cpp` Launch Flags (Optimal Reasoning Settings)

```bash
./llama-server \
  -m Qwen3.8-27B-DR-4.0bpw-v4.gguf \
  -ngl 999 -fa on -ctk q8_0 -ctv q8_0 \
  -c 68000 \
  --temp 0.60 --top-p 0.95 --top-k 20 --min-p 0.03 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --reasoning-format deepseek --reasoning-budget 65536
```
