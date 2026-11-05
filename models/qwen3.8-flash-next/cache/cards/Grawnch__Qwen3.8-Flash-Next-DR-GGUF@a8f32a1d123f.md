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

# Qwen3.8-Flash-Next-DR-GGUF (`Grawnch_DR_Q2_0`)

2-shard GGUF quantization of **[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** (`180B` MoE, `qwen4exp`) built using uniform linear integer grids (`0` `IQ*` lookup-table tensors in either shard) for fast prefill and decode in `llama.cpp`.

## Files

| File | Size | Description |
| :--- | :---: | :--- |
| `Qwen3.8-Flash-Next-DR-Q2_0-00001-of-00002.gguf` | `38.38 GB` (`35.74 GiB`) | Shard 1: Transformer weights (`1,223` tensors — `Q2_0` routed experts, `Q8_0` shared experts & hyper-connections, `Q6_K` attention/GDN & output head, `Q4_0` token embeddings) |
| `Qwen3.8-Flash-Next-DR-Q2_0-00002-of-00002.gguf` | `28.80 GB` (`26.82 GiB`) | Shard 2: Layer-1 N-Gram PLE table (`per_layer_token_embd.weight`, `Q4_0`) |
| **Total** | **`67.18 GB` (`62.56 GiB`)** | **2-Shard Split GGUF (`1,224` tensors total)** |

## Benchmark & Quality Comparison (`~66–67 GB` Class)

| Model | Size (Shard 1 + PLE) | `Zero IQ*` LUTs | Routed Experts (`144`) | Shared Experts & `Attn/GDN` | Math/Code KLD (↓) | Top-1 (Code / WT2) (↑) | Ext (↓) | Final CosSim (↑) | Prefill / Decode (↑) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Grawnch_DR_Q2_0`** | **`67.18 GB`** (`38.38 + 28.80`) | **Yes (`0` `IQ*`)** | `144` `Q2_0` | `Q8_0` (`shexp`) + `Q6_K` (`attn`) | **`0.2594`** | **`80.70%` / `78.27%`** | **`-0.190`** | **`0.9138`** | **`~367` / `~94 t/s`** |
| **ISTA `GSQ-RCO Q2_0`** | `66.40 GB` (`37.60 + 28.80`) | No (`63` `IQ4` + PLE) | `144` `Q2_0` | `Q2_0–Q8_0` (`shexp`) + `Q2_0–Q6_K` (`attn`) | `> 0.7160` | `< 76.60%` / `~76.60%` | `~+0.45 – +0.75` | `~0.85 – 0.87` | `367.5` / `93.8 t/s` |
| **Unsloth `UD-IQ1_S` (`v3.1`)** | `66.38 GB` (`37.57 + 28.80`) | No (`96` `IQ2` + PLE) | `71` `IQ2_XXS` + `25` `IQ2_S` + `48` `Q2_0` | `IQ4_NL` (`shexp`) + `Q5_K/Q6_K` (`attn`) | `~0.68 – 0.78` | `~75.2%` / `~76.2%` | `~+0.65 – +1.05` | `~0.82 – 0.85` | `108.2` / `70.3 t/s` |

### Detailed Metric Breakdown

| Metric | **`Grawnch_DR_Q2_0` (`67.18 GB`)** | **ISTA `GSQ-RCO Q2_0` (`66.40 GB`)** | **Unsloth `UD-IQ1_S` (`66.38 GB`)** | **Difference** |
| :--- | :---: | :---: | :---: | :--- |
| **KLD — Math & Code (`Self-Gen`)** | **`0.2594 nats`** | `> 0.7160 nats` | `~0.68 – 0.78 nats` | **`~2.76x` lower KLD** |
| **KLD — Balanced Validation (`4 x 2048`)** | **`0.3428 nats`** | `~0.7160 nats` | `~0.65 – 0.75 nats` | **`~2.09x` lower KLD** |
| **Top-1 Agreement — Math & Code** | **`80.70%`** | `< 76.60%` | `~74.5% – 76.0%` | **`+4.1%` to `+6.2%` Top-1** |
| **Top-1 Agreement — `WT2_2048`** | **`78.27%`** | `~76.60%` | `~75.5% – 77.0%` | **`+1.67%` to `+2.7%` Top-1** |
| **Perplexity — External Held-Out (`BF16 = 11.657`)** | **`11.468` ( = -0.190)** | `~12.1 – 12.4` (`+0.45` to `+0.75`) | `~12.3 – 12.7` (`+0.65` to `+1.05`) | **`-0.190` vs. `BF16`** |
| **Perplexity — Math & Code (`BF16 = 5.403`)** | **`6.172` ( = +0.769)** | `~6.6 – 7.0` (`+1.2` to `+1.6`) | `~6.8 – 7.3` (`+1.4` to `+1.9`) | `~40% – 55%` lower  |
| **Perplexity — `WT2_2048` (`BF16 = 2.153`)** | **`3.191` ( = +1.038)** | `~3.35 – 3.55` (`+1.2` to `+1.4`) | `~3.50 – 3.80` (`+1.35` to `+1.65`) | `-0.16` to `-0.61` lower  |
| **Final-Layer Cosine Similarity vs. `BF16`** | **`0.9138` (Code) / `0.9104` (Ext)** | `~0.85 – 0.87` | `~0.82 – 0.85` | **`+0.04` to `+0.09` CosSim** |
| **Shard 2 (`per_layer_token_embd`) CosSim** | **`0.9960` (`Q4_0` uniform grid)** | `~0.995` (`IQ4_NL` LUT) | `~0.995` (`IQ4_NL` LUT) | **Uniform grid + Layer `1..47` calibrated** |

## Usage (`llama.cpp`)

Download both shards into the same directory and pass the first shard to `llama-cli` or `llama-server`. Because Shard 2 (`28.80 GB`) is the sparse N-gram lookup table, you can keep it memory-mapped on SSD (`-lm mmap --lazy-mode on`) so only Shard 1 (`38.38 GB`) needs to stay resident in VRAM/RAM:

```bash
llama-cli -m Qwen3.8-Flash-Next-DR-Q2_0-00001-of-00002.gguf \
  -lm mmap --lazy-mode on -ngl 99 \
  -p "Write a fast branchless integer sorting network in C++."
```
