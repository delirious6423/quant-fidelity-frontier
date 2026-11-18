---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - qwen3.8
  - qwen35
  - kenzan
  - pin-grid
  - apple-silicon
  - metal
  - 16gb
  - iq2_xxs
  - iq2_xs
  - imatrix
language:
  - en
  - ru
  - multilingual
pipeline_tag: text-generation
---

> **🟢 Stock-Compatible:** Works out-of-the-box in **LM Studio / Ollama / stock `llama.cpp`** — Standard `IQ2_XXS` / `Q4_K` / `Q5_K` / `Q8_0` GGUF tensors — runs on Apple Silicon 16 GB and Windows/Linux 12–16 GB VRAM GPUs (RTX 4070Ti / 4080 / 3090).

<p align="center">
  <img src="assets/cover.png" alt="Kenzan-Qwen3.8-27B-PinGrid" width="512" />
</p>

☕ **Support Local AI Development**: Help me acquire a Mac mini 24GB/32GB to optimize and test local MoE models up to 80B parameters. [Donate via Ko-fi →](https://ko-fi.com/kabcpc)





> [!TIP]
> **Kenzan-Qwen3.8-27B-PinGrid — "27 Billion Dense Parameters, PPL 8.50, Zero 1-Bit Compromise on 16 GB."**
> *Sixty-four hybrid DeltaNet/Attention layers, `imatrix` surgical pin-grid placement across the exploding tail (`blk.36..63`), and one sneaky `nextn_predict_layers=1` header bug squashed — clean `Qwen/Qwen3.8-27B` running natively on a 16 GB Mac mini M6.*

## Overview

`Kenzan-Qwen3.8-27B-PinGrid-IQ2_XXS-M6-16G.gguf` (**10.77 GiB**, `3.39 BPW`) is an `imatrix`-guided, Pin-Grid quantized GGUF release built strictly from clean, non-abliterated **`Qwen/Qwen3.8-27B`** weights and tuned for **Apple Silicon 16 GB Unified Memory** (Mac mini M6).

Unlike ternary `PTQ1_0` repacks that sacrifice tail precision for sub-7 GiB footprints, this release preserves the full high-precision `Always-On` backbone and upgrades the sensitive mid-late layers (`blk.36..57`) to `IQ2_XS` and tail anchors (`blk.58..63`) to `Q4_0/Q5_K/Q6_K`, achieving **WikiText-2 PPL = `8.4997 ± 0.19278`** — outperforming the previous 6.42 GiB `Core16-Q4_1` build (`9.6249`) by **-1.1252 PPL**.

### Key Highlights
- **Pure `Qwen/Qwen3.8-27B` Checkpoint:** Quantized directly from native `Q8_0` weights with `unsloth/Qwen3.8-27B-GGUF` importance matrix (`496` tensors calibrated over `1251` chunks).
- **No Custom Ternary Kernel Required:** Uses standard `IQ2_XXS` / `IQ2_XS` / `IQ3_XXS` / `Q4_0` / `Q5_K` / `Q6_K` / `Q8_0` formats — runs cleanly on upstream `llama.cpp` (`qwen35` architecture) as well as Apple Metal builds.
- **GGUF Header Fix Included (`nextn_predict_layers = 0`):** Pruning the MTP layer (`blk.64`) with `--prune-layers 64` leaves `qwen35.nextn_predict_layers = 1` in raw `llama-quantize` output, which causes `llama.cpp` to misclassify active layer `blk.63` as NextN and skip its FFN (degrading PPL to `24.86`). This GGUF has `qwen35.nextn_predict_layers = 0` patched directly in the header, so it works out of the box with no `--override-kv` flags.

---

## Pin-Grid Architecture (`recipes/tensor_types_c3.txt`)

`imatrix` L2 sensitivity (`in_sum2`) in `Qwen3.8-27B` spans **5 orders of magnitude** from early blocks (`~1e4`) through mid-network peaks (`1e6–6.6e7`) to the tail (`blk.54..63`, up to `3.14e9` at `blk.63.ffn_down`). The Pin-Grid places precision strictly where variance explodes:

| Tensor Group | Precision | Purpose |
| :--- | :---: | :--- |
| `output.weight` | `Q8_0` | Clean logit head without quantization collapse |
| `token_embd.weight` | `Q6_K` | Preserves 248,320-token multilingual/Russian vocabulary |
| `attn_k.weight`, `attn_v.weight` | `Q6_K` | High-precision attention keys/values across all layers |
| `blk.(58,61,63).ffn_down` & `blk.(54..63).(attn_output,ssm_out)` | `Q5_K` | Tail stability anchors (`in_sum2` up to `3.14e9`) |
| `blk.(0,58,61,63).ffn_(up,gate)` & `blk.(2,18,35,42,46,50,54).ffn_down` | `Q4_0` | Entry/peak-layer FFN anchors |
| `attn_q`, `attn_output`, `attn_qkv`, `ssm_out` (non-tail) | `Q4_0` | `Always-On` attention & Gated DeltaNet backbone |
| `blk.(59,60,62).ffn_*` & `attn_gate` | `IQ3_XXS` | Tail FFN & gating precision |
| `blk.(36..57).ffn_(up,gate,down)` | `IQ2_XS` | Mid-late FFN precision boost (`+165 MiB` over pure `IQ2_XXS`) |
| Remaining `blk.(1..35).ffn_*` | `IQ2_XXS` | Compact base crown |

---

## Verified Benchmarks (Apple Mac mini M6, 16 GB Unified Memory, Metal + `q4_0` KV)

Measured on `wt2_128k.txt` (`-c 2048 --chunks 14 -ctk q4_0 -ctv q4_0 -fa 1 -ngl 99`) and `llama-bench` (`-p 512 -n 128 -r 3`):

| Model | Size | PPL (`wt2`, `c=2048`) | `pp512` (t/s) | `tg128` (t/s) | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `Kenzan-Qwen3.8-27B-Abliterated-Core16-Q4_1` (baseline) | 6.42 GiB | `9.6249 ± 0.22173` | 225.61 | **16.42** | 1-bit `PTQ1_0` base |
| **`Kenzan-Qwen3.8-27B-PinGrid-IQ2_XXS-M6-16G` (this release)** | **10.77 GiB** | **`8.4997 ± 0.19278`** | **248.52** | **11.31** | **-1.1252 PPL vs baseline** |

- **Live `llama-server` test (`ngram-map-k` `n=3, m=4`, Russian + C++17):** `11.25 t/s` generation (`118.40 t/s` prompt eval, `55/198` draft tokens accepted), `12.01 GiB` RSS, zero model swapping.

---

## Quick Start & Recommended Deployment (Apple Silicon 16 GB)

This model runs on standard `llama.cpp` (`qwen35` support) with Metal Flash Attention and 4-bit KV cache:

```bash
llama-server   -m Kenzan-Qwen3.8-27B-PinGrid-IQ2_XXS-M6-16G.gguf   --host 127.0.0.1 --port 11435   -ngl 99 -fa on   -c 32768 --cache-type-k q4_0 --cache-type-v q4_0   -np 1 -kvu -t 4 --threads-draft 2   -b 512 -ub 512 --jinja   --spec-type ngram-map-k   --spec-ngram-map-k-size-n 3 --spec-ngram-map-k-size-m 4   --spec-draft-n-max 3 --spec-draft-n-min 1   --temp 0.20 --top-p 0.95 --top-k 40 --min-p 0.08   --repeat-penalty 1.05 --repeat-last-n 256   --presence-penalty 0.00 --frequency-penalty 0.00   --alias kenzan-qwen3.8:27b-pingrid
```

> **16 GB RAM Tip:** On 16 GB Apple Silicon machines, keep `-c 8192` to `-c 32768` with `--cache-type-k q4_0 --cache-type-v q4_0` and close heavy background apps to stay comfortably under the ~15 GB Metal working-set limit.

---

## Acknowledgements & Inspiration

* **Inspired by Bonsai-2 & Prism ML:** Deep gratitude and inspiration go to the creators of **Bonsai-2** at **[Prism ML (huggingface.co/prism-ml)](https://huggingface.co/prism-ml)** for pioneering extreme low-bit quantization, blockwise Hadamard phase stabilization, and proving that frontier-class models can thrive on consumer unified memory.
* **Special Thanks to Misha:** Sincere gratitude to **Misha** for unwavering support and financial assistance, which made the hardware testing, calibration runs, and open-weights releases in the Kenzan series possible.
* **Base Model & Calibration:** Built from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) using the `imatrix` from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) and the open-source `llama.cpp` Metal ecosystem.

---

## Support the Project & Funding Goals

All **Kenzan** weights, calibration recipes, and tooling are released openly and free of charge for the local AI community. If this model saved you cloud API costs or helped you run 27B–40B reasoning locally on a base Mac, please consider supporting future releases:

☕ **[Donate via Ko-fi →](https://ko-fi.com/kabcpc)**

### What Your Donations Directly Fund

1. **Upgrading to a Mac mini 24 GB / 32 GB (Apple Silicon):** Breaking past the 16 GB ceiling to quantize, calibrate, and benchmark **70B–80B** hybrid models with wider `Q4_0` pin grids.
2. **Cloud GPU Compute for I-Matrix & Calibration Sweeps:** Covering cloud VRAM hours needed for full-precision importance-matrix (`imatrix`) generation, layer sensitivity sweeps, and multi-corpus perplexity verification.
3. **Open-Source Metal Kernels & Tooling:** Funding continued development of `llama.cpp` Metal optimizations and `kenzan_router.py` features for the local-first community.
