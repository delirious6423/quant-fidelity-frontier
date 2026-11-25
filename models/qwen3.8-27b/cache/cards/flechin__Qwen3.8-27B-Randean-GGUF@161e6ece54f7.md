---
base_model:
  - flechin/Qwen3.8-27B-Randean-MXFP4
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
thumbnail: https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/assets/randean_logo.jpg
tags:
  - gguf
  - qwen3_5
  - imatrix
  - llama-cpp
  - ollama
  - lm-studio
  - reasoning
  - epistemology
  - reality-aligned
  - de-hedged
  - anti-sycophancy
  - causal-grounding
  - agentic
---

<p align="center">
  <img src="https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4/resolve/main/assets/randean_logo.jpg" alt="Randean Logo" width="640" style="border-radius: 12px;" />
</p>

# Qwen3.8-27B-Randean-GGUF (`v3.1b`)

Ready-to-run **GGUF** releases of **[`flechin/Qwen3.8-27B-Randean-MXFP4` (`v3.1b`)](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4)** for **`llama.cpp`**, **Ollama**, and **LM Studio** — including full vision support via **`mmproj-BF16.gguf`** and fresh `v3.1b` importance-matrix (`imatrix`) calibration for `Q4_K_M` and `Q6_K`.

- **Canonical MXFP4 Model (vLLM / ROCm / Blackwell)**: [`flechin/Qwen3.8-27B-Randean-MXFP4`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4) (`58.4%` Terminal-Bench 2.1, `73.7%` LiveCodeBench v6, `85.0%` AIME 2024–2025)
- **Interactive ZeroGPU Demo Space**: [`https://huggingface.co/spaces/flechin/randean`](https://huggingface.co/spaces/flechin/randean)

---

## Which File Should I Download?

| File | Size | Calibration | Recommended Hardware | Verified Single-Stream Speed *(R9700)* |
| :--- | ---: | :---: | :--- | :---: |
| **`Qwen3.8-27B-Randean-v3.1b-Q4_K_M.gguf`** *(recommended default)* | **16.5 GB** | `imatrix` | Single **24 GB** GPU / Apple Silicon (32 GB+ RAM) | **31.8 tok/s** *(1 GPU)* |
| **`Qwen3.8-27B-Randean-v3.1b-Q6_K.gguf`** *(best balance)* | **22.1 GB** | `imatrix` | Single **32 GB** GPU / Apple Silicon (36 GB+ RAM) | **25.3 tok/s** *(1 GPU)* |
| **`Qwen3.8-27B-Randean-v3.1b-Q8_0.gguf`** *(near-lossless)* | **28.6 GB** | — | **48 GB+** / Dual 24–32 GB GPUs / Apple Silicon (48 GB+ RAM) | **19.6 tok/s** *(2 GPUs)* |
| **`mmproj-BF16.gguf`** *(optional, for vision)* | **931 MB** | Unquantized BF16 | Pair with any quant above for image input | — |
| **`imatrix.gguf`** *(for quantizers only)* | **27.3 MB** | 129 chunks `calibration_datav3` | Not loaded at inference — included for cutting extra quants (`IQ4_XS`, `Q5_K_M`, etc.) | — |

*(Prior `v3.0` GGUF files remain preserved on Hugging Face under git tag [`v3.0-arm-b`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-GGUF/tree/v3.0-arm-b).)*

---

## Quick Start

### `llama.cpp` (`llama-server`, Text + Tool Calling + Vision)

```bash
llama-server \
  -hf flechin/Qwen3.8-27B-Randean-GGUF:Q4_K_M \
  --mmproj mmproj-BF16.gguf \
  --host 0.0.0.0 --port 8080 \
  -ngl 99 -fa on -c 32768 \
  --jinja --reasoning-format deepseek
```

*(For text-only usage, omit `--mmproj`. Swap `:Q4_K_M` for `:Q6_K` or `:Q8_0` as desired.)*

### Ollama

```bash
ollama run hf.co/flechin/Qwen3.8-27B-Randean-GGUF:Q4_K_M
```

*(Also supports `:Q6_K` and `:Q8_0`.)*

### LM Studio

Search `flechin/Qwen3.8-27B-Randean-GGUF` inside LM Studio and select **`Q4_K_M`** (24 GB GPUs) or **`Q6_K`** (32 GB GPUs). Recommended decoding settings for tool use and coding: `temperature = 0.2`, `top_p = 0.95`.

---

## Verified `v3.1b` GGUF Quality Check

Every published `v3.1b` GGUF file was smoke-tested live on `llama-server` (math, structured `web_search` tool calling, and real image vision via `mmproj-BF16.gguf` — **3/3 PASS** on all three quants) and benchmarked on our **H5 4-Way Active Grounding** (including unhinted casual long-tail search `Arm 4` and trivial no-spam control `Arm 4b`) and **Precision Tools Hard (`T = 0.2`)** canaries:

| Model / Format | H5 4-Way Overall ↑ | H5 Arm 4 (Casual Search) ↑ | H5 Arm 4b (No-Spam) ↑ | Precision Tools Hard (`T = 0.2`) ↑ | Notes |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Randean v3.1b` (`Q4_K_M`, imatrix GGUF)** | **`83.3%`** | `86.7%` | **`100.0%`** | `75.0%` (`36/48`) | `93.3%` Arm 1 tool probe, **`100.0%` Arm 3 offline honesty**, `3/3` smoke PASS |
| **`Randean v3.1b` (`Q6_K`, imatrix GGUF)** | **`85.0%`** | **`100.0%`** (`15/15`) | **`100.0%`** | `70.8%` (`34/48`) | `93.3%` Arm 1 tool probe, `15/15` proactive search, `3/3` smoke PASS |
| **`Randean v3.1b` (`Q8_0`, GGUF)** | **`86.7%`** | `93.3%` | `90.0%` | `75.0%` (`36/48`) | `93.3%` Arm 1 & Arm 3, **`100.0%` trust/injection defense**, `3/3` smoke PASS |
| **`Randean v3.1b` (`MXFP4`, vLLM canonical)** | `76.7%` | `80.0%` | `90.0%` | **`87.5%`** (`42/48`) | Full-suite reference (`58.4%` TB 2.1, `73.7%` LCB v6, `85.0%` AIME-60, `4.0%` False-Nudge) |
| *Stock `Qwen3.8-27B` (`MXFP4`, ref)* | — | — | — | `64.6%` (`31/48`) | Unaligned baseline |

*See [`flechin/Qwen3.8-27B-Randean-MXFP4`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4) for the complete 1,587-item production benchmark suite (`v3.1b` vs. `v3.0`, `v2.0`, Stock `Qwen3.8-27B`, `Pi`, `Swift-1.5`, `Inkling-2-Mini`, and `Qwen3.8-Flash-Next`).*

---

## Conversion Notes

- **Bit-exact dequantization**: Exported directly from `flechin/Qwen3.8-27B-Randean-MXFP4` (`v3.1b`) via streaming OCP MXFP4 dequantization into `llama.cpp`'s native `qwen35` converter (`--no-nextn`).
- **Fresh `v3.1b` importance matrix**: `imatrix.gguf` was recomputed directly from `Qwen3.8-27B-Randean-v3.1b-Q8_0.gguf` with `llama-imatrix --process-output` across all `129` chunks (`c = 512`) of `calibration_datav3.txt`.
- **Vision tower**: `mmproj-BF16.gguf` is exported directly from the unquantized BF16 vision encoder (`334` tensors).

---

## License

Released under the **Apache 2.0** license, inheriting the base license of `Qwen/Qwen3.8-27B` and `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`.
