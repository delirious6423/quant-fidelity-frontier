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

# Qwen3.8-27B-Randean-GGUF (`v3.0`)

Ready-to-run **GGUF** releases of **[`flechin/Qwen3.8-27B-Randean-MXFP4` (`v3.0`)](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4)** for **`llama.cpp`**, **Ollama**, and **LM Studio** — including full vision support via **`mmproj-BF16.gguf`** and importance-matrix (`imatrix`) calibration for `Q4_K_M` and `Q6_K`.

- **Canonical MXFP4 Model (vLLM / ROCm / Blackwell)**: [`flechin/Qwen3.8-27B-Randean-MXFP4`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4)
- **Interactive ZeroGPU Demo Space**: [`https://huggingface.co/spaces/flechin/randean`](https://huggingface.co/spaces/flechin/randean)

---

## Which File Should I Download?

| File | Size | Calibration | Recommended Hardware | Verified Single-Stream Speed *(R9700)* |
| :--- | ---: | :---: | :--- | :---: |
| **`Qwen3.8-27B-Randean-v3.0-Q4_K_M.gguf`** *(recommended default)* | **16.5 GB** | `imatrix` | Single **24 GB** GPU / Apple Silicon (32 GB+ RAM) | **31.6 tok/s** *(1 GPU)* |
| **`Qwen3.8-27B-Randean-v3.0-Q6_K.gguf`** *(best balance)* | **22.1 GB** | `imatrix` | Single **32 GB** GPU / Apple Silicon (36 GB+ RAM) | **25.2 tok/s** *(1 GPU)* |
| **`Qwen3.8-27B-Randean-v3.0-Q8_0.gguf`** *(near-lossless)* | **28.6 GB** | — | **48 GB+** / Dual 24–32 GB GPUs / Apple Silicon (48 GB+ RAM) | **19.5 tok/s** *(2 GPUs)* |
| **`mmproj-BF16.gguf`** *(optional, for vision)* | **931 MB** | Unquantized BF16 | Pair with any quant above for image input | — |
| **`imatrix.gguf`** *(for quantizers only)* | **27.3 MB** | 129 chunks `calibration_datav3` | Not loaded at inference — included for cutting extra quants (`IQ4_XS`, `Q5_K_M`, etc.) | — |

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

## Verified GGUF Quality Check

Every published GGUF file was smoke-tested live on `llama-server` (math, structured tool calling, and vision via `mmproj-BF16.gguf` — **3/3 PASS** on all three quants) and benchmarked on our **Precision Tools Hard (`T = 0.2`)** and **H5 Active Grounding** canaries:

| Model / Format | Precision Tools Hard (`T = 0.2`) ↑ | H5 Active Grounding ↑ | Notes |
| :--- | :---: | :---: | :--- |
| **`Randean v3.0` (`Q4_K_M`, imatrix GGUF)** | **`100.0%`** (`48/48`) | `75.6%` | `imatrix` lifted `Q4_K_M` by `+20.8%` over uncalibrated `Q4_K_M` (`79.2%`) |
| **`Randean v3.0` (`Q6_K`, imatrix GGUF)** | **`89.6%`** (`43/48`) | **`77.8%`** | Best single-GPU 32 GB GGUF balance (`100%` ambiguity, unit/schema, and state recovery) |
| **`Randean v3.0` (`Q8_0`, GGUF)** | `85.4%` (`41/48`) | `75.6%` | `100%` (`16/16`) physical/math impossibility restraint |
| **`Randean v3.0` (`MXFP4`, vLLM canonical)** | `91.7%` (`44/48`) | **`82.2%`** | Full-suite reference (`56.2%` Terminal-Bench 2.1, `72.0%` LiveCodeBench v6, `#1 / 171` BullshitBench v2) |
| *Stock `Qwen3.8-27B` (`MXFP4`, ref)* | `64.6%` (`31/48`) | `60.0%` | Unaligned baseline |

*See [`flechin/Qwen3.8-27B-Randean-MXFP4`](https://huggingface.co/flechin/Qwen3.8-27B-Randean-MXFP4) for the full 1,374-item frontier benchmark breakdown (`v3.0` vs. `v2.0`, Stock `Qwen3.8-27B`, `Pi`, `Swift-1.5`, `Inkling-2-Mini`, and `Qwen3.8-Flash-Next`).*

---

## Conversion Notes

- **Bit-exact dequantization**: Exported directly from `flechin/Qwen3.8-27B-Randean-MXFP4` (`v3.0-arm-b`) via streaming OCP MXFP4 dequantization into `llama.cpp`'s native `qwen35` converter (`--no-nextn`, verified `0.000000` max diff across weights and `+1`-shifted RMSNorms vs. source tensors).
- **Importance matrix**: `imatrix.gguf` was computed with `llama-imatrix --process-output` across all `129` chunks (`c = 512`) of the standard public `calibration_datav3.txt` corpus.
- **Vision tower**: `mmproj-BF16.gguf` is exported directly from the unquantized BF16 vision encoder (`334` tensors).

---

## License

Released under the **Apache 2.0** license, inheriting the base license of `Qwen/Qwen3.8-27B` and `amd/Qwen3.8-27B-Quark-AWQ-MXFP4`.
