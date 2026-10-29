---
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
license: other
license_name: lfm1.0
license_link: LICENSE
library_name: gguf
tags:
- text-generation
- gsq
- rco
- llama.cpp
- quantized
- reasoning
- uncensored
- liquid
- lfm2.5
- moe
- mixed-precision
- speculative-decoding
- dspark
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
---

# LFM2.5-24B-A2B Uncensored GSQ-RCO GGUF

<div align="center">

[![Model Requests](https://img.shields.io/badge/Model_Requests-Submit-f59e0b?logo=huggingface&logoColor=white)](https://abenzerps.com/request-a-model/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?logo=github&logoColor=white)](https://github.com/abenzerps)

</div>

<p align="center">
  <img src="./assets/benchmark.png" alt="LFM2.5-24B-A2B GSQ-RCO Benchmark Comparison" width="1010">
</p>

Mixed-precision **GSQ-RCO** GGUF quantizations of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), engineered for lossless routing, multi-step reasoning, and high-speed `DSpark` speculative decoding in `llama.cpp`.

- **Architecture:** Hybrid `30 ShortConv + 10 GQA + 64-Expert MoE` (`24B` total · `2.0B` active parameters)
- **Context & Speed:** **`131,072 (128K)` tokens** (`4×` smaller KV cache · **125–130 tok/s decode** & **~1,500 tok/s prefill** on RTX 3060-class 12 GB GPUs without `DSpark`)
- **Recommended Sampling:** `temperature = 0.1` · `top_k = 50` · `repeat_penalty = 1.05`

<details>
<summary><b>12 GB VRAM Footprint &amp; 64K / 128K Launch Configuration (<code>IQ2_XS</code>)</b></summary>
<br>

Measured on a 12 GB GPU (`RTX 3060`-class) with `GSQ-RCO-IQ2_XS` (`8.52 GB`) and `Q8_0` KV cache (`125–130 tok/s` decode, `~1,500 tok/s` prefill without `DSpark`):

| Context Window | VRAM Usage | Free Headroom (12 GB GPU) |
| :--- | :---: | :---: |
| **64K Context** (`-c 65536`) | `~9.88 GB` | `2.38 GB free` |
| **128K Context** (`-c 131072`) | `~10.82 GB` | `1.45 GB free` |

```bash
llama-server \
  -m LFM2.5-24B-A2B-GSQ-RCO-IQ2_XS.gguf \
  -ngl 99 \
  -c 131072 \
  -b 4096 \
  -ub 2048 \
  -ctk q8_0 \
  -ctv q8_0 \
  -t 6 \
  -np 1 \
  --flash-attn on
```
<sub>For 64K context (`~9.88 GB` VRAM), replace `-c 131072` with `-c 65536`.</sub>

</details>

## GGUF Files

| File | bpw | Size | Recovery | Target Hardware & Notes |
| :--- | :---: | :---: | :---: | :--- |
| [LFM2.5-24B-A2B-GSQ-RCO-IQ2_XXS.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ2_XXS.gguf) | 2.65 | 7.93 GB | 89.6% | 8 GB VRAM · Ultra-compact (`6.04×` compression) |
| [LFM2.5-24B-A2B-GSQ-RCO-IQ2_XS.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ2_XS.gguf) | 2.85 | 8.52 GB | 94.4% | 10–12 GB VRAM · Best sub-9 GB & coding accuracy |
| **[LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf)** *(Recommended)* | **3.50** | **10.54 GB** | **98.9%** | **12 GB VRAM · Optimal Pareto sweet spot** |
| [LFM2.5-24B-A2B-GSQ-RCO-IQ4_XS.gguf](LFM2.5-24B-A2B-GSQ-RCO-IQ4_XS.gguf) | 4.60 | 13.84 GB | 101.9% | 16 GB VRAM · Matches `21.86 GB` UD-Q6_K_XL accuracy |
| [LFM2.5-24B-A2B-DSpark-F16.gguf](LFM2.5-24B-A2B-DSpark-F16.gguf) *(Drafter)* | 16.00 | 664 MB | — | Speculative drafter (`--spec-type draft-dspark`) |
| [LFM2.5-24B-A2B-DSpark-Q8_0.gguf](LFM2.5-24B-A2B-DSpark-Q8_0.gguf) *(Drafter)* | 8.50 | 356 MB | — | Lightweight speculative drafter |

## Benchmarks

Evaluated across **AIME 2025** (`30`), **GPQA Diamond** (`198`), and **LiveCodeBench v6** (`175`) following the 3-task evaluation suite of [ISTA-DASLab](https://huggingface.co/ISTA-DASLab):

| Model | bpw | Size | AIME 2025 | GPQA Diamond | LiveCodeBench v6 | 3-Task Avg | Recovery |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [BF16 Base](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF?show_file_info=LFM2.5-24B-A2B-BF16.gguf) | 16.00 | 47.96 GB | 80.00% | **75.76%** | 54.86% | 70.20% | 100.0% |
| [Unsloth UD-Q6_K_XL](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF?show_file_info=LFM2.5-24B-A2B-UD-Q6_K_XL.gguf) | 7.30 | 21.86 GB | **86.67%** | 72.73% | 55.43% | **71.61%** | 102.0% |
| **[GSQ-RCO-IQ4_XS](LFM2.5-24B-A2B-GSQ-RCO-IQ4_XS.gguf)** | **4.60** | **13.84 GB** | **86.67%** | **73.74%** | 54.29% | **71.56%** | **101.9%** |
| **[GSQ-RCO-IQ3_XS](LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf)** *(Recommended)* | **3.50** | **10.54 GB** | **83.33%** | 70.71% | 54.29% | **69.44%** | **98.9%** |
| **[GSQ-RCO-IQ2_XS](LFM2.5-24B-A2B-GSQ-RCO-IQ2_XS.gguf)** | **2.85** | **8.52 GB** | 73.33% | 66.67% | **58.86%** | 66.29% | 94.4% |
| **[GSQ-RCO-IQ2_XXS](LFM2.5-24B-A2B-GSQ-RCO-IQ2_XXS.gguf)** | **2.65** | **7.93 GB** | 70.00% | 66.67% | 52.00% | 62.89% | 89.6% |

## Quickstart

### llama.cpp

**Speculative Decoding with `DSpark` (`IQ3_XS`)**
```bash
llama-cli \
  -m LFM2.5-24B-A2B-GSQ-RCO-IQ3_XS.gguf \
  -md LFM2.5-24B-A2B-DSpark-F16.gguf \
  --spec-type draft-dspark --spec-draft-n-max 3 \
  -ngl 99 -ngld 99 --flash-attn on -b 2048 -c 32768 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05 -cnv
```

**64K / 128K Long-Context on 12 GB VRAM (`IQ2_XS` · `125–130 tok/s` decode, `~1,500 tok/s` prefill)**  
<sub>Uses `~9.88 GB` VRAM (`2.38 GB` free) at 64K (`-c 65536`) and `~10.82 GB` VRAM (`1.45 GB` free) at 128K (`-c 131072`):</sub>
```bash
llama-server \
  -m LFM2.5-24B-A2B-GSQ-RCO-IQ2_XS.gguf \
  -ngl 99 \
  -c 131072 \
  -b 4096 \
  -ub 2048 \
  -ctk q8_0 \
  -ctv q8_0 \
  -t 6 \
  -np 1 \
  --flash-attn on
```

### Ollama

```bash
ollama run hf.co/abenzerps/LFM2.5-24B-A2B-Uncensored-GSQ-RCO-GGUF:IQ3_XS
```

## Technical Details

<details>
<summary><b>Why GSQ-RCO? (Architecture &amp; Precision Map)</b></summary>
<br>

`LFM2.5-24B-A2B` combines **30 ShortConv blocks**, **10 GQA Attention layers**, and a **64-Expert Sparse MoE** (`top-2` active per token; `23.97B` total / `2.0B` active parameters). Standard uniform quantization corrupts the shared routing and convolutional state backbone, causing expert misrouting and reasoning drift at low bit-widths.

- **RCO (Routing & Convolution Shield):** Locks all 38 MoE router gates (`ffn_gate_inp`), all 30 depthwise `ShortConv` kernels (`shortconv.conv`), and all normalization/routing-bias tensors in unquantized **`F32`**; keeps all 10 GQA `K/V` projections in **`F16`**; and preserves the leading dense blocks (`blk.0–1`), shared token embeddings (`token_embd`), and `Q/O` + `ShortConv` projections in **`Q8_0`**.
- **GSQ (Hessian-Guided Sparse Expert Calibration):** Calibrates all `2,432` sparse expert matrices (`38 × 64`) over an 8,000-chunk (`16.38M`-token) math, science, and code calibration corpus (`100.0%` expert activation coverage) using uniform non-linear `_XS / _XXS` importance lattices.

| Tensor Group | Precision | Purpose |
| :--- | :---: | :--- |
| MoE Router Gates (`ffn_gate_inp`) & Expert Bias | `F32` | Prevents top-2 expert routing flips across 64 experts |
| ShortConv Kernels (`shortconv.conv`) & Norms | `F32` | Eliminates state drift across 30 ShortConv blocks |
| GQA Attention Key / Value (`attn_k`, `attn_v`) | `F16` | Preserves long-horizon retrieval across 10 GQA layers |
| Token Embeddings (`token_embd`) & Dense Blocks (`blk.0–1`) | `Q8_0` | Anchors early representations and `DSpark` draft alignment |
| Attention `Q/O` & ShortConv Projections (`in_proj`, `out_proj`) | `Q8_0` | Protects shared non-expert signal flow across all 40 layers |
| Sparse MoE Experts (`blk.2–39.ffn_*_exps`) | `IQ4_XS` – `IQ2_XXS` | Calibrated with `16.38M`-token Hessian importance matrix |

</details>

<details>
<summary><b>Evaluation Protocol &amp; Key Benchmark Findings</b></summary>
<br>

- **Evaluation Protocol:** Models were evaluated across all **403 problems** (**AIME 2025** [`30`] + **GPQA Diamond** [`198`] + **LiveCodeBench v6** [`175`]), following the reasoning & generation benchmark suite of [ISTA-DASLab](https://huggingface.co/ISTA-DASLab), under deterministic greedy decoding (`temperature = 0.0`), `40,960` context length, up to `32,768` thinking tokens per problem, and unquantized `F16` KV cache.
- **`IQ4_XS` (`13.84 GB` / `4.60 bpw`):** Achieves a **`71.56%`** 3-task average (`101.9%` recovery vs. `BF16` `70.20%`, and `267/403` [`66.25%`] total solved), matching `Unsloth UD-Q6_K_XL` (`71.61%` / `21.86 GB`) while saving **`8.02 GB` (`-36.7%`)** in footprint.
- **`IQ3_XS` (`10.54 GB` / `3.50 bpw`):** Retains **`98.9%`** of `BF16` 3-task accuracy (`69.44%` vs. `70.20%`) at `4.55×` compression, outperforming `BF16` on AIME 2025 (`83.33%` vs. `80.00%`) with `87.59%` (`353/403`) exact question-level agreement.
- **`IQ2_XS` (`8.52 GB` / `2.85 bpw`):** Retains **`94.4%`** of `BF16` 3-task accuracy (`66.29%`) at `5.63×` compression and achieves the highest LiveCodeBench v6 score (**`58.86%`** / `103/175`) across all tested models.

</details>

## Credits & License

- **Benchmark & Methodology Credit:** The 3-task reasoning & coding evaluation suite (`AIME 2025`, `GPQA Diamond`, `LiveCodeBench v6`) and the foundational **GSQ** ([arXiv:2604.18556](https://arxiv.org/abs/2604.18556)) & **RCO** ([arXiv:2605.00649](https://arxiv.org/abs/2605.00649)) quantization frameworks are credited to [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) (Deep Algorithms and Systems Lab, Institute of Science and Technology Austria).
- **Base Model & Drafter:** [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) and [LiquidAI/LFM2.5-24B-A2B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-DSpark), licensed under the [LFM Open License v1.0](LICENSE).
