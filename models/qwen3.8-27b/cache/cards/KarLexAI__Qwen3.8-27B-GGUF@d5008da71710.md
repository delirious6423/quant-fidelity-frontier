---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - quantized
  - 1-bit
  - 2-bit
  - 3-bit
  - 4-bit
  - qwen3.8
  - reasoning
  - thinking
---

<div align="center">

# Qwen3.8-27B · KLX GGUF

### 27B reasoning from 6.2 GB. It still thinks.

![License](https://img.shields.io/badge/license-Apache--2.0-3d7be0) ![Format](https://img.shields.io/badge/format-GGUF-3d7be0) ![Runtime](https://img.shields.io/badge/runtime-stock%20llama.cpp-3d7be0) ![AIME](https://img.shields.io/badge/AIME%2025%2B26-53.8%25%20at%206.2%20GB-0891b2)

</div>

The smallest GGUFs of a 27B reasoning model usually stop reasoning: they loop, run out of tokens and miss easy
problems. KLX files keep more of the model at every size. At 6.2 GB, **the KLX file solves 53.8% of AIME 2025+2026;
Unsloth's file of the same size solves 1.3%**. Standard GGUF, no fine-tuning, no custom runtime.

## ✨ Highlights

- 🧠 **53.8% on AIME 2025+2026 and 87.8% on MATH-500 from a 6.19 GB file.** Unsloth's 6.19 GB file scores 1.3% and
  32.6%.
- 📈 **At 7.2 GB, 8.75 more AIME answers right in every 100** than Unsloth's file of the same size.
- 🎯 **Near full quality from 10.9 GB.** 94.2% AIME, 94.3% IFEval and 99.0% MATH-500, against 96.0%, 93.4% and 99.2%
  for full BF16 (about 54 GB).
- 🔌 **Plain GGUF.** Only standard llama.cpp tensor types: nothing to patch, no fork to build.

## 📦 Pick your file

| File | Size | AIME 2025+2026 | IFEval | MATH-500 | Pick it for |
|---|---|---|---|---|---|
| [Qwen3.8-27B-KLX-IQ1_S.gguf](https://huggingface.co/KarLexAI/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-KLX-IQ1_S.gguf) | **6.19 GB** | 53.8 | 50.5 | 87.8 | the smallest download |
| [Qwen3.8-27B-KLX-IQ2_XXS.gguf](https://huggingface.co/KarLexAI/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-KLX-IQ2_XXS.gguf) | **7.21 GB** | 82.7 | 88.5 | 98.4 | the biggest step up per GB |
| [Qwen3.8-27B-KLX-IQ2_XS.gguf](https://huggingface.co/KarLexAI/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-KLX-IQ2_XS.gguf) | **8.00 GB** | 89.4 | 93.5 | 97.8 | strong reasoning at 8 GB |
| [Qwen3.8-27B-KLX-IQ3_XXS.gguf](https://huggingface.co/KarLexAI/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-KLX-IQ3_XXS.gguf) | **10.93 GB** | 94.2 | 94.3 | 99.0 | near full quality |
| [Qwen3.8-27B-KLX-IQ4_XS.gguf](https://huggingface.co/KarLexAI/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-KLX-IQ4_XS.gguf) | **14.25 GB** | 95.2 | 94.3 | 98.8 | the closest to full quality |
| Qwen3.8-27B BF16, for reference | about 54 GB | 96.0 | 93.4 | 99.2 | |

The file names give the size class. Each file mixes several standard tensor types.

## 🥊 Head to head

Same harness, same questions, paired differences in points with one standard error.

| Size | KLX file | Against | AIME 2025+2026 | IFEval | MATH-500 |
|---|---|---|---|---|---|
| 6.2 GB | IQ1_S | Unsloth UD-IQ1_S (6.19 GB) | **+52.5** ± 4.9 | **+28.8** ± 2.6 | **+55.2** ± 2.5 |
| 7.2 GB | IQ2_XXS | Unsloth UD-IQ2_XXS (7.27 GB) | **+8.75** ± 2.6 | **+8.3** ± 1.9 | +1.8 ± 0.9 |
| 8 GB | IQ2_XS | ISTA GSQ-RCO IQ2_XS (8.42 GB) | +0.2 ± 1.5 | -1.1 ± 1.1 | +0.2 ± 0.8 |
| 8 GB | IQ2_XS | Unsloth UD-IQ2_S (8.37 GB) | +1.0 ± 1.4 | **+2.8** ± 1.3 | -0.8 ± 0.8 |
| 10.9 GB | IQ3_XXS | Unsloth UD-IQ3_XXS (10.93 GB) | +1.0 ± 1.2 | -0.4 ± 0.8 | +0.2 ± 0.4 |
| 14.3 GB | IQ4_XS | Unsloth UD-IQ4_XS (14.25 GB) | -0.6 ± 0.9 | -1.1 ± 0.8 | +0.2 ± 0.5 |

From 10.9 GB up, both makers' files are close to full quality, and the benchmarks above cannot separate them.
Setup, every score and the file hashes: [BENCHMARKS.md](https://huggingface.co/KarLexAI/Qwen3.8-27B-GGUF/blob/main/BENCHMARKS.md).

## 🚀 Run it with two commands

```bash
hf download KarLexAI/Qwen3.8-27B-GGUF Qwen3.8-27B-KLX-IQ1_S.gguf --local-dir .
llama-server -m Qwen3.8-27B-KLX-IQ1_S.gguf -ngl 99 -c 32768 --jinja
```

Then open http://localhost:8080 and chat. Thinking is on by default, and Qwen's sampling settings (temperature 1.0,
top-k 20, top-p 0.95) are stored in the file.

### 🖼️ Images

Qwen3.8 also reads images. Download the vision projector (about 0.9 GB) and add `--mmproj`:

```bash
hf download KarLexAI/Qwen3.8-27B-GGUF mmproj-F16.gguf --local-dir .
llama-server -m Qwen3.8-27B-KLX-IQ1_S.gguf --mmproj mmproj-F16.gguf -ngl 99 -c 32768 --jinja
```

`mmproj-BF16.gguf` works the same way, with any of the files above (we tested IQ1_S and IQ4_XS on llama.cpp commit
`351025ec`).

## 🧾 The fine print

- Post-training quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). No fine-tuning. The
  architecture, tokenizer and chat template are unchanged; the multi-token prediction (MTP) head is not included.
  The vision encoder is in the separate mmproj files, unquantized (F16 and BF16), converted with upstream llama.cpp.
- Every number above comes from one harness on stock llama.cpp, in thinking mode with Qwen's sampler.
- Apache-2.0, like the base model. See [NOTICE](https://huggingface.co/KarLexAI/Qwen3.8-27B-GGUF/blob/main/NOTICE).

<div align="center">

**Made by [KarLex AI](https://karlex.ai).** If these files save you memory, a ❤️ on this repo helps others find them.

<sub>Not affiliated with Qwen, Unsloth or ISTA.</sub>

</div>
