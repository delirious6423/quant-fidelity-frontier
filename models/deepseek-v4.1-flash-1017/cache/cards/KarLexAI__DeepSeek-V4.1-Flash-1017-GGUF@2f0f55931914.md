---
license: mit
license_link: https://huggingface.co/KarLexAI/DeepSeek-V4.1-Flash-1017-GGUF/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - quantized
  - 2-bit
  - deepseek
  - moe
  - reasoning
---

<div align="center">

# DeepSeek V4.1 Flash · KLX GGUF

### 763B parameters in 233 GB. Runs on four H100s.

![Format](https://img.shields.io/badge/format-GGUF-3d7be0) ![Runtime](https://img.shields.io/badge/runtime-stock%20llama.cpp-3d7be0) ![GPUs](https://img.shields.io/badge/runs%20on-4%C3%97%20H100%2080%20GB-3d7be0) ![MATH-500](https://img.shields.io/badge/MATH--500-91.0%25-0891b2)

</div>

DeepSeek V4.1 Flash is a 763B-parameter mixture-of-experts model, 510 GB as released. **KLX-IQ1_S is 233 GB and runs
on four H100 80 GB GPUs with eight 80K-token conversations at once.** Standard GGUF, stock llama.cpp, no fork.

## ✨ Highlights

- 🖥️ **Four GPUs, not a cluster.** On 4× H100 80 GB with eight parallel 81,920-token slots, each GPU holds about
  44 GiB; the rest of the file stays memory-mapped in system RAM.
- 🧠 **Still reasons.** 100% on AIME 2025, 96.7% on AIME 2026, 91.0% on MATH-500 and 95.5% on IFEval (strict).
- ⚖️ **Matches Unsloth's UD-IQ1_S, and sits closer to the full model.** On the same 1,099 questions and
  seeds, the scores are equal within error, with 24% lower KL divergence to the full model and a
  slightly smaller file.
- ⚡ **Shorter reasoning.** About 16% fewer tokens per MATH-500 answer and 18% fewer per AIME 2025 answer than
  UD-IQ1_S in the same runs.
- 🔌 **Plain GGUF.** Only standard llama.cpp tensor types; the chat template is built in.

## 📦 The file

| Folder | Size | Mean KL vs. full model ↓ | Pick it for |
|---|---|---|---|
| [KLX-IQ1_S](https://huggingface.co/KarLexAI/DeepSeek-V4.1-Flash-1017-GGUF/tree/main/KLX-IQ1_S) | **232.6 GB** | 0.082 | running DeepSeek V4.1 Flash on 4× H100 |

You need more memory than the file size, in GPU memory plus system RAM. Scores, the head-to-head and how every
number was measured: [BENCHMARKS.md](https://huggingface.co/KarLexAI/DeepSeek-V4.1-Flash-1017-GGUF/blob/main/BENCHMARKS.md).

## 🚀 Run it with two commands

```bash
hf download KarLexAI/DeepSeek-V4.1-Flash-1017-GGUF --include "KLX-IQ1_S/*" --local-dir .
llama-server -m KLX-IQ1_S/DeepSeek-V4.1-Flash-1017-KLX-IQ1_S-00001-of-00006.gguf --fit on -c 131072 --jinja --reasoning-format deepseek --temp 1.0 --top-p 1.0
```

Then open http://localhost:8080 and chat. You need a llama.cpp build with DeepSeek V4.1 support (we tested commit
`bcb0152`). With less GPU memory, `--fit on` keeps part of the experts in system RAM.

### 🖼️ Images

DeepSeek V4.1 Flash also reads images. Download the vision projector (about 1 GB) and add `--mmproj`:

```bash
hf download KarLexAI/DeepSeek-V4.1-Flash-1017-GGUF mmproj-F16.gguf --local-dir .
llama-server -m KLX-IQ1_S/DeepSeek-V4.1-Flash-1017-KLX-IQ1_S-00001-of-00006.gguf --mmproj mmproj-F16.gguf --fit on -c 131072 --jinja --reasoning-format deepseek --temp 1.0 --top-p 1.0
```

`mmproj-BF16.gguf` works the same way (we tested llama.cpp commit `351025ec`).

## 🧾 The fine print

- Post-training quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017).
  No fine-tuning. The vision encoder is in the separate mmproj files, unquantized (F16 and BF16), converted with
  upstream llama.cpp and tested with KLX-IQ1_S.
- Chat template: Unsloth's llama.cpp port of DeepSeek's template (MIT), which renders the same prompts as DeepSeek's
  own `chat_template.jinja`.
- Use is governed by the MIT license of the base model: see [LICENSE](https://huggingface.co/KarLexAI/DeepSeek-V4.1-Flash-1017-GGUF/blob/main/LICENSE).

<div align="center">

**Made by [KarLex AI](https://karlex.ai).** If this file saves you GPUs, a ❤️ on this repo helps others find it.

<sub>Not affiliated with DeepSeek or Unsloth.</sub>

</div>
