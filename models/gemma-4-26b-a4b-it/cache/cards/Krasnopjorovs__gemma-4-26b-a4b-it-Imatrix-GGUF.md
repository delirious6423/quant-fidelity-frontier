---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - imatrix
  - llama.cpp
  - quantized
  - verified
quantized_by: Krasnopjorovs
---

# 🐘 gemma-4-26b-a4b-it — Imatrix GGUF

> **Verified quants only.** Sub-4-bit variants are excluded from this release because they produce degenerate output on this model size — no point shipping broken files.

GGUF imatrix builds of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).

Built with [`llama.cpp`](https://github.com/ggerganov/llama.cpp) and importance-matrix calibration on a public multilingual + code + math corpus. Every quant is loaded, prompted, and visually checked before publication.

---

## 👋 From the author

Hi local LLM enthusiasts! You may have noticed a few quants from me over the past months — those were one-off experiments. This release is the first one from a fully automated pipeline.

**Please try them out, share feedback, and let me know which quants you find useful or which ones you wish I had made.** For now, anything below 4-bit produces degraded output on this architecture, so I am holding off on those until they actually work well. More quants and more models coming.

---


## 🎯 Pick a quant

| Quant | Size | What it's for |
|---|---|---|
| [`Q8_0`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q8_0.gguf) | 26G | Almost the original. Pick this if RAM isn't a concern. |
| [`Q6_K_L`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q6_K_L.gguf) | 22G | Near-lossless with Q8_0 embeddings. Best of the K-quants. |
| [`Q6_K`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q6_K.gguf) | 22G | Near-lossless. Excellent fidelity at smaller size than Q8_0. |
| [`Q5_K_L`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q5_K_L.gguf) | 18G | Q5_K_M with Q8_0 embeddings. High quality, small overhead. |
| [`Q5_K_M`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q5_K_M.gguf) | 18G | Sweet spot between Q4 and Q6. Solid all-rounder. |
| [`Q5_K_S`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q5_K_S.gguf) | 17G | Slightly smaller than Q5_K_M, virtually identical output. |
| [`Q4_K_L`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q4_K_L.gguf) | 16G | Q4_K_M with Q8_0 embeddings. The smart 4-bit choice. |
| [`Q4_K_M`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q4_K_M.gguf) | 16G | The default. Most-downloaded quant for a reason. |
| [`Q4_K_S`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-Q4_K_S.gguf) | 15G | A bit smaller than Q4_K_M, a tiny step down. |
| [`IQ4_NL`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-IQ4_NL.gguf) | 14G | ARM-optimized 4-bit. For Raspberry Pi & friends. |
| [`IQ4_XS`](https://huggingface.co/Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF/blob/main/gemma-4-26b-a4b-it-IQ4_XS.gguf) | 13G | Tightest 4-bit format. Quality close to Q4_K_S, smaller. |


`*_L` variants override the output tensor and embeddings to Q8_0 — small disk cost, better output stability.

**Tl;dr:**
- 🟢 **Q4_K_M** — start here unless you have a reason not to
- 🟡 **Q5_K_M** — small quality bump if you have RAM to spare
- 🔵 **Q8_0** — pick this if you're not RAM-constrained and want max quality
- 🟠 **IQ4_XS** — when 4-bit is too big
- 🟣 **IQ4_NL** — running on ARM (Pi, phones)

---

## 💬 Prompt format

```
<bos><start_of_turn>user
{prompt}<end_of_turn>
<start_of_turn>model
```

---

## 📥 Download

Single file (recommended):

```bash
hf download Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF \
  --include "gemma-4-26b-a4b-it-Q4_K_M.gguf" --local-dir .
```

Whole repo:

```bash
hf download Krasnopjorovs/gemma-4-26b-a4b-it-Imatrix-GGUF --local-dir ./gemma-4-26b-a4b-it-gguf
```

---

## ▶ Run

```bash
./llama-server \
  -m gemma-4-26b-a4b-it-Q4_K_M.gguf \
  -c 32768 -ngl 99 \
  --host 0.0.0.0 --port 8080
```

Then point your favorite chat client at `http://localhost:8080/v1`.

---

## 🔬 Calibration

Imatrix generated from [reapmix](https://huggingface.co/datasets/eaddario/imatrix-calibration) — community calibration mix (~400K tokens, multilingual + code + math). Same class of public calibration data used by other community publishers; this release makes no unique calibration claim.

The output tensor and embeddings carry disproportionate weight in quantized output. The `*_L` variants keep these at Q8_0 — small disk cost for noticeably better stability at low bit-rates.

---

## ✅ Build info

- **Source:** [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
- **llama.cpp:** latest mainline
- **Quantization:** CPU
- **Imatrix calibration:** GPU
- **Generated:** 2026-06-06

---

## 🙏 Credits

- Source model — [Google](https://huggingface.co/google) (Apache 2.0)
- Calibration corpus — [reapmix](https://huggingface.co/datasets/eaddario/imatrix-calibration) by [eaddario](https://huggingface.co/eaddario)
- Quantization tooling — [llama.cpp](https://github.com/ggerganov/llama.cpp)
