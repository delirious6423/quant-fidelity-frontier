---
license: apache-2.0
library_name: llama.cpp
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
language:
- ko
- en
tags:
- gguf
- llama.cpp
- quantized
- conversational
- multimodal
- vision-language
- archsloth
- autoround
- kl-divergence
- korean
- multilingual
- local-llm
- on-device
- qwen3_5
- 27b
---

<p align="center">
  <img src="archsloth_hero.png" alt="ARCHSLOTH — the summit of quantization" width="820">
</p>

<p align="center">
  <b>Most quantized weights ship with an adjective. Ours ship with a table.</b>
</p>

<p align="center">
  <b>Head to head with <a href="https://huggingface.co/unsloth/Qwen3.8-27B-GGUF"><code>unsloth/Qwen3.8-27B-GGUF</code></a>,
  same <code>Q6_K</code> rung &mdash; KL divergence from bf16, lower is better:</b><br>
  Korean <b>&minus;36.9&#8201;%</b> &#183; English <b>&minus;15.0&#8201;%</b><br>
  <sub>Two axes measured, two ahead. What we did not win is further down, with its number.</sub>
</p>

---

> ### 📚 Collection
> **▶ [Archsloth MODELs](https://huggingface.co/collections/Archsloth/archsloth-models)** — every build in the ARCHsloth Q4 grade
> [Qwen3-4B](https://huggingface.co/Archsloth/Qwen3-4B-GGUF) · [Qwen3.5-9B](https://huggingface.co/Archsloth/Qwen3.5-9B-GGUF) · [Qwen3.8-27B](https://huggingface.co/Archsloth/Qwen3.8-27B-GGUF)

# Qwen3.8-27B · ARCHsloth Q6

### **Same rung. Same tensor types. Only the rounding differs.**

[![License](https://img.shields.io/badge/License-Apache_2.0-0f6e56)](https://www.apache.org/licenses/LICENSE-2.0)
[![Runtime](https://img.shields.io/badge/runtime-stock_llama.cpp-eb6834)](https://github.com/ggml-org/llama.cpp)
[![Floor](https://img.shields.io/badge/floor-Q4-b8860b)]()
[![Base](https://img.shields.io/badge/base-Qwen3.8--27B-2a78d6)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Vision](https://img.shields.io/badge/vision-mmproj_included-4a3aa7)]()

---

## Which file do I want?

| File | Size | What it is |
|---|---|---|
| **`Qwen3.8-27B-AS-Q6_K.gguf`** ⭐ | 22.43 GB | The weights. This is the model. |
| `mmproj-F32.gguf` | 1.84 GB | The vision tower. **Only needed if you want the model to see images.** |

Qwen3.8-27B is a vision-language model. Text-only chat needs the first file alone; pass
`--mmproj mmproj-F32.gguf` to `llama-server` to turn the eyes on. The vision tower is **not**
quantized — it is carried at full precision, exactly as the base model shipped it.

**There is no Q4 file in this repository.** [Why not.](#what-we-did-not-win)

---

## `[measured]` Q6_K — the same rung, both sides

KL divergence from the bf16 original. **Lower is better.** `llama-perplexity --kl-divergence`,
ctx 512, 60 chunks, held-out Korean and English text that is not in the calibration set.

<p align="center"><img src="chart_axes.svg" alt="KL divergence, ARCHsloth against unsloth, Korean and English" width="820"></p>

| | bytes | Korean | English |
|---|---:|---:|---:|
| **ARCHsloth `Q6_K`** | **22,430,999,840** | **0.001385** | **0.001254** |
| unsloth `UD-Q6_K` | 21,983,677,344 | 0.002195 | 0.001476 |
| unsloth `UD-Q6_K_M` | 23,088,409,504 | 0.001632 | 0.001137 |

| Against | Korean | English |
|---|---:|---:|
| unsloth `UD-Q6_K` — same rung | **−36.9 %** | **−15.0 %** |
| unsloth `UD-Q6_K_M` — 2.9 % *larger* than ours | **−15.1 %** | +10.3 % |

Same-top-p moves the same direction as the divergence on every row (98.222 % vs 97.693 % on
Korean). Full statistics: [`EVAL.md`](EVAL.md). Raw per-run logs: [`eval/logs/`](eval/logs).

---

## Quickstart

```bash
# text only
llama-cli -m Qwen3.8-27B-AS-Q6_K.gguf -p "안녕하세요" -ngl 99

# OpenAI-compatible server, with vision
llama-server -m Qwen3.8-27B-AS-Q6_K.gguf --mmproj mmproj-F32.gguf \
  --host 127.0.0.1 --port 8080 --jinja
```

A 27B is a dense model: every token reads the whole file. Plan for the weights **plus** the KV
cache, and the cache is what decides whether a long-context session fits.

---

## What is actually different

```
--scheme GGUF:Q6_K       optimize the rounding for the quantizer that is used at export
--enable_alg_ext         sign-gradient rounding search (SignRoundV2, arXiv:2512.04746)
--dataset cal_archsloth.jsonl
--iters 200  --nsamples 128
```

Nothing else. Bit widths and tensor types are llama.cpp's standard `Q6_K`. Nothing below
4 bits anywhere.

### The calibration set — it ships in this repo

`cal_archsloth.jsonl`, 500 samples of exactly 9,000 characters each, Korean and English
**interleaved at the sample level**. Interleaving is what mattered, not the ratio.
Details: [`CALIBRATION.md`](CALIBRATION.md).

---

## What we did not win

- **`[measured]` Our Q4_K_M lost English, so it is not in this repository.** Against
  `Qwen3.8-27B-UD-Q4_K_M.gguf` it took Korean by **−12.8 %** and gave English away by
  **+15.8 %**. A rung we do not win is a rung we do not ship. It returns when it wins.
- **`[measured]` `UD-Q6_K_M` keeps English.** That file is 2.9 % *larger* than ours. We take
  Korean from it by **−15.1 %** and it takes English back by **+10.3 %**. Against the file at
  our own rung, `UD-Q6_K`, we are ahead on both.
- **`[measured]` Two axes, not ten.** Korean and English are measured here. The ten-axis sweep
  that the [4B card](https://huggingface.co/Archsloth/Qwen3-4B-GGUF) carries has not been run
  on this model. We publish only what we ran ourselves.
- **`[pending]`** Long-context retrieval, agentic tool use, and the vision axis.

---

## Reproduce this

```bash
# 1. the comparison file, straight from its own repository
hf download unsloth/Qwen3.8-27B-GGUF Qwen3.8-27B-UD-Q6_K.gguf --local-dir rival

# 2. the reference the distance is measured from
python llama.cpp/convert_hf_to_gguf.py Qwen/Qwen3.8-27B --outtype bf16 --outfile ref-BF16.gguf

# 3. teacher logits, once per evaluation text
llama-perplexity -m ref-BF16.gguf -f eval/eval_ko.txt \
  --kl-divergence-base base_ko.dat -ngl 99 -c 512 --chunks 60

# 4. every candidate against the same teacher, same text, same chunks
llama-perplexity -m <candidate>.gguf -f eval/eval_ko.txt \
  --kl-divergence --kl-divergence-base base_ko.dat -ngl 99 -c 512 --chunks 60
```

If a number here does not replicate on your hardware, open a discussion and we will correct it.

---

## Lineage

| Component | Origin |
|---|---|
| Weights | **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, unmodified — quantized from the original, not from anyone else's quantization |
| Layout, bit widths, tensor types | llama.cpp standard `Q6_K`, unchanged |
| Vision tower | carried at full precision, not quantized |
| Runtime | upstream llama.cpp, unmodified |
| Rounding search | Intel **[AutoRound](https://github.com/intel/auto-round)** with `enable_alg_ext` (SignRoundV2) |
| **Calibration corpus, evaluation set, and the measurement** | **ours** — all three ship in this repo |

---

## License

Apache-2.0, inherited from the base model. Comparison files are named so you can fetch the
exact bytes we measured.
