---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - z-lab/Qwen3.8-27B-DFlash2
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - yasakani
  - quantized
  - windows
  - rtx-5090
---

# Qwen3.8-27B for YASAKANI (max-quality model)

The max-quality model folder of **[YASAKANI](https://github.com/yasakani-dev/yasakani)**, an open-source (Apache License 2.0) local LLM
engine for **Windows** and **one NVIDIA RTX GPU** (tested on the RTX 5090 only). These files are in YASAKANI's own file format. **Only
YASAKANI can load them**; other engines (transformers, llama.cpp, vLLM, SGLang ...) cannot.

It is the same Qwen3.8-27B as the quality model, converted with more bits. Its predictions are the closest to the original of the
YASAKANI models, and it is larger and writes at about 0.8 to 0.9 times the speed of the quality model. For everyday use the quality model is still the recommended one.

日本語：YASAKANI 専用の「最高品質モデル」のフォルダーです。品質モデルより元のモデルに近く、ファイルは大きく、生成は品質モデルの約 0.8〜0.9 倍の速さです。使い方は GitHub の README.ja.md にあります。

## Which YASAKANI

This file uses 5-bit codes. It needs YASAKANI 0.2.0 or newer; earlier versions stop with an error when they open it.

## Files

| File | What it is |
|---|---|
| `qwen38_27b_max_quality.yskn` | Qwen3.8-27B, converted to the YASAKANI format and compressed to 15.72 GB |
| `bdraft2_k8.yskn`, `bdraft2_k8.shead` | the small draft model that proposes next tokens, derived from the DFlash 2 draft model for Qwen3.8-27B (z-lab); the same files as in the quality model |
| `kvq_u5.bin` | settings for the compressed key-value cache |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `config.json`, `generation_config.json` | the original Qwen3.8-27B files, unchanged |
| `selfcheck.json` | the reference output for `yasakani selfcheck` (made on the RTX 5090) |
| `yasakani.json` | the list of files with their SHA-256 checksums, used by `yasakani verify` |
| `LICENSE`, `NOTICE` | the Apache License 2.0, and who made what |

About 18.3 GB in total. Keep all files together in one folder.

| | quality model | max-quality model |
|---|---|---|
| the 400 matrices of the body | 288 at 4 bits, 112 at 3 bits per weight | 288 at 5 bits, 112 at 4 bits per weight |
| input embedding table | 3 bits | 4 bits |
| output matrix | 4 bits | 5 bits |
| weights file | 12.37 GB | 15.72 GB |

## How to use

1. Install YASAKANI (Windows 11, Python 3.10 or newer, see the GitHub page). The quickstart offers this model as its third choice
   (`START.cmd --model max`).
2. Or download every file of this repository into one folder, for example `D:\models\qwen38-max-quality`:

```bat
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-max-quality --local-dir D:\models\qwen38-max-quality
yasakani verify    --model-dir D:\models\qwen38-max-quality
yasakani selfcheck --model-dir D:\models\qwen38-max-quality --dll D:\yasakani\yskn.dll
yasakani chat      --model-dir D:\models\qwen38-max-quality --dll D:\yasakani\yskn.dll
```

## How close to the original

Measured on one RTX 5090 on November 20, 2026, with the method of `tools/quality/` on GitHub: 20 thinking texts that were not used in the
conversion (20,460 positions), against the original BF16 model, with the engine's default KV5 cache.

| Model | Weights file | KL on thinking text (lower is closer) | Same most likely next token |
|---|---|---|---|
| quality model | 12.37 GB | 0.0126 | 95.35% |
| max-quality model | 15.72 GB | 0.0040 | 97.40% |

This measures agreement with the original model on thinking text, not task accuracy. The model is **not identical** to the original
Qwen3.8-27B.

## Speed and memory

Against the quality model on the same PC (one RTX 5090, default settings, one request at a time, November 20, 2026):

| Task | quality model | max-quality model | Ratio |
|---|---|---|---|
| `yasakani bench`, 4 tasks, thinking off (mean, tok/s) | 439.3 | 403.4 | 0.92 |
| `yasakani bench`, 4 tasks, thinking on (mean, tok/s) | 392.4 | 317.9 | 0.81 |
| code edit (tok/s) | 2,149.2 | 2,123.2 | 0.99 |
| JSONL edit (tok/s) | 2,335.8 | 2,267.6 | 0.97 |
| long JSONL edit (tok/s) | 2,157.5 | 2,102.7 | 0.97 |
| first read of 8,192 tokens (input tok/s) | 5,174 | 5,118 | 0.99 |
| first read of 32,768 tokens (input tok/s) | 4,739 | 4,719 | 1.00 |

Bench: 512 tokens, greedy, median of 6 runs per task (2 loads, 3 runs each). Edits: thinking off, median of 6 runs. Reads: median of 4.
Reading and edits run at about the same speed as the quality model; plain generation at about 0.8 to 0.9 times.

GPU memory (nvidia-smi): 25,977 MiB after loading (quality model: 22,795 MiB), at most 26,821 MiB while reading 32,768 tokens and
generating 2,048 (quality model: 23,637 MiB). Context: 262,144 tokens on the RTX 5090.

## Limits

* Tested on the RTX 5090 only. The code for RTX 30 and 40 cards was run only on the RTX 5090, not on those cards.
* 24 GB cards (RTX 3090, 3090 Ti, 4090): by the package's own calculation the context would be 163,840 tokens with 500 MiB in use by the
  desktop (196,608 with nothing else on the GPU). This is a calculation, not checked on such a card.
* 16 GB cards: the model does not fit. Use the fastest model there.
* Generation is about 0.8 to 0.9 times as fast as with the quality model (above).

The details are on the GitHub page. This repository is not an official release of Qwen, Alibaba Cloud, z-lab or Inco AI, and none of
them endorses it.

## License and credits

Apache License 2.0 (`LICENSE`). Derived from Qwen3.8-27B by the Qwen team (Alibaba Cloud) and from the DFlash 2 draft model for
Qwen3.8-27B by z-lab (Inco AI); see `NOTICE`. The files `qwen38_27b_max_quality.yskn`, `bdraft2_k8.yskn` and `bdraft2_k8.shead` are
**not** the original weights.
