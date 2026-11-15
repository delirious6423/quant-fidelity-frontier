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

# Qwen3.8-27B for YASAKANI - fastest model

A model folder of **YASAKANI** (https://github.com/yasakani-dev/yasakani), an open-source (Apache License 2.0) local LLM engine for **Windows** and
**one NVIDIA RTX GPU** (tested on the **RTX 5090**; other cards: see GPU below). These files are in YASAKANI's own file format.
**Only YASAKANI can load them**; other engines (transformers, llama.cpp, vLLM, SGLang ...) cannot.

日本語：YASAKANI 専用のモデルのフォルダー（最速モデル）です。使い方は GitHub の README.ja.md にあります。

## Two models

YASAKANI publishes two conversions of Qwen3.8-27B. You need only one of them.

| | Quality model | Fastest model (this repository) |
|---|---|---|
| Main file | `qwen38_27b_quality.yskn`, 12.37 GB | `qwen38_27b_speed.yskn`, 10.31 GB |
| Made for | staying close to the original model, at high speed (recommended for everyday use) | the highest generation speed and the smallest size |
| GPU memory | 24 GB or more | 16 GB or more |
| Same most likely next token as the original model | 95.24% | 93.01% |
| KL divergence from the original model (smaller is closer) | 0.0122 | 0.0304 |
| Where | https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality | this repository |

The two quality rows were measured against the original BF16 model on 20 thinking (reasoning) transcripts from our own use of the model, none
of them used when the files were made (20,460 positions, FP16 key-value cache, November 8, 2026). They say how close the next-token predictions are to the
original model's. They are **not** task accuracy, and they cover this kind of text only. The speed measurements and all conditions are on
the GitHub page.

## GPU

YASAKANI has been tested only on the **NVIDIA GeForce RTX 5090** (32 GB). Since version 0.1.3 it is also built to run on NVIDIA RTX 30,
40 and 50 series cards with 16 GB of GPU memory or more:

| GPU memory | Cards | Models |
|---|---|---|
| 32 GB | RTX 5090 (tested) | both models |
| 24 GB | RTX 3090, RTX 3090 Ti, RTX 4090 | both models |
| 16 GB | RTX 4060 Ti 16GB, RTX 4070 Ti SUPER, RTX 4080, RTX 4080 SUPER, RTX 5060 Ti 16GB, RTX 5070 Ti, RTX 5080 | the fastest model only |

**This model (the fastest model) needs 16 GB or more**, so it is the one to use on a 16 GB card. The quality model needs 24 GB or more:
https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-quality

Running on a card is not the same as being optimized for it. YASAKANI is not optimized for the cards other than the RTX 5090, and we
have not tested it on them (we have only an RTX 5090), so we cannot guarantee that it runs on them or how fast it is. These cards are
slower than the RTX 5090, and we expect the speed-up from YASAKANI to be smaller on them; all speed numbers on the GitHub page were
measured on the RTX 5090. On a card with less than 32 GB, the context length is set from the free GPU memory each time the model starts,
so it can be shorter. Reports from your card are welcome as a GitHub issue: https://github.com/yasakani-dev/yasakani/issues

YASAKANI does not run on cards with 12 GB or less, or on the RTX 20 series and older. AMD GPUs and Macs are not officially supported.

## Files

| File | What it is |
|---|---|
| `qwen38_27b_speed.yskn` | Qwen3.8-27B, converted to the YASAKANI format and compressed to 10.31 GB |
| `bdraft2_k8.yskn`, `bdraft2_k8.shead` | the small draft model that proposes next tokens, derived from the DFlash 2 draft model for Qwen3.8-27B (z-lab) |
| `kvq_u5.bin` | settings for the compressed key-value cache |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `config.json`, `generation_config.json` | the original Qwen3.8-27B files, unchanged |
| `selfcheck.json` | the reference output for `yasakani selfcheck` |
| `yasakani.json` | the list of files with their SHA-256 checksums, used by `yasakani verify` |
| `LICENSE`, `NOTICE` | the Apache License 2.0, and who made what |

About 13 GB in total. Keep all files together in one folder.

## How to use

1. Install YASAKANI (Windows 11, an NVIDIA RTX GPU with 16 GB or more from the list above, Python 3.10 or newer; on a card other than the RTX 5090, version 0.1.3 or newer): see the GitHub page.
2. Download every file of this repository into one folder, for example `D:\models\qwen38-speed`:

```bat
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-speed --local-dir D:\models\qwen38-speed
```

3. Check the download, then start:

```bat
yasakani verify    --model-dir D:\models\qwen38-speed
yasakani selfcheck --model-dir D:\models\qwen38-speed --dll D:\yasakani\yskn.dll
yasakani chat      --model-dir D:\models\qwen38-speed --dll D:\yasakani\yskn.dll
```

## Not the original model

The model is compressed to 10.31 GB (the original BF16 weights are 53.8 GB), so it is **not identical** to the original Qwen3.8-27B. On the
thinking texts described above, its most likely next token is the same as the original model's 93.01% of the time, which is lower than the
quality model's 95.24%. If closeness to the original model matters more to you than speed, use the quality model (it needs 24 GB of GPU memory or more).
The details, the speed
measurements and the limits of this release are on the GitHub page.

This repository is not an official release of Qwen, Alibaba Cloud, z-lab or Inco AI, and none of them endorses it.

## License and credits

Apache License 2.0 (`LICENSE`). Derived from Qwen3.8-27B by the Qwen team (Alibaba Cloud) and from the DFlash 2 draft model for Qwen3.8-27B by
z-lab (Inco AI); see `NOTICE`. The files `qwen38_27b_speed.yskn`, `bdraft2_k8.yskn` and `bdraft2_k8.shead` are **not** the original weights.
