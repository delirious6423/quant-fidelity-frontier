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

# Qwen3.8-27B for YASAKANI - quality model

A model folder of **YASAKANI** (https://github.com/yasakani-dev/yasakani), an open-source (Apache License 2.0) local LLM engine for **Windows** and
**one NVIDIA RTX 5090**. These files are in YASAKANI's own file format. **Only YASAKANI can load them**; other engines (transformers, llama.cpp,
vLLM, SGLang ...) cannot.

日本語：YASAKANI 専用のモデルのフォルダー（品質モデル。ふだん使いに推奨）です。使い方は GitHub の README.ja.md にあります。

## Two models

YASAKANI publishes two conversions of Qwen3.8-27B. You need only one of them.

| | Quality model (this repository) | Fastest model |
|---|---|---|
| Main file | `qwen38_27b_quality.yskn`, 12.37 GB | `qwen38_27b_speed.yskn`, 10.31 GB |
| Made for | staying close to the original model, at high speed (recommended for everyday use) | the highest generation speed and the smallest size |
| Same most likely next token as the original model | 95.24% | 93.01% |
| KL divergence from the original model (smaller is closer) | 0.0122 | 0.0304 |
| Where | this repository | https://huggingface.co/yasakani-dev/Qwen3.8-27B-YASAKANI-speed |

The two quality rows were measured against the original BF16 model on 20 thinking (reasoning) transcripts from our own use of the model, none
of them used when the files were made (20,460 positions, FP16 key-value cache, November 8, 2026). They say how close the next-token predictions are to the
original model's. They are **not** task accuracy, and they cover this kind of text only. The speed measurements and all conditions are on
the GitHub page.

## Files

| File | What it is |
|---|---|
| `qwen38_27b_quality.yskn` | Qwen3.8-27B, converted to the YASAKANI format and compressed to 12.37 GB |
| `bdraft2_k8.yskn`, `bdraft2_k8.shead` | the small draft model that proposes next tokens, derived from the DFlash 2 draft model for Qwen3.8-27B (z-lab) |
| `kvq_u5.bin` | settings for the compressed key-value cache |
| `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `config.json`, `generation_config.json` | the original Qwen3.8-27B files, unchanged |
| `selfcheck.json` | the reference output for `yasakani selfcheck` |
| `yasakani.json` | the list of files with their SHA-256 checksums, used by `yasakani verify` |
| `LICENSE`, `NOTICE` | the Apache License 2.0, and who made what |

About 15 GB in total. Keep all files together in one folder.

## How to use

1. Install YASAKANI (Windows 11, an RTX 5090, Python 3.10 or newer): see the GitHub page.
2. Download every file of this repository into one folder, for example `D:\models\qwen38-quality`:

```bat
hf download yasakani-dev/Qwen3.8-27B-YASAKANI-quality --local-dir D:\models\qwen38-quality
```

3. Check the download, then start:

```bat
yasakani verify    --model-dir D:\models\qwen38-quality
yasakani selfcheck --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
yasakani chat      --model-dir D:\models\qwen38-quality --dll D:\yasakani\yskn.dll
```

## Not the original model

The model is compressed to 12.37 GB (the original BF16 weights are 53.8 GB), so it is **not identical** to the original Qwen3.8-27B. On the
thinking texts described above, its most likely next token is the same as the original model's 95.24% of the time. On general web text, an
additional evaluation showed a KL divergence 3-10% larger than our earlier conversion of the same size, so the result on thinking text does
not carry over to every kind of text. The details, the speed measurements and the limits of this release are on the GitHub page.

This repository is not an official release of Qwen, Alibaba Cloud, z-lab or Inco AI, and none of them endorses it.

## License and credits

Apache License 2.0 (`LICENSE`). Derived from Qwen3.8-27B by the Qwen team (Alibaba Cloud) and from the DFlash 2 draft model for Qwen3.8-27B by
z-lab (Inco AI); see `NOTICE`. The files `qwen38_27b_quality.yskn`, `bdraft2_k8.yskn` and `bdraft2_k8.shead` are **not** the original weights.
