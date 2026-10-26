---
license: apache-2.0
library_name: splash
pipeline_tag: text-generation
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - splash
  - apple-silicon
  - metal
  - local-inference
  - dflash2
  - speculative-decoding
  - qwen3.8
  - 8-bit
  - lm-studio
---

# Qwen3.8-27B Splash 8-bit (unofficial)

A Splash package of Qwen3.8-27B with an **8-bit target model (GGUF Q8_0)**, for the Splash runtime in LM Studio on Apple
silicon. It pairs the 8-bit weights with the same DFlash2 draft, vision tower, tokenizer and chat template as the
official 4-bit package [incoai/Qwen3.8-27B-Splash](https://huggingface.co/incoai/Qwen3.8-27B-Splash).

**Unofficial and independent.** Not made or endorsed by Inco AI (Splash, DFlash2), LM Studio, Alibaba Cloud (Qwen) or
Unsloth. No weights were changed: every weight file is a byte-for-byte copy of its source.

## Why 8-bit

The official package's target is 4-bit. 8-bit weights stay much closer to the original model, at roughly 60% of the
4-bit package's writing speed (see [Measurements](#measurements)).

## What's inside

| Part | Source | Notes |
| --- | --- | --- |
| Target model | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) @ `4ca7207`, `Qwen3.8-27B-Q8_0.gguf` | unchanged |
| Draft model | [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) @ `015e795` | unchanged BF16; Splash quantizes it to 4-bit at load, exactly as in the official package |
| Vision tower | `mmproj-BF16.gguf` from the Unsloth repo; `vision/model.bin` from the official package | the engine reads the mmproj; `vision/model.bin` is there so that LM Studio enables image input |
| Tokenizer, chat template | [incoai/Qwen3.8-27B-Splash](https://huggingface.co/incoai/Qwen3.8-27B-Splash) | identical to the official package |
| `model.json`, `config.json` | written by Splash's own installer (splash `0d5a035e`, `install/models.py --model unsloth/Qwen3.8-27B-GGUF:Q8_0 prepare`) | local file paths removed |
| `manifest.json` | this package | lets LM Studio list and download it |

## Requirements

- Apple silicon M3 or newer, macOS 26.4 or later (Splash's requirements).
- LM Studio with the Splash runtime enabled (Settings → Runtime → Splash). Tested with LM Studio 0.4.26+4 and the Splash
  runtime 1.3.0.
- About 32 GB of memory for the model when loaded; tested on a 64 GB Mac. Download size: 34.8 GB.

## Install

In LM Studio, enable the Splash runtime, then download `dmeight/Qwen3.8-27B-Splash-8bit` from the model search. It
appears as **Qwen3.8 27B Splash 8bit**.

## Known issues

- **LM Studio labels it "4bit".** LM Studio currently gives every Splash package a fixed 4-bit label, whatever its
  weights. The weights here are 8-bit: the package is 34.8 GB and takes about 31.5 GB of memory when loaded, while the
  official 4-bit package is 17.4 GB.
- **It relies on undocumented behaviour, so an update may break it.** LM Studio lists a folder as a Splash model only if
  its `manifest.json` names the 4-bit package format (`splash-packed-q4`). The Splash engine reads an installer
  assembly (`model.json`) before a manifest. This package carries both: the manifest for LM Studio, the assembly for the
  engine. A newer LM Studio or Splash runtime may stop loading it.
- **It is a stopgap.** Once LM Studio ships Splash 1.3.1 or later, which loads 8-bit models directly, prefer the
  official route.

## Without LM Studio

This package is for LM Studio only. Elsewhere, use the original sources:

- **Standalone Splash** rejects this package, because its installer expects the official package's per-layer files.
  It doesn't need the package either: `splash serve --model unsloth/Qwen3.8-27B-GGUF:Q8_0` builds the same set (the
  same GGUF, vision projector and DFlash2 draft) from the original repositories. Splash 1.3.1 and later also load MLX
  8-bit (`mlx-community/Qwen3.8-27B-8bit`).
- **llama.cpp, Ollama and other GGUF tools** can use `target/Qwen3.8-27B-Q8_0.gguf` and `vision/mmproj.gguf` as they
  are: plain GGUF files, identical to Unsloth's. They run without the DFlash2 speedup. It is simpler to download them
  from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
- **MLX tools** cannot use it: the weights are GGUF, not MLX.

## Measurements

Mac Studio, Apple M5 Max (40-core GPU), 64 GB, macOS 27.0.1, 2026-10-26.

**Speed through LM Studio.** Mixed prompts (Polish and English prose, code), thinking on, temperature 1.0, top_p 0.95,
top_k 20, 800-token answers, two runs each, median.

| | This package (8-bit) | Official Splash (4-bit) | LM Studio MLX 8-bit, no speculative decoding |
| --- | ---: | ---: | ---: |
| Writing, tokens/s | 54.8 | 91.2 | 18.1 |
| Writing at a 15K-token context, tokens/s | 39.2 | 80.7 | 17.2 |
| Reading a 15K-token prompt, s | 13.5 | 13.3 | 17.8 |

**Draft acceptance.** The share of DFlash2 proposals the target accepted. 24 prompts in six groups, 512-token answers,
ABBA order (4-bit, 8-bit, 8-bit, 4-bit), counters read from the Splash engine.

| | 4-bit target | 8-bit target |
| --- | ---: | ---: |
| Sampling (T 1.0, two seed sets): accepted | 25.7% | 28.5% |
| Tokens per verify cycle | 2.80 | 3.00 |
| Greedy: accepted | 29.1% | 28.7% |

With sampling, the 8-bit target accepts more of the draft's proposals: +3.7 points on average over the 24 prompts, 95%
bootstrap interval +1.1 to +6.5. Greedy decoding shows no difference. Greedy outputs were identical across repeated
runs.

## Licenses and credits

Every part is licensed under Apache-2.0 (see `LICENSE`); `NOTICE` lists the sources and what this package adds.

- Qwen3.8-27B by Alibaba Cloud (Qwen team).
- GGUF quantization and vision projector by Unsloth.
- Splash engine, package format, DFlash2 draft model and the official package's tokenizer and vision files by
  Inco AI.

Packaged by dmeight.
