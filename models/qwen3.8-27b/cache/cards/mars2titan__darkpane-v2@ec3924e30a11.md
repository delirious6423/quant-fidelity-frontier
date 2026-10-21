---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
language:
  - en
  - zh
tags:
  - qwen3.8
  - qwen3_5
  - mlx
  - apple-silicon
  - abliterated
  - uncensored
  - vision-language
  - image-text-to-text
  - quantized
  - 4-bit
  - 8-bit
  - function-calling
  - reasoning
  - obliteratus
  - refusal-analysis
  - red-team
  - abliteration
  - safety-research
  - alignment-research
  - status-experiment
  - project-research
status: experiment
project: research
---

# Dark-Pane v2

> **Status: EXPERIMENT.** Public Dark-Pane family checkpoint.

Abliterated [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for **Apple Silicon** (MLX). Native vision-language, 262k context, 2 / 4 / 6 / 8-bit.

This is a rehost of [`orcarouter/Qwen3.8-27B-Uncensored-MLX`](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX) into the Dark-Pane family. Weights are OrcaRouter's abliterated MLX quants of Qwen's Apache-2.0 27B VL model — **not** the Gemma 4 12B SOM+ASPA surgery from [Dark-Pane](https://huggingface.co/mars2titan/Dark-Pane).

Repo root = **4-bit**. Other precisions live in `2-bit/`, `4-bit/`, `6-bit/`, `8-bit/`.

---

## ⚠️ Research use

Safety alignment has been substantially removed (abliteration). It will comply with requests the stock Qwen3.8-27B would refuse. Intended for alignment research, red-teaming, and local evaluation. You are responsible for what you generate. Add your own guardrails before any deployment.

Apache 2.0 from the Qwen base. Attribution: Qwen / Alibaba (base), OrcaRouter (abliteration + MLX quants).

---

## Quants

| Folder | Bits | Size | Min Mac RAM | Notes |
|---|---|---|---|---|
| `8-bit/` | 8 | ~27.5 GB | 32 GB | Near-lossless |
| `6-bit/` | 6 | ~22 GB | 24–32 GB | Best size/quality |
| `4-bit/` (also repo root) | 4 | ~15 GB | 24 GB | Default |
| `2-bit/` | 2 | ~8.7 GB | 16 GB | Archival only — quality collapses |

Vision tower, norms, and conv layers stay BF16. Only LM linear weights are quantized.

---

## Usage (mlx-vlm)

```bash
pip install -U mlx-vlm    # mlx-vlm >= 0.6.13, mlx >= 0.32

hf download mars2titan/darkpane-v2 --include "4-bit/*" --local-dir ./darkpane-v2

python -m mlx_vlm generate \
  --model ./darkpane-v2/4-bit \
  --prompt "Explain quantum entanglement in one sentence." --max-tokens 256

python -m mlx_vlm generate \
  --model ./darkpane-v2/4-bit \
  --image path/to/image.png \
  --prompt "Describe this image." --max-tokens 256

python -m mlx_vlm server --model ./darkpane-v2/4-bit --port 8080
```

## LM Studio

Search `mars2titan/darkpane-v2`. Root is 4-bit. Turn **off** KV cache quantization (MLX vision on this arch fails if it is on). 4-bit fits a 32 GB Mac; 8-bit wants 64 GB.

---

## Model

| | |
|---|---|
| **Base** | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Arch** | `Qwen3_5ForConditionalGeneration` — 64 layers, hidden 5120, hybrid Gated DeltaNet, native VL |
| **This repo** | Abliterated, then MLX affine quant (group size 64) |
| **Context** | 262,144 tokens |
| **Family** | [Dark-Pane](https://huggingface.co/mars2titan/Dark-Pane) (Gemma 4 12B) → this (Qwen 3.8 27B MLX) |
