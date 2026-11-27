---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- mlx-lm
- qwen3_5
- layerfit
- 2bit
---

# Qwen3.8-27B LayerFit™ MLX 2-bit

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple Silicon, built with **LayerFit™**: a per-layer search that spends the bits where this model needs them most. 9.1 GB on disk, for 16 GB Macs. Standard MLX format, so stock mlx-lm, mlx-vlm and LM Studio load it. **Text-only** (vision tower removed to fit a 16 GB Mac), 262K context. No MTP head: at 2-bit it made decoding slower, so this build leaves it out.

<!-- mac-picker:start -->
## Which build fits my Mac?

Each pick leaves room for macOS and a long prompt (the download is at most about 60% of your memory). Other models for your Mac: [which build fits my Mac?](https://huggingface.co/netprojects).

| Your Mac | Build |
|---|---|
| 16 GB | **[2-bit](https://huggingface.co/netprojects/Qwen3.8-27B-LayerFit-MLX-2bit) (9.1 GB, text-only)** (this build) |
| 16 GB ⚡ stretch | [2.5-bit](https://huggingface.co/netprojects/Qwen3.8-27B-LayerFit-MLX-2.5bit) (11.5 GB) |
| 24 GB | [3-bit](https://huggingface.co/netprojects/Qwen3.8-27B-LayerFit-MLX-3bit) (13.1 GB) |
| 32 GB | [4-bit](https://huggingface.co/netprojects/Qwen3.8-27B-LayerFit-MLX-4bit) (16.5 GB) |
| 48 GB and up | Uniform 6-bit / 8-bit builds from lmstudio-community or mlx-community are a good fit; LayerFit adds nothing there |

⚡ *Stretch:* close other apps, keep prompts to a few thousand tokens, and let the GPU use more memory with `sudo sysctl iogpu.wired_limit_mb=12800` (lasts until restart).

<!-- mac-picker:end -->

## Why LayerFit

A uniform quant gives every layer the same bits. LayerFit first measures how much each part of the model suffers at low bits (every layer's MLP, attention and linear-attention blocks, plus the embeddings and output head), then fills a fixed size budget where it helps most. Same size as a standard build, better output:

| Same size, same test | Top-1 agreement with bf16 | Error (KL) | GSM8K-50 |
|---|---|---|---|
| Standard 2-bit (same bits in every layer) | 68.2% | baseline | 30/50 |
| **LayerFit 2-bit** | **76.7%** | **46% lower** | **46/50** |

Our harness: language-model weights only, 32 passages × 512 tokens across English, other languages, code, tool calls and math, against our MLX bf16 conversion; GSM8K-50 thinking off. The exact recipe is in `layerfit-recipe.json`.

## This build

- Top-1 token agreement with bf16 on our 12-prompt card check: **84.8%** (KL 0.297)
- GSM8K-50, thinking off: 42/50
- Tool calls, coding 3/3, thinking 2/2: pass
- Peak memory with an 8K-token prompt: 11.7 GB
- Decode on an M5 Max (not a quiet machine, indicative): 46 tok/s

For Qwen's own benchmark results, see the [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B).

## Usage

```bash
pip install -U mlx-lm
mlx_lm.generate --model netprojects/Qwen3.8-27B-LayerFit-MLX-2bit --prompt "Hello"
```

## Licence

Apache-2.0, same as Qwen3.8-27B. See LICENSE and NOTICE.

Built by Paris Michalas, [NetProjects](https://netprojects.com). LayerFit™ is a NetProjects trademark. Not affiliated with Alibaba Cloud or the Qwen team.
