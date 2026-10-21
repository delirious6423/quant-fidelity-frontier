---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
thumbnail: https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/hero.png
base_model:
- google/gemma-4-26B-A4B-it
base_model_relation: quantized
quantized_by: AtomicChat
pipeline_tag: text-generation
library_name: mlx
tags:
- atomic-chat
- gemma
- gemma4
- google
- mlx
- apple-silicon
- quantized
---

<center>

<div style="display:flex; justify-content:center; align-items:center; gap:2%; max-width:560px; margin:0 auto;">
<a href="https://atomic.chat" style="flex:0 1 auto; min-width:0;"><img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/pill_atomic_v3.png" alt="Atomic Chat" style="width:100%; height:auto; max-width:186px;"></a>
<a href="https://discord.gg/8wGSsvmg4V" style="flex:0 1 auto; min-width:0;"><img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/pill_discord_v3.png" alt="Join Discord" style="width:100%; height:auto; max-width:184px;"></a>
<a href="https://github.com/AtomicBot-ai/Atomic-Chat" style="flex:0 1 auto; min-width:0;"><img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/pill_github_v3.png" alt="GitHub" style="width:100%; height:auto; max-width:141px;"></a>
</div>

<br/>

<img src="https://huggingface.co/AtomicChat/gemma-4-26B-A4B-it-MLX-8bit/resolve/main/hero.png" alt="Gemma 4 26B A4B" style="width:100%; max-width:100%; height:auto; margin-bottom:0.6em;"/>

<div style="display:flex; justify-content:center; gap:0.5em;">
<a href="https://huggingface.co/google/gemma-4-26B-A4B-it"><strong>Base model: google/gemma-4-26B-A4B-it</strong></a>
</div>
</center>

**Gemma 4 26B A4B**, self-quantized to MLX by [Atomic Chat](https://atomic.chat). Built straight from Google's original weights with a per-tensor importance matrix, so this is not a repack of somebody else's files. Runs fully offline.

## Highlights

- **25.2B total / 3.8B active per token parameters**: the weights this repo quantizes.
- **Context length**: 256K tokens, as published by Google.
- **30 layers**: Mixture-of-Experts, hybrid sliding-window (1024) and global attention.
- **Modalities**: Text, Image.
- **Full imatrix ladder**: every quant is calibrated with an importance matrix.
- **Reasoning**: All models in the family are designed as highly capable reasoners, with configurable thinking modes.
- **Diverse & Efficient Architectures**: Offers Dense and Mixture-of-Experts (MoE) variants of different sizes for scalable deployment.

> [!NOTE]
> These MLXs are **self-quantized from the original weights**, not a repack. The importance matrix keeps low-bit quants closer to the full-precision model.

## Model Overview

| Property | Value |
|---|---|
| Base model | `google/gemma-4-26B-A4B-it` |
| Parameters | 25.2B total / 3.8B active per token |
| Layers | 30 |
| Experts | 128 routed (top-8) |
| Sliding window | 1024 tokens |
| Context length | 256K tokens |
| Vocabulary | 262K |
| Modalities | Text, Image |
| Architecture | Mixture-of-Experts, 128 experts (top-8), hybrid sliding-window (1024) and global attention, 16 attention heads over 8 KV heads, `Gemma4ForConditionalGeneration` |
| This repo | MLX weights |

## Benchmarks

| Benchmark | Score |
|---|---|
| MMLU Pro | 82.6% |
| AIME 2026 no tools | 88.3% |
| LiveCodeBench v6 | 77.1% |
| Codeforces ELO | 1718 |
| GPQA Diamond | 82.3% |
| Tau2 (average over 3) | 68.2% |
| HLE no tools | 8.7% |
| HLE with search | 17.2% |
| BigBench Extra Hard | 64.8% |
| MMMLU | 86.3% |
| MMMU Pro | 73.8% |
| OmniDocBench 1.5 (average edit distance, lower is better) | 0.149 |
| MATH-Vision | 82.4% |
| MedXPertQA MM | 58.1% |
| MRCR v2 8 needle 128k (average) | 44.1% |

Scores are Google's published results for the base `google/gemma-4-26B-A4B-it`, not our own measurements. Quantization preserves the large majority of this; `Q4_K_M` and up stay close to full precision.

## Get started

- **[Atomic Chat](https://atomic.chat):** search `AtomicChat/gemma-4-26B-A4B-it-MLX-8bit` and hit **Use this model**.
- **mlx-lm:** `mlx_lm.generate --model AtomicChat/gemma-4-26B-A4B-it-MLX-8bit --prompt "Hello" --max-tokens 512`
- **Server:** `mlx_lm.server --model AtomicChat/gemma-4-26B-A4B-it-MLX-8bit --port 8080`

## Best practices

| Parameter | Value |
|---|---|
| temperature | 1.0 |
| top_p | 0.95 |
| top_k | 64 |

Google's recommended sampling configuration for `google/gemma-4-26B-A4B-it`.

## How these were made

1. Download `google/gemma-4-26B-A4B-it` (original weights).
2. Convert and quantize with `mlx_lm.convert` on our pipeline.

## License

Original model by Google, released under the Apache 2.0 license. Full terms: [Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license). Quantized by Atomic Chat.
