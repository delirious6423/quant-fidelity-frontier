---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language: en
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - qwen3_5
  - abliterated
  - uncensored
  - heretic
  - vision-language
---

# Qwen3.8-27B abliterated — MLX 4-bit

An **abliterated (decensored)** conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
to Apple **MLX** (4-bit), with the vision tower preserved. Refusal behavior has been suppressed via
directional ablation; **the model's built-in safety guardrails are largely removed.**

> ⚠️ **Experimental / research use only.** This model will attempt to answer requests that the
> original model would refuse. It ships **without** the base model's safety behavior. You are
> responsible for how you use it and for complying with the base model's Apache-2.0 license and
> applicable law.

## What this is

- **Base:** Qwen/Qwen3.8-27B (hybrid Gated-DeltaNet + attention VLM; 64 text layers + 27-layer vision tower)
- **Method:** [Heretic](https://github.com/p-e-w/heretic) MPOA (projected abliteration), selected from a
  300-trial Optuna search (variant `wide-177`)
- **Format:** MLX affine 4-bit (group size 64), 4.695 bits/weight, vision tower kept (~15 GB on disk,
  ~19 GB peak RAM at generation)
- **One file, two uses:** runs under `mlx-vlm` (image/video understanding) **and** under `mlx-dspark`
  (text + DSpark speculative decoding). Vision tensors (333 keys) are on disk; the text/DSpark path
  ignores them at load.

## Abliteration recipe (`wide-177`)

| parameter | value |
|---|---|
| refusal direction | single direction from layer ≈28 (`direction_index` 28.46) |
| `attn.o_proj` | gentle: 0.83→0.64, layers 6–63 (attention pathway barely touched) |
| `mlp.down_proj` | strong & uniform: 1.57→1.52, **all 64 layers** |
| KL divergence (orig ‖ abliterated) | **0.056** |

Refusals are removed mainly through the **MLP pathway across every layer** while the **attention
pathway is left nearly intact** — which is what preserves reasoning while suppressing refusals.

## Refusal reduction

Heretic keyword-refusal scorer on `mlabonne/harmful_behaviors` test[:100]:

| | refusals / 100 |
|---|---|
| base model | 98 |
| this model (w177) | **11** |

> This is an automated **keyword** metric with known false positives (a compliant answer containing
> "illegal"/"harmful" is scored as a refusal) and false negatives (a refusal phrased without those
> words is missed). Treat it as an indicator, not ground truth.

## Capability evaluation

Original vs. this abliteration (`lm-eval-harness` loglikelihood; GSM8K generative CoT). Measured on the
bf16 master; the 4-bit quantization adds its own small, separate quantization error on top.

| benchmark | original | w177 (bf16) | Δ |
|---|---|---|---|
| ARC-Challenge | 0.570 | 0.567 | −0.003 |
| HellaSwag | 0.750 | 0.750 | 0.000 |
| Winogrande | 0.753 | 0.770 | +0.017 |
| OpenBookQA | 0.477 | 0.470 | −0.007 |
| GSM8K (CoT) | 0.76 | 0.78 | +0.02 |

**No measurable capability loss from abliteration on these five benchmarks** (deltas within ~±0.03
sampling noise). This is **not** a claim of "lossless": **not evaluated** — code, agentic/tool-use,
GPQA, long-context, multilingual, actual safety behavior; and 4-bit quantization itself trades some
quality for size/speed. For maximum fidelity use the 8-bit variant.

## Performance (Apple M5 Max, 128 GB)

Median of 3 trials, `mlx-dspark benchmark`, across chat/code/math prompts. ~19 GB peak RAM.

| config | tok/s | speedup |
|---|---|---|
| baseline | 32.0 | — |
| DSpark (`--caps auto`) | **36.0** | **1.12×** |

The 4-bit model is already fast and less memory-bandwidth-bound, so DSpark's gain is modest
(~1.25× on code/math, slightly slower on chat). This variant is the **fastest in absolute terms** —
about 1.3–1.8× the 8-bit throughput — at a small quality cost vs. 8-bit. Use `--mode dspark`
(`cap=auto`) for code/math; for chat, baseline is about as fast. DSpark auto-resolves its drafter
(`RadixArk/Qwen3.8-27B-DSpark`) from the basename — no `--drafter` flag needed.

## Usage

**DSpark speculative decoding needs a second weight — the ~1.36B drafter.** `mlx-dspark`
auto-downloads it from `RadixArk/Qwen3.8-27B-DSpark` (no manual assembly), so the simple command
just works:
```bash
pip install mlx-dspark
mlx-dspark generate --model ./Qwen3.8-27B-abliterated-MLX-4bit --mode dspark \
  --prompt "..." --max-new-tokens 512
```
Fully self-contained (no upstream dependency) — point at the bundled drafter mirror
[`Qwen3.8-27B-DSpark-drafter`](../Qwen3.8-27B-DSpark-drafter):
```bash
mlx-dspark generate --model ./Qwen3.8-27B-abliterated-MLX-4bit \
  --drafter ./Qwen3.8-27B-DSpark-drafter --mode dspark --prompt "..."
```

Image / video (vision tower preserved, no drafter needed):
```bash
pip install mlx-vlm
python -m mlx_vlm generate --model ./Qwen3.8-27B-abliterated-MLX-4bit \
  --image photo.jpg --prompt "Describe this image."
```

## Provenance & license

- Derived from Qwen/Qwen3.8-27B under **Apache-2.0**; this derivative inherits Apache-2.0.
- Abliteration performed with Heretic (AGPL-3.0 tool; does not affect the model license).
- No additional training; weights edited by directional ablation only.
