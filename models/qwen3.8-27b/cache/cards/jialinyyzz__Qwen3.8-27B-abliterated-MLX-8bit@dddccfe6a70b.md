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

# Qwen3.8-27B abliterated — MLX 8-bit

An **abliterated (decensored)** conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
to Apple **MLX** (8-bit), with the vision tower preserved. Refusal behavior has been suppressed via
directional ablation; **the model's built-in safety guardrails are largely removed.**

> ⚠️ **Experimental / research use only.** This model will attempt to answer requests that the
> original model would refuse. It ships **without** the base model's safety behavior. You are
> responsible for how you use it and for complying with the base model's Apache-2.0 license and
> applicable law.

## What this is

- **Base:** Qwen/Qwen3.8-27B (hybrid Gated-DeltaNet + attention VLM; 64 text layers + 27-layer vision tower)
- **Method:** [Heretic](https://github.com/p-e-w/heretic) MPOA (projected abliteration), selected from a
  300-trial Optuna search (variant `wide-177`)
- **Format:** MLX affine 8-bit (group size 64), 8.627 bits/weight, vision tower kept
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

Original vs. this abliteration (`lm-eval-harness` loglikelihood; GSM8K generative CoT):

| benchmark | original | w177 | Δ |
|---|---|---|---|
| ARC-Challenge | 0.570 | 0.567 | −0.003 |
| HellaSwag | 0.750 | 0.750 | 0.000 |
| Winogrande | 0.753 | 0.770 | +0.017 |
| OpenBookQA | 0.477 | 0.470 | −0.007 |
| GSM8K (CoT) | 0.76 | 0.78 | +0.02 |

**No measurable capability loss on these five benchmarks** (all deltas within ~±0.03 sampling noise).
This is **not** a claim of "lossless": **not evaluated** — code generation, agentic/tool-use
(SWE-bench etc.), GPQA, long-context, multilingual, and actual safety behavior; these may differ.
KL 0.056 means the harmless-prompt output distribution is close to the original, but only the
harmless first-token distribution and the benchmarks above were checked.

## Performance (Apple M5 Max, 128 GB)

Median of 3 trials, `mlx-dspark benchmark`, across chat/code/math prompts:

| config | tok/s | speedup |
|---|---|---|
| baseline | 17.8 | — |
| DSpark (`--caps auto`) | **26.9** | **1.51×** |

DSpark helps most on structured content — **1.84× on math, 1.65× on code, ~1.0× on chat**. Use
`--mode dspark`; `cap=auto` is optimal (larger fixed caps are slower on this hybrid because rejected
drafts must rebuild the Gated-DeltaNet recurrent state). DSpark auto-resolves its drafter
(`RadixArk/Qwen3.8-27B-DSpark`) from the model basename — no `--drafter` flag needed. Single lucky
prompts can hit ~2×, but the sustained median is ~1.5×. For maximum tok/s overall, the 4-bit variant
is faster in absolute terms (see that repo).

## Usage

**DSpark speculative decoding needs a second weight — the ~1.36B drafter.** `mlx-dspark`
auto-downloads it from `RadixArk/Qwen3.8-27B-DSpark` (no manual assembly), so the simple command
just works:
```bash
pip install mlx-dspark
mlx-dspark generate --model ./Qwen3.8-27B-abliterated-MLX-8bit --mode dspark \
  --prompt "..." --max-new-tokens 512
```
Fully self-contained (no upstream dependency) — point at the bundled drafter mirror
[`Qwen3.8-27B-DSpark-drafter`](../Qwen3.8-27B-DSpark-drafter):
```bash
mlx-dspark generate --model ./Qwen3.8-27B-abliterated-MLX-8bit \
  --drafter ./Qwen3.8-27B-DSpark-drafter --mode dspark --prompt "..."
```

Image / video (vision tower preserved, no drafter needed):
```bash
pip install mlx-vlm
python -m mlx_vlm generate --model ./Qwen3.8-27B-abliterated-MLX-8bit \
  --image photo.jpg --prompt "Describe this image."
```

## Provenance & license

- Derived from Qwen/Qwen3.8-27B under **Apache-2.0**; this derivative inherits Apache-2.0.
- Abliteration performed with Heretic (AGPL-3.0 tool; does not affect the model license).
- No additional training; weights edited by directional ablation only.
