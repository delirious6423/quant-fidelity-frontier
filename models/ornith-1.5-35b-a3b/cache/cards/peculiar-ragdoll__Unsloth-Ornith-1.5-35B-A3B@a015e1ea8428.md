---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - imatrix
  - unsloth-dynamic
  - quantized
  - qwen35moe
  - moe
  - ornith
---

<div align="center">
  <img src="unslothornith_banner.png" alt="UnslothOrnith — Ornith-1.5 35B-A3B, imatrix-calibrated, dynamically quantized" width="100%">
</div>

# UnslothOrnith-1.5-35B-A3B · GGUF

An imatrix-calibrated, **Unsloth-Dynamic-style** GGUF ladder for
[**Ornith-1.5-35B-A3B**](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — the model's own
weights and chat template, re-quantized with a per-tensor precision policy and a fresh importance
matrix. **The MTP (`nextn`) block is removed from every tier** — Ornith-1.5 ships it untrained, so it
could only add size. [Why](#why-no-mtp).

> ### Not affiliated with Unsloth
> This is an independent, best-effort community reproduction of Unsloth's
> [Dynamic GGUF](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs) method — **not** an
> official Unsloth release, and not endorsed by or connected to Unsloth in any way. Unsloth had not
> published their own Ornith-1.5 quants at the time of writing and a lot of people wanted them, so
> we published these while we wait. If and when Unsloth ship an official build, use theirs. "Unsloth"
> is used here only to name the *method* these quants reproduce.

## Method

Two ingredients, both applied to Ornith's own `BF16`:

1. **Unsloth-Dynamic per-tensor policy.** Instead of one bit-width for the whole model, each tensor
   gets a type chosen by sensitivity — more bits where they matter (embeddings, attention,
   early/late layers), fewer where they don't. We transfer the published per-tensor recipe from
   Unsloth's [`Qwen3.6-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF) (same
   `qwen35moe` architecture) tier-for-tier.
2. **A fresh importance matrix.** Computed on Ornith's own `BF16` over an interleaved multilingual +
   code calibration corpus, so the quantizer knows which weights to protect for *this* model rather
   than inheriting someone else's statistics.

The chat template is **Ornith's own, unchanged** — no custom system prompt, so these behave exactly
like the base model, just smaller.

## Tiers

| Tier | Approx size | Notes |
|---|---|---|
| `UD-Q2_K_XL`  | ~12 GB | 16 GB VRAM, or 6 GB VRAM + partial offload |
| `UD-IQ3_XXS`  | ~13 GB | higher-quality 16 GB-VRAM pick (better than Q2 at similar size) |
| `UD-Q3_K_XL`  | ~16 GB | |
| `UD-IQ4_XS`   | ~17 GB | full-context option for 24 GB VRAM (leaves the most room for context) |
| `UD-Q4_K_S`   | ~20 GB | 24 GB VRAM friendly |
| `UD-Q4_K_XL`  | ~22 GB | recommended balance |
| `UD-Q5_K_XL`  | ~25 GB | |
| `UD-Q6_K_XL`  | ~30 GB | near-lossless |
| `UD-Q8_K_XL`  | ~37 GB | reference |

All tiers are published.

## Download

```bash
hf download peculiar-ragdoll/Unsloth-Ornith-1.5-35B-A3B \
  Unsloth-Ornith-1.5-35B-A3B-UD-Q4_K_XL.gguf --local-dir .
```

Then run with any llama.cpp-based stack (llama.cpp, LM Studio, Ollama, etc.).

## Why no MTP

Ornith-1.5 carries an MTP (`nextn`) block and earlier versions of this repo shipped it. It is not
trained. Every weight matrix in `blk.40` sits at standard deviation 0.020, kurtosis 3.00, and a
largest value of 5 standard deviations — which is exactly a fresh `N(0, 0.02)` initialization. Every
trained layer in the same file has kurtosis 4 to 39 and outliers past 14 standard deviations, because
training grows heavy tails. The same measurement on Qwen3.6-27B's and Qwen3.8-27B's MTP heads gives
kurtosis near 900 and outliers past 60 — those are trained; this one never was.

Drafts from random weights are accepted at chance, so the block was 2.4% of every file doing no work.
Removing it changes no output and saves roughly 0.3 to 0.9 GB per tier. Nothing else was touched: the
remaining weights are byte-identical to the previous upload, not re-quantized.

## Credits

- [**ornith-ai**](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — the Ornith-1.5-35B-A3B model and weights (MIT).
- [**Unsloth**](https://huggingface.co/unsloth) — the Dynamic GGUF quantization method this reproduces.
- [**llama.cpp**](https://github.com/ggml-org/llama.cpp) — `llama-quantize` / `llama-imatrix`.

MIT, inheriting Ornith-1.5's license.
