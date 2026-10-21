---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
language: [en, ko]
tags: [gptq, w4a16, compressed-tensors, quantized, vllm, edge]
pipeline_tag: text-generation
base_model_relation: quantized
library_name: transformers
---

# Qwen3.8-27B-W4A16-GPTQ

**18.59 GB.** The smallest checkpoint we have of this model, and on the benchmarks we ran it is not
distinguishable from bf16. It exists because a 30 GB NVFP4 build does not fit a 32 GB card and this one
does, with room left for KV cache.

We also publish the thing most quantization cards leave out: **we measured how far two builds of this
exact recipe land from each other, and on one axis it is very far.** Read the noise-floor section before
you compare this checkpoint against anything.

## What it is

GPTQ W4A16 via `llmcompressor`, `pack-quantized`, group size 128, symmetric int4, `actorder: static`,
`dampening_frac: 0.01`. Calibration: 1,024 samples at 2,048 tokens. Quantization took ~38 minutes on a
single GPU.

Ignored (left at original precision): `re:.*vision.*`, `re:.*visual.*`, `lm_head`, `re:.*embed_tokens.*`.
This is a hybrid VL model — 64 layers, 16 full-attention and 48 linear-attention (GDN) — so the naive
"27B at 4-bit ≈ 13.5 GB" arithmetic does not hold. The vision tower, embeddings and recurrent state stay
wide, which is why the file is 18.59 GB rather than ~14.

## Quality

**GSM8K, full 1,319-item set, greedy, vLLM 0.27.1 + lm-eval 0.4.12. All arms in one serving session,
identical serve config, model path the only difference.**

| | en strict | en flexible | ko strict | ko flexible |
|---|---|---|---|---|
| bf16 | 0.5421 | 0.5481 | 0.6171 | 0.6353 |
| **this build** | 0.5155 | 0.5603 | 0.5785 | 0.6505 |

Against bf16 on flexible-extract this build is **+2.28pp (en)** and **+1.52pp (ko)** — that is, slightly
ahead, and well inside noise either way. Against a third-party INT4 build of the same model it is
+0.76pp / +0.08pp. **On these tasks the three are not distinguishable.**

⛔ **Judge on `flexible-extract`, not `strict-match`.** strict-match requires the answer wrapped in a
specific format; 4-bit builds of this model (ours and third-party alike) often answer correctly without
that wrapper and get scored wrong. We had four blind judges read 40 disputed cases: **35 of 40 were
correct answers rejected on formatting**.

## ⛔ Rebuild noise floor — read this before comparing checkpoints

We ran **the same recipe a second time** — same model, same settings, same calibration — and evaluated
both builds in one session. They did not land in the same place:

| Task | Metric | Gap between two identical-recipe builds |
|---|---|---|
| GSM8K (en) | strict | **3.56pp** |
| GSM8K (en) | flexible | 2.27pp |
| GSM8K (ko) | strict / flexible | 2.13pp / 1.37pp |
| **RAG, absent-information** | **fabrication rate** | **27.08pp (7.5σ)** |

The last row is the one that matters. On questions whose answer is **not** in the retrieved document,
this build fabricated an answer 9.17% of the time (n=240). The second build of the same recipe did so
**36.25%** of the time. A third-party INT4 build scored 1.25%; bf16 scored 3.33%.

Two consequences, stated plainly:

1. **A difference under 2× the floor for that metric is not attributable to the recipe.** That rule
   invalidates several comparisons we ourselves had drawn, including some in our sibling cards.
2. **Our GPTQ pipeline is not reproducible on abstention behaviour.** Run it twice and you cannot predict
   whether you get 9% or 36%. We do not currently know why. If refusing to answer when the document is
   silent is load-bearing for you, **use a frozen third-party artifact instead of rebuilding this recipe** —
   or evaluate the specific file you intend to ship.

And this failure is invisible to accuracy metrics: the 36% build had the **highest** extraction accuracy
of the four arms and an identical grounded-answer score. An accuracy-only gate ships it.

Full data: [`ThakiCloud/quantization-rebuild-noise-floor`](https://huggingface.co/datasets/ThakiCloud/quantization-rebuild-noise-floor).

## Throughput — this is the slow one, and that is the trade

Same run, one B200, identical serve config, 2,048 in / 256 out:

| build | ceiling tok/s | TTFT | size |
|---|---|---|---|
| NVFP4 (`-NVFP4-GPTQ-txt`) | 3,597.9 | 2.09 s | 30.14 GB |
| **this build (W4A16, Marlin)** | **1,022.8** | 10.66 s | **18.59 GB** |

**3.52× slower at the ceiling and 5.1× worse TTFT.** W4A16 goes through Marlin dequantization; NVFP4 runs
a native Blackwell kernel. So the choice is not "which is better" but which constraint binds:

- **32 GB card** → this build. NVFP4's 30.14 GB leaves no room for KV cache.
- **96 GB card** → take the NVFP4 build; the memory you saved buys nothing there.

On long context the ordering holds. KV on this model is **64 KiB/token** (only the 16 full-attention
layers contribute), and `--kv-cache-dtype fp8` halves that for **1.95× the KV pool** at no measured
throughput or quality cost — worth turning on before you shop for a bigger card.

## Usage

```bash
vllm serve ThakiCloud/Qwen3.8-27B-W4A16-GPTQ \
  --max-model-len 131072 --max-num-seqs 256 \
  --kv-cache-dtype fp8 --mamba-cache-mode align
```

`--mamba-cache-mode align` is required — this is a hybrid SSM model. Native context is 262,144.

## Scope of what we measured

GSM8K (en/ko, full set), a Korean procurement-document RAG suite (extraction / grounded / absent, n=240),
throughput and TTFT on one B200. One model, one method, one hardware generation. We did **not** measure:
MMLU-class breadth, code, long-context retrieval quality, or behaviour on other GPU families.
