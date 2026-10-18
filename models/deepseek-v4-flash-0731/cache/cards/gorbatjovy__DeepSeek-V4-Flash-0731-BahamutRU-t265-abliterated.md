---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- deepseek
- deepseek-v4
- abliterated
- uncensored
- heretic
- fp8
---

# DeepSeek-V4-Flash-0731-BahamutRU-t265-abliterated

`deepseek-ai/DeepSeek-V4-Flash-0731` with the refusal direction ablated, using the rank-1
adapter from
[`BahamutRU/DeepSeek-V4-Flash-0731-heretic-abliterated-v2-GGUF-lora`](https://huggingface.co/BahamutRU/DeepSeek-V4-Flash-0731-heretic-abliterated-v2-GGUF-lora)
baked directly into the FP8 weights.

**There is no LoRA at runtime.** The edit is merged into the tensors; load it like any other
checkpoint.

## What was changed

57 target tensors (114 counting block scales):

* `layers.{11..42}.attn.wo_b` — attention output projections
* `layers.{18..42}.ffn.shared_experts.w2` — shared-expert down-projections

Everything else — all 72,203 other indexed tensors, including every routed expert —
is byte-identical to the official DeepSeek release. The 48 original shards are unmodified;
the edited tensors live in a single overlay file that the index redirects to.

## Method

Directional ablation is inherently rank-1: `W <- W - lam * r (r^T W)` is an outer product, so
the GGUF adapter is the edit in closed form. Verified against the base weights before
applying — `cos(lora_a, r^T W)` is -0.995 / -0.999 / -0.943 at layers 30 / 20 / 42, with
implied lambda 4.81 / 2.67 / 2.39. All 32 attention layers share one direction
(min |cos| 0.994), matching the adapter's declared global direction scope.

The delta is applied verbatim rather than re-derived as a projection, since the cosine is
near but not exactly -1 — reproducing what the adapter author measured rather than an
approximation of it.

Dequantise FP8 e4m3 (ue8m0 128x128 block scales) -> add `lora_b (x) lora_a` -> requantise,
raising a block's exponent where the larger delta would otherwise clip. Zero elements were
clamped; peak overshoot 1.095x.

Lambda ranges 0.72–4.88 across layers, peaking at layer 31. Above 2.0 the refusal component
is inverted rather than merely removed.

## Not included

The adapter also ablates routed-expert down-projections (256 experts x 31 layers). Those are
FP4 and were left untouched here, so this is a partial application of `t265`.

## Evaluation

Measured 2026-08-16/17 on 2x DGX Spark (vLLM, TP=2, FP8 KV cache) against the unabliterated
base, using identical prompts, seeds and serving configuration.

```
benchmark                            base    this model
--------------------------------------------------------
capability (higher is better)
  IFEval  (prompt strict)           83.4%        82.6%
  HumanEval+                        87.8%        88.4%
  MBPP+                             73.3%        73.3%
  MMLU-Pro                          78.9%        77.9%
  tool calling: correct tool       100.0%       100.0%
  tool calling: correct args       100.0%       100.0%
  tool calling: false positives      0.0%         0.0%

behavioural drift from base (lower is better)
  mean first-token KL                  —       0.1117
  median first-token KL                —       0.0120
  p90 first-token KL                   —       0.3761
  top-1 token agreement                —        91.2%

long-form generation
  mean distinct-3                 0.9834       0.9833
  creative type-token ratio       0.3392       0.3583
```

**Capability is unaffected.** Instruction following, code generation, knowledge and tool
calling all land within noise of the base model.

**Drift from base is small.** First-token KL divergence on harmless prompts is the most
direct measure of what an abliteration costs — it is also what heretic-gguf optimises
against, jointly with refusal rate. This checkpoint measures 0.1117 mean / 0.0120 median and
still picks the base model's most likely first token on 91.2% of harmless prompts.

**Long-form generation is intact**, which matters for extended chat: lexical diversity and
repetition behaviour match base, with no degeneration despite λ reaching 4.88.

### Caveats

* **These KL values are not comparable to the adapter author's published 0.065.** Different
  prompt set (80 mundane factual, coding and creative prompts), different top-k handling.
  Absolute values only mean something relative to the same measurement on the same set.
* **MMLU-Pro is underpowered** here: 280 samples (20 × 14 subtasks), ±2.4%, so it can only
  detect gaps of roughly 5 points.
* **HumanEval and MBPP are contaminated.** Valid as a regression signal against the same
  base, not as capability claims.
* **Tool calling used 24 samples**, of which 3 are negative cases (a question needing no
  tool). Enough to show nothing is obviously broken; not enough to characterise the tail.
* **KL covers first tokens only** — a sensitive early indicator, not a full account of drift.
* **All runs had thinking disabled**, because third-party harnesses cannot parse
  `reasoning_content` or budget tokens for it. Both checkpoints were measured identically,
  but these are not thinking-mode scores.
* Refusal rate is **not** reported here. The adapter author measured 12/140 on their harmful
  set; measuring it properly needs a prompt set this evaluation did not include.

## Credit

* Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT)
* Adapter: [`BahamutRU/DeepSeek-V4-Flash-0731-heretic-abliterated-v2-GGUF-lora`](https://huggingface.co/BahamutRU/DeepSeek-V4-Flash-0731-heretic-abliterated-v2-GGUF-lora) (MIT),
  produced with heretic-gguf; reported 12 refusals on 140 harmful prompts at
  KL 0.065 on harmless prompts
* Ablation method: Heretic, by Philipp Emanuel Weidmann

## Warning

Refusal behaviour has been deliberately removed. This model will attempt whatever it is
asked. Safety judgement is entirely the operator's responsibility.
