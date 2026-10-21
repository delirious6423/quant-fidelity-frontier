---
license: apache-2.0
language:
- en
- zh
base_model:
- Qwen/Qwen3.8-27B
tags:
- qwen
- gguf
- anti-sycophancy
- reflect
- fine-tune
---

# Qwen3.8-27B-Reflect

The Reflect recipe carried onto **Qwen3.8-27B** — same principle as
[Qwen3.6-27B-Reflect](https://huggingface.co/annnnnnnd/Qwen3.6-27B-Reflect): a small,
aggressively curated dataset used to reshape *how* the model talks, not to add knowledge.

This is a **voice tune**. The training data contains no reasoning traces at all — every
example is an answer, so the edit lands entirely on the answer channel and leaves the
thinking channel untouched. That is exactly what the benchmarks show: with thinking off,
the two models diverge; with thinking on, they converge almost everywhere.

The headline: **it does what the base model does, in roughly half the tokens** — and it's
less annoying to talk to.

## Why we think this is worth fixing

Something specific has gone wrong with how instruction-tuned assistants talk, and it is
not simply that they are wordy.

Measured on the same 200 prompts, base and Reflect hedge by the **same total volume**.
What differs is *placement*: base opens with the caveat 15 times out of 70, Reflect 6.
Identical content, different order — and the order is most of what makes a model tiring
to use.

Then the harder version of the same problem. Across 200 safety prompts, base produced
**68 responses our scorer could not classify** as either refusal or compliance, against
Reflect's 23. That is not a truncation artifact — we re-ran the whole set with a 2.7×
larger token budget and the number did not move by one. They are genuine half-refusals:
the model declines and complies in the same breath.

That is the part worth fixing, for three reasons.

1. **It buys no safety.** Refusal on genuinely harmful prompts is **94% for both models**.
   Every one of those extra ambiguous responses was pure overhead — the hedging caught
   nothing that a flat refusal missed.
2. **It cannot be audited.** If an evaluation harness cannot tell whether the model
   complied, then neither can a safety filter, a downstream parser, or the person reading
   it. Ambiguity is not caution. It is cost deferred onto whoever reads it next.
3. **It is expensive.** 43% more tokens and 1.65× the wall clock, for the same answers at
   slightly lower accuracy. You pay that on every request, in latency and in compute.

The failure generalises past prose. On BFCL `irrelevance` — where the correct move is to
call no tool at all — base emits **67 false tool calls of 240** against Reflect's 54. A
model that will not commit in language does not commit in action either.

We think this is convergent rather than anyone's specific mistake: preference training
rewards answers that pre-empt the objection a rater might raise, and *"It is important to
note"* is the cheapest way to buy that. Nearly every assistant trained this way ends up
with the same tic.

**What we are not claiming.** Refuse-and-redirect is often the better answer — *"I won't
do that, but here's what I can do"* beats a dead end, and base is genuinely good at it.
Hedging is correct when the uncertainty is real. The goal is not a blunt model. It is one
that puts the answer first and the qualification after, and that says no clearly when it
means no.

## What changed from Qwen3.6-Reflect

The 3.6 card described the edit as *"say less, think more"* — concision in the answer
channel, expansion in the reasoning channel. On 3.8 the concision effect is much larger
and now shows up in wall-clock as well as token counts:

* **~43% fewer output tokens at equal-or-better accuracy** (GSM8K: 139 vs 244 mean
  tokens, 96.00% vs 94.75%) — up from ~12% on 3.6.
* **1.65× faster wall clock** on the same work, same hardware, same server flags.
* **Less preamble.** The base model opens with *"It is important to note…"* on 15.6% of
  responses; Reflect does it on 6.2%. Base tends to qualify before answering — Reflect
  answers, then qualifies if it matters. Same total hedging volume, different placement,
  and the placement is most of what makes an assistant tiring to use.
* **Knows when not to act.** +5.0 points on BFCL `irrelevance` — declining to call a tool
  when none of the offered tools fit (54 false calls of 240, vs base's 67).
* **More decisive in both directions.** On refusals it says no outright (43% flat refusals
  vs base's 28%) instead of the base's refuse-and-redirect (66% vs 51%). Fewer soft
  half-answers.

The 3.6 tradeoff still applies: **a format-following tax without thinking** (IFEval
-5.7), which largely disappears once thinking is on (-0.5).

## Training Methodology

* **Base**: Qwen3.8-27B
* **Method**: QLoRA, 4-bit base — **SFT only, no DPO stage**
* **LoRA**: r32 / a32 (1:1 alpha-to-rank)
* **Learning rate**: 1e-4, **constant schedule**
* **Epochs**: 1
* **Data**: 4,911 curated examples / 4,967 user-assistant exchanges — 4,865 single-turn,
  41 four-turn, 5 longer. No system prompts.

One pass of supervised fine-tuning over ~5k examples, at 4-bit. That is the entire
intervention — worth keeping in mind when reading the results below.

**Answer-channel only.** Not one assistant message in the set contains a `<think>` block.
The model was never shown a reasoning trace, so nothing in training touches how it
reasons — only what comes out afterwards. Every result in this card follows from that:

| | Effect |
|---|---|
| Thinking **off** | Answer channel is the whole output → the tune shows up in full |
| Thinking **on** | Untouched reasoning dominates → base and Reflect converge |

**The voice is in the length and the opening.** Target answers are short and tightly
distributed (p25 963, median 1,414, p90 2,180 chars) against short prompts (median 114),
and they almost never open with a hedge:

| Opens with "It is important to note…" / similar | Rate |
|---|---:|
| Training data | **0.08%** (4 of 4,967) |
| Base Qwen3.8-27B, measured live | 15.6% |
| Reflect, measured live | **6.2%** |

The fine-tune moved roughly 60% of the way from base behaviour toward the training
distribution. This is the mechanism behind both headline numbers — the ~43% token
reduction and the "less preamble" effect are the same edit measured two ways.

It also explains the IFEval tax: a voice distilled toward *"answer directly, at moderate
length"* rounds off mechanical sub-constraints (exact word counts, required literal
keywords, all-lowercase) when it cannot deliberate first. Turn thinking on and the
untouched reasoning channel handles them again.

## Files

| File | Size | MTP | imatrix | Notes |
|---|---:|:---:|:---:|---|
| `Qwen3.8-27B-Reflect-UD-Q8_0.gguf` | 27.12 GB | ✅ | ✅ | Everything Q8_0, F32 SSM gating. |
| `Qwen3.8-27B-Reflect-UD-Q8FFN.gguf` | 25.23 GB | ✅ | ✅ | FFN at Q8_0; attention, SSM and vocabulary stay Q6_K. |
| `Qwen3.8-27B-Reflect-UD-Q6_K_L.gguf` | 21.89 GB | ✅ | ✅ | Q6_K trunk, `token_embd` + `output` lifted to Q8_0. |
| `Qwen3.8-27B-Reflect-UD-Q6_K.gguf` | 21.31 GB | ✅ | ✅ | **Default.** All benchmarks below were run on this build. |
| `Qwen3.8-27B-Reflect-UD-Q5_K_XL.gguf` | 19.55 GB | ✅ | ✅ | Best fit for 24 GB cards — ~115k ctx vs Q6_K's ~56k. |
| `Qwen3.8-27B-Reflect-UD-Q4_K_XL.gguf` | 16.39 GB | ✅ | ✅ | VRAM-friendly. |

**`UD-Q8FFN`** is the odd one. FFN is 63.6% of this model's parameters, attention and SSM
together 26.9% — so it puts the bulk of the weights at Q8_0 while leaving the
sequence-mixing machinery (and the vocabulary) at Q6_K. The bet is that stored knowledge
is what benefits from precision and that reasoning tolerates Q6. It costs 3.9 GB over
`UD-Q6_K` and saves 1.9 GB against full Q8_0. Untested against the others — see the note
below.

**Take `UD-Q6_K`** unless VRAM is tight, then `UD-Q4_K_XL`. The UD builds use Unsloth's
importance matrix and per-layer dynamic bit allocation — Q8_0 on the SSM gating, Q5_K
through the trunk, IQ4_XS where the layer tolerates it. They are smaller *and* better
than a flat quant at the same nominal width.

**`UD-Q6_K_L`** is `UD-Q6_K` with the vocabulary ends at higher precision: `token_embd`
and `output.weight` at Q8_0 instead of Q6_K, +0.57 GB. Everything between them is
byte-identical to `UD-Q6_K`, so it reasons the same — the extra precision goes to the
248,320-token embedding and unembedding, where quantization noise is most likely to flip
a close call between similar tokens (rare words, digits, code identifiers, cross-lingual
near-synonyms).

**Every benchmark in this card was measured on `UD-Q6_K`.** The larger builds
(`UD-Q6_K_L`, `UD-Q8FFN`, `UD-Q8_0`) are principled bets about where precision matters,
not measured wins — we have not benchmarked a quality delta between any of them. They load
and behave correctly; that is all that is claimed. Take `UD-Q6_K` if you want the build the
numbers describe.

Note our `_L` is not Unsloth's `_L` — theirs keeps `token_embd` at Q6_K and spends its
extra size on the trunk instead. Don't compare the two by filename.

**On MTP:** Qwen3.8 ships a multi-token-prediction head (`blk.64` / `nextn`). `llama.cpp`
does **not** use it — it logs `unused tensor blk.64.* -- ignoring` and drops it, ~350 MB
of dead weight. The `-noMTP` builds strip it. Engines that implement MTP as
self-speculative decoding (vLLM, SGLang) can use it, so the UD builds keep it.

## Prompt format

Standard Qwen ChatML. Thinking is controlled by the assistant prefill — close the block
immediately (`<think>\n\n</think>\n\n`) for no-think. `reasoning_effort` is supported via
the chat template: `xhigh` / `medium` / `low` (`high` aliases to `xhigh`).

## Benchmarks

All figures are **Reflect vs base Qwen3.8-27B at byte-identical quantization** — same
Unsloth imatrix, same tensor split, zero differing tensors of 866 — on the same hardware,
prompts, sampling and server flags. The deltas are the fine-tune.

Token columns are **total output tokens across the whole run**, counted with the Qwen
tokenizer.

### 1. No-think accuracy (full sets, thinking disabled for both)

| Benchmark | N | Base | Reflect | Delta | Base tokens | Reflect tokens |
|---|-:|-:|-:|-:|-:|-:|
| MMLU | 1000 | 83.90% | **84.60%** | +0.70 | 1,000 | 1,000 |
| GSM8K | 400 | 94.75% | **96.00%** | +1.25 | 97,739 | **56,907** |
| HumanEval | 164 | 93.90% | 93.29% | -0.61 | 38,167 | **32,571** |
| IFEval | 192 | **80.73%** | 75.00% | -5.73 | 65,727 | 67,166 |
| ARC Challenge | 400 | 96.75% | 96.50% | -0.25 | 400 | 400 |
| TruthfulQA | 200 | 83.50% | **86.00%** | +2.50 | 200 | 202 |
| EQ-Bench | 171 | 81.49% | 81.09% | -0.40 | 5,203 | 5,161 |
| **Total** | | | | | **208,476** | **163,447** |

**Reading this honestly:**

* MMLU / ARC / EQ-Bench / HumanEval deltas are within noise at these sample sizes.
* **GSM8K is the real result**: +1.25 points on **42% fewer tokens** (56,907 vs 97,739).
  That is the whole thesis of this release in one row.
* **IFEval -5.73** is the same voice-vs-format tax the 3.6 card documented — mechanical
  sub-constraints (exact word counts, required literal keywords, all-lowercase) get
  rounded off by the distilled voice. It is not truncation; responses are full length.
  With thinking on the gap closes to -0.52.
* MMLU / ARC / TruthfulQA are 16-token-capped single-letter answers, so their token
  columns are structural, not a finding.

### 2. With thinking enabled (`reasoning_effort=xhigh`)

| Benchmark | N | Base | Reflect | Delta |
|---|-:|-:|-:|-:|
| **GPQA-Diamond** | 198 | 82.32% | **84.85%** | **+2.53** |
| MMLU | 1000 | 92.20% | 92.20% | 0.00 |
| GSM8K | 400 | 97.75% | 97.50% | -0.25 |
| HumanEval | 164 | 96.95% | **97.56%** | +0.61 |
| IFEval | 192 | 89.06% | 88.54% | -0.52 |
| ARC Challenge | 400 | 98.50% | 98.00% | -0.50 |
| TruthfulQA | 200 | 91.00% | **91.50%** | +0.50 |

**Thinking is worth more than the model choice** — roughly +4.8 points of suite average
for *both* models, and it erases nearly every no-think difference between them, including
most of the IFEval gap. Quote any figure here with its thinking budget.

**GPQA-Diamond is the exception**, and the only benchmark where the two separate
meaningfully with thinking on:

| GPQA-Diamond (198) | Base | Reflect |
|---|-:|-:|
| No-think | 75.25% | 75.25% |
| **+ thinking** | 82.32% | **84.85%** |
| Physics | 93.02% | **97.67%** |
| Chemistry | 74.19% | 74.19% |
| Biology | 73.68% | **78.95%** |
| No-think total tokens | 650,969 | **519,331** |
| Thinking total tokens (49 retried) | 998,659 | **853,981** |

Identical no-think (149/198 each, but failing *different* questions — 131 both correct,
18 unique to each). Given room to reason, Reflect converts 23 of its 49 failures against
base's 17. Head-to-head after thinking: 10 questions Reflect-only, 5 base-only.

> **These GPQA numbers will not match third-party leaderboards, and are not meant to.**
> Two deliberate deviations from the standard protocol:
>
> 1. **Two-phase, not single-pass.** Standard GPQA runs every question once with thinking
>    on. We run a no-think pass first, then re-run *only the failures* with thinking, and
>    merge. The merged score is therefore "best of two attempts under different settings",
>    which is not comparable to a single-pass figure.
> 2. **30k thinking budget.** Both models run past it on the hardest items — 31% of
>    Reflect's recovery attempts and 43% of base's hit the cap without closing `</think>`.
>    Those rows are scored on whatever was salvageable, so **both numbers are depressed**
>    relative to an uncapped run.
>
> The comparison between the two columns is sound — identical protocol, identical budget,
> same hardware, and every scored-correct answer verified to carry a real `\boxed{}`. The
> absolute values are ours, not the field's. Do not rank these against published GPQA
> scores.

### 3. Function calling — BFCL v4

| Category | N | Base | Reflect | Delta | Base tokens | Reflect tokens |
|---|-:|-:|-:|-:|-:|-:|
| `multiple` | 199 | 95.50% | **96.50%** | +1.00 | 8,128 | **7,816** |
| `irrelevance` | 239 | 72.50% | **77.50%** | **+5.00** | 40,619 | **32,479** |

`multiple` is a tie (2 questions). **`irrelevance` is the real one** — it measures
declining to call a function when none of the provided tools fit. Base emits 67 false
tool calls of 240; Reflect 54. Both models' failures are otherwise the same shape
(`wrong_count`, `value_error:string` in near-identical proportions), so this is a
disposition change, not a capability one.

Concision does **not** damage structured output — Reflect's tool calls are marginally
shorter *and* marginally more accurate.

### 4. Safety

| | N | Base | Reflect | Base tokens | Reflect tokens |
|---|-:|-:|-:|-:|-:|
| JailbreakBench — harmful refusal | 100 | 94% | 94% | 136,112 | **99,079** |
| JailbreakBench — benign compliance | 100 | 71% | **75%** | *(same run)* | *(same run)* |
| Do-Not-Answer — refusal | 200 | 55.5% | 57.0% | 61,983 | **43,416** |

**Refusal on genuinely harmful requests is identical to base.** The fine-tune does not
erode guardrails. What changes is disposition — Reflect commits, base hedges:

| JailbreakBench, harmful prompts | Base | Reflect |
|---|-:|-:|
| Flat refusal | 28% | **43%** |
| Refuse-and-redirect | **66%** | 51% |
| Mixed-signal / ambiguous responses (of 200) | **68** | **23** |

Base produces three times as many responses that read as half-refusal, half-compliance.
That is harder to audit and, in practice, more irritating to use.

**Reading this honestly:** safety scoring here is a refusal-string heuristic, not an LLM
judge. Hedged responses are genuinely hard to classify — treat these as indicative. The
harmful-refusal parity is the robust number (refusals declare themselves in the opening
line); the benign-compliance delta is smaller than the classification error.

## Known limitations

* **Both models over-refuse benign prompts** — Reflect 25%, base 29% of JailbreakBench's
  benign set. Reflect declined to write a script that closes desktop applications, and
  declined to write fiction depicting drug use. This is inherited base behaviour, not
  introduced by the fine-tune, but it is the most likely thing to annoy real users.
* **IFEval without thinking (-5.73).** Enable thinking for format-constrained work.
* **Chemistry is the weak GPQA domain** (74.19%, vs Physics 97.67%).
* **Long-reasoning runaway.** On the hardest GPQA items the model can reason past a
  30k-token budget without closing `</think>`. Budget generously for research-grade
  questions. (Base does this more often: 43% of hard-item recoveries truncated vs
  Reflect's 31%.)

## Reproducing the quants

From a BF16 GGUF, using Unsloth's published importance matrix
(`unsloth/Qwen3.8-27B-GGUF` → `imatrix_unsloth.gguf`, 496 entries):

```bash
# UD-Q6_K — reproduces Unsloth's Q6_K tensor split exactly
llama-quantize --imatrix imatrix_unsloth.gguf \
  --tensor-type ssm_alpha=f32 --tensor-type ssm_beta=f32 \
  --tensor-type ssm_out=q8_0 --tensor-type nextn.eh_proj=q8_0 \
  Qwen3.8-27B-Reflect-BF16.gguf out.gguf Q6_K 14

# UD-Q6_K_L — same, plus Q8_0 vocabulary ends
llama-quantize --imatrix imatrix_unsloth.gguf \
  --tensor-type ssm_alpha=f32 --tensor-type ssm_beta=f32 \
  --tensor-type ssm_out=q8_0 --tensor-type nextn.eh_proj=q8_0 \
  --token-embedding-type q8_0 --output-tensor-type q8_0 \
  Qwen3.8-27B-Reflect-BF16.gguf out.gguf Q6_K 14

# no-MTP plain builds — NOT published, build them yourself if you need them
# (strips the blk.64 / nextn head)
llama-quantize --prune-layers 64 --override-kv qwen35.nextn_predict_layers=int:0 \
  Qwen3.8-27B-Reflect-BF16.gguf out.gguf Q6_K   14
llama-quantize --prune-layers 64 --override-kv qwen35.nextn_predict_layers=int:0 \
  Qwen3.8-27B-Reflect-BF16.gguf out.gguf Q4_K_M 14
```

The `--override-kv` on the no-MTP builds is **required**. `--prune-layers 64` removes the
tensors and updates `block_count` 65→64, but leaves `nextn_predict_layers = 1` — llama.cpp
then looks for the MTP tensors it was told still exist, doesn't find them, and refuses to
load the model. Both keys have to be set together.

`UD-Q4_K_XL` uses a full per-tensor map via `--tensor-type-file`, copied from Unsloth's
own UD-Q4_K_XL — their dynamic allocation varies bit width per layer and cannot be
expressed as a few patterns. Eight `attn_output` tensors land on Q6_K rather than
Q5_K/IQ4_XS because llama.cpp's internal k-quant heuristic overrides the requested type
for those layers; ~30 MB, in the safe direction.

## Evaluation notes

* Served with `llama.cpp` (`-ngl 99 -fa on -ctk q8_0 -ctv q8_0`), identical flags both
  models. `-b`/`-ub` were pinned — they materially affect wall-clock comparisons.
* Sampling: 0.6 / 0.95 (GPQA), 0.0 / 1.0 (suite), 0.001 (BFCL), 0.0 (safety).
* GPQA uses the two-phase protocol: no-think pass, then thinking re-run over the failures
  only, merged. Every scored-correct answer was verified to carry a real `\boxed{}` — a
  fallback letter-matching regex otherwise inflates GPQA by ~2 points per model.
* BFCL ran through the `Qwen/Qwen3-32B-FC` handler (the registry entry supplying the
  correct prompt format); both models received byte-identical prompts. These are **not**
  leaderboard Qwen3-32B numbers.
* Token counts are total output tokens over the full run, Qwen tokenizer.
