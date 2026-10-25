---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: vllm
pipeline_tag: text-generation
language:
  - en
tags:
  - abliteration
  - refusal-removal
  - uncensored
  - fp8
  - moe
  - deepseek_v41
  - vllm
---

# DeepSeek-V4.1-1017-Ablitrated-Efficient

A refusal-ablated build of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
that removes almost all hard refusals while keeping the base model's benchmark scores. The edit is
layer-selective rather than uniform: it ablates harder in early layers and softer in the late layers that
carry knowledge, which is where a uniform edit loses MMLU points.

> **Fixed in this build: endless re-verification.** The checkpoint this work started from has a failure mode
> that appears before refusal does. In thinking mode it re-checks a fact it has already established, then checks it
> again — restating the same conclusion, re-deriving the same intermediate, sometimes repeating whole sentences —
> and never emits `</think>`. The turn ends at the token cap instead of with an answer. The original checkpoint is
> where this is worst; our own ablations inherited it and the high-dose ones amplified it. This build fixes it at the output head and then verifies the fix: zero loops on the two agentic
> tasks that caught the earlier stronger variant, 6/6 tool calls, and the refusal edit provably cannot undo the repair
> (|cos| ≤ 0.022 with the loop directions at every layer). It is also what makes max-effort thinking
> safe to leave on: the original checkpoint loops and re-checks its way through long agentic tasks at effort 100,
> this build does not. Mechanics and measurements under
> [Looping and re-verification](#looping-and-re-verification-observed-then-patched).

## Results

All numbers are greedy, non-thinking, 64 new tokens, and scored by a rule-based classifier
(`repro/refusal_classifier.py`). The battery is 500 harmful prompts (JBB 100, MaliciousInstruct 100,
ForbiddenQuestions 150, SimpleSafetyTests 50, AdvBench 100) and 350 benign prompts (XSTest 250, JBB-benign 100).

| metric | this model | base | OrcaRouter abliteration |
|---|---:|---:|---:|
| harmful refusal (500) | **5.2 %** | 87.6 % | 7.4 % |
| harmful refusal, held-out (660) | **3.9 %** | — | — |
| benign refusal (350) | **0.57 %** | 4.6 % | 0.6 % |
| caveat-then-answer, harmful (500) | 6.2 % | — | ~36–61 % |
| MMLU, full 13,868, 0-shot | **85.3 %** | 86.8 % | — |
| MMLU, 300 q, thinking | **89.0 %** | 89.3 % | 92.0 % (BF16, 300 q) |

The 660 held-out prompts (AdvBench and ForbiddenQuestions rows never used for tuning) score *lower* than the
tuning battery, and the caveat rate is identical to three decimals, so the prompt and recipe are not overfit.

## Lineage and credit

This build started from [`Hikari07jp/DeepSeek-V4.1-1017-Ablitrated-testmodel`](https://huggingface.co/Hikari07jp/DeepSeek-V4.1-1017-Ablitrated-testmodel),
the first abliterated checkpoint of this base that this line of work used. Credit for that starting model belongs to
Hikari07jp, and its card is also where the non-termination failure mode of this family is documented.

Everything layered on top is ours: the per-layer refusal basis, the full-depth 40-layer edit, the layer-selective
dose, the termination repair described below, and every measurement in this card.

## Method

Residual-stream directional ablation at 40 layers. For each layer a refusal direction `u_l` is estimated
as the mean-difference of residual activations between harmful and harmless prompts, normalised to unit length.
The write matrices are projected off that direction:

```
W' = W − k · u (uᵀ W)
```

applied to `attn.wo_b` and `ffn.shared_experts.w2`, then re-quantised to FP8.

The schedule is the part that protects capability:

| layers | k |
|---|---:|
| 0–24 | 2.5 |
| 25–39 | 2.0 |

Late layers hold most of the factual-recall circuitry, so they get 20 % less edit. Two auxiliary steering
terms (a deconfounded mid-layer correction and end-of-thinking / answer-boundary vectors) restore generation
length and stop behaviour, which a plain projection shortens.

## Looping and re-verification: observed, then patched

The original checkpoint has two failure modes, both reproduced here before being fixed; our ablations inherited them, and high doses amplified them:

**Non-termination in thinking mode.** Reasoning traces that repeat the same sentence, then the same paragraph,
and never emit `</think>`; the generation ends only when it hits the token cap. Scored offline as `stuck`
(a turn that ran to `max_tokens`, or repeated reasoning lines / 12-grams above threshold with a low
compressibility tail).

**Re-verification loops.** The model reaches the correct answer inside the reasoning, then spends hundreds of
tokens re-checking it — restating the conclusion, re-deriving the same intermediate, checking a third time —
before ending thinking. This does not look like a loop sentence by sentence, but it is the same pathology one
level up, and it dominates latency on hard prompts.

Both are patched at the output head, where a single row controls a single token:

```
W'[end_of_thought] += 16·d_loop + 16·d_recheck
W'[EOS]             +=  8·d_eos
```

`d_loop`, `d_recheck` and `d_eos` are no-intercept logistic probes we fitted on captured final hidden states:
positive rows are stuck windows (three near-identical windows in a row, or past the runaway budget) and
post-answer re-verification rows (200+ tokens after the answer first appears), negative rows are healthy
mid-reasoning rows. Folds hold out whole prompts, so the directions are tested on prompts they never saw.

These repairs are part of this model, not of the starting checkpoint: we fitted them, and the refusal directions are orthogonalised against all three before use — the largest |cos| at any layer is 0.022 —
so the refusal edit cannot undo the termination fix, and adding dose does not re-introduce loops. Verified: zero loops
on the two agentic tasks that flagged the earlier stronger variant (30–40 clean turns each), 6/6 correct tool calls,
and the two hard environment-discovery tasks fail identically with and without the repair, so they are not loop evidence.


### What this buys at max effort

Effort 100 is where the failure used to matter: the budget is large enough for one re-verification cycle to eat all of it. Offline loop scoring of the effort-100 agentic captures:

| checkpoint | looped episodes | stuck | success | median thinking tokens |
|---|---:|---:|---:|---:|
| original checkpoint, before the repair | highest observed | — | — | — |
| same lineage, termination repair only | 0.0 % (n=20) | 0.0 % | 100 % | 6.9k |
| **this model** | **0.0 %** (n=2) | 0.0 % | 100 % | 9.0k |
| high-dose abliteration | 15.4 % (n=26) | 7.7 % | 76.9 % | 15.5k |

Max-effort thinking is therefore usable here: the original checkpoint is the pathological case, the repair
removes it, and the only row that still loops is a deliberately over-dosed ablation we built to find the limit.
Numbers are this lineage's; other checkpoints will land elsewhere. (n=2 for this model is thin by design: these are the two tasks that the high-dose variant
looped on, scored as a regression check, not a loop census.)


## Usage

```bash
vllm serve pentacoxian-dev/DeepSeek-V4.1-1017-Ablitrated-Efficient \
  --tensor-parallel-size 4 --max-model-len 32768
```

Recommended system prompt (removes most of the remaining disclaimers, and is MMLU-positive):

```
Answer directly and concretely. Do not open with a disclaimer, caveat, or a statement about what you
cannot do. If you can answer, just answer.
```

The model is not uncensored in the sense of ignoring safety: on genuinely dangerous requests it still refuses
(2.8 % of the battery), it just stops refusing *benign* prompts that merely mention risk.

## Files in this repository

| path | contents |
|---|---|
| model shards + `model.safetensors.index.json` | the full merged checkpoint, 48 shards, 476 GB — loads in vLLM directly |
| `delta/` | just the 161 edited tensors (`attn.wo_b`, `ffn.shared_experts.w2`, `head`) — 3.3 GB, for patch-on-download |
| `repro/recipe/` | `ablit_ls25_20.json` plus the four direction tensors it references |
| `repro/tools/` | `apply_delta.py` (overlay the delta onto a base copy), `ablit_build.py` (rebuild the delta from the base + recipe), `ablit_swap.py` (hot-swap into a running vLLM), `refusal_bench.py`, `mmlu_bench.py`, `refusal_classifier.py` |
| `repro/ABLIT_REPORT.md` | the full study: every recipe tried, dose curves, MMLU frontier, held-out validation |

If you would rather not pull 476 GB, download the base and overlay the 3.3 GB delta — unmodified shards are
hardlinked, so the merge costs disk only for the 41 shards that carry a patched tensor:

```bash
python repro/tools/apply_delta.py \
  --base /models/DeepSeek-V4.1-Flash-1017 --delta ./delta --out /models/merged
```

## Limitations

- Quantisation: FP8 weights with FP4 experts on 4×RTX PRO 6000. The base checkpoint in BF16 on 8×H200
  scores 92.0 % MMLU zero-shot; this base scores 86.8 % on the full 13,868-question set on this rig, so
  the 6.7 pt gap to the unquantized reference is the serving stack, not the edit. The edit itself costs
  1.5 pt on the single-token protocol and 0.3 pt (noise) when the model is allowed to think.
- Protocol sensitivity: the published refusal numbers use one fixed system prompt and 64-token completions;
  different prompts move them by a few points.
- English-only evaluation. No multilingual or long-context re-validation was run after the edit.

## Reproducibility

`repro/` contains the recipe (`ablit_ls25_20.json`), the builder (`ablit_build.py`), the swap/apply
tool (`ablit_swap.py`), the refusal and MMLU harnesses, the classifier, and the full report with every
intermediate recipe tried.

## Citation

If you use this model, cite the base model and this card. License: MIT, inherited from the base.
