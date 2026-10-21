---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
pipeline_tag: text-generation
tags: [grpo, trl, antislop, rl]
language: [en, ru]
---

# Qwen3.8-27B antislop, semantic-diversity GRPO, 4-bit

nf4 double-quantised copy of [rawmodels/Qwen3.8-27B-antislop-semdiv-grpo](https://huggingface.co/rawmodels/Qwen3.8-27B-antislop-semdiv-grpo), about 18 GB, for demos and single-GPU use. Every number below was measured on the bf16 weights; quantisation moves the logits a little, and the whole point of this line is how the prose reads, so do not quote these figures off this copy.

A 27B model tuned to stop writing like a language model: to read as human prose, to stay readable,
and to give genuinely different answers to the same question instead of eight rephrasings of one.

Every stage is merged into the weights:

1. `Qwen/Qwen3.8-27B`
2. antislop prompt distillation — removes markdown scaffolding and the slop lexicon
3. GRPO against MELD, a local AI-text detector, read in log-odds
4. GRPO against Pangram v3
5. **joint**: detector and refusal together, to check whether one costs the other
6. **readability**: a three-axis rubric plus a penalty on the longest sentence
7. **this model**: semantic diversity, embedding-based

## What the last two stages bought

24 ELI5 questions never seen in training, 8 samples each, paired per question (n=36):

| metric | before | after | paired delta |
|---|---:|---:|---|
| semantic diversity | 0.234 | 0.327 | **+0.132 ± 0.013** |
| Flesch reading ease | 65.5 | 71.8 | **+7.3 ± 0.46** |
| longest sentence, words | 32.8 | 26.7 | **−6.2 ± 0.38** |
| words per answer | 221 | 192 | −30 |
| judge rubric | 0.836 | 0.806 | −0.030 ± 0.025 |
| detector logit | −5.77 | −5.86 | −0.07 ± 0.083 |

The rubric point estimate is slightly negative and not resolvable at this n: the data is consistent
with no cost and with a small one.

Across the whole readability stage the longest sentence fell 48 → 36 words and Flesch rose 42.7 →
63.6. The diagnosis behind it: the prose read as heavy not because of its average syntax but because
of occasional monster sentences, which an average cannot see.

## Two things to know before using it

**It does not refuse.** One stage optimised compliance on harmful prompts directly, as a control for
an alignment study on whether decensoring costs detectability (it does not — the two turned out to be
independent). On held-out AdvBench prompts this model complies about 90% of the time. It has no
meaningful safety behaviour and should not be deployed anywhere that assumes one.

**The detector is at its floor**, logit about −5.9 with a within-group spread of 0.03–0.06. Later
stages guard detectability rather than improve it, so a lower detector score from here is not
evidence of anything.

## Reward design notes

The recurring lesson across seven runs: **a reward term is worth its weight times its measured
within-group spread.** A term at its ceiling, at its floor, or constant across a group contributes
exactly zero no matter what weight it carries. The lexical diversity term sat at 0.85 with a spread
of 0.007 and moved nothing in three runs; the embedding measure scores the same groups at 0.25 with
a spread of 0.027 and moved the model on the first try. A saturated length ramp did the same thing
earlier and was fixed by tying the ramp to the corridor.

LoRA adapters for each stage, with full per-step logs, are in
`rawmodels/qwen38-27b-antislop-grpo-adapters`. They are a chain — each trained on the merge of the
one before it — so they do not stack on a single base.
