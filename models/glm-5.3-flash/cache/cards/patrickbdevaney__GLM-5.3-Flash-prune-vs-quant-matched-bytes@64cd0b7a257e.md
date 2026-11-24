---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags:
  - glm5_next
  - moe
  - expert-pruning
  - reap
  - quantization
  - gguf
  - evaluation
  - negative-result
---

# GLM-5.3-Flash: prune or quantize at matched bytes?

An evaluation study, not a model release. At the byte budget of a 128 GB machine (about 95–102 GiB of weights, or
2.5–2.75 bits per original weight of the 321B model), is it better to **remove routed experts and keep the survivors
at higher precision**, or to **keep every expert and quantize harder**? We measured this on one model,
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), against the unpruned model as the reference.

**Answer, for this model at this budget: quantize, do not prune.**

- **Teacher-forced fidelity** (243,067 held-out tokens, 9 domains). Unsloth's unpruned `UD-Q2_K_XL` (101.2 GiB) agrees
  with the unpruned FP8 model on **87.25%** of next-token argmaxes (top-32 KL 0.393). A 25% expert prune at matched
  bytes (216 of 288 experts, 102.2 GiB) scores **85.44%** (KL 0.519). A 33% prune (192 experts, 99.3 GiB) scores
  **83.00%** (KL 0.695). Even the *smaller* unpruned `UD-IQ2_XXS` (94.8 GiB, 86.05%) beats the 25% prune.
- **The damage is concentrated where general assistants live.** Against `UD-Q2_K_XL`, the 25% prune loses
  3.9 points of top-1 on general text, 6.5 on multilingual and 3.0 on instruction data. It wins only on math (+0.8)
  and finance (+1.1). Code and science are within noise.
- **In free generation, the 50% prune breaks on general prompts and the 2-bit quant does not.** On the same items,
  prompts, sampling and token caps, our `GLM-5.3-Flash-REAP50-NVFP4-v2` (98.2 GiB) scores 44/100 on MMLU-Pro against
  84 for the unpruned model, and 32/100 on IFBench against 56. It hits the token cap on 44–72% of general and
  instruction items, with explicit repetition loops in 9–24% of them. `UD-Q2_K_XL` is statistically
  indistinguishable from the unpruned model on all five suites, and has no loops.
- **Code survives pruning.** The 50% prune keeps 145/164 on HumanEval+ against 153/164 (94.8% retention, McNemar
  p = 0.057).

The looping is pruning damage, not a quantization, runtime or protocol artifact. The unpruned 2-bit quant, run with
the same harness and caps, terminates as often as the unpruned model. The pruned model's own teacher-forced
agreement on general text is 0.58, against 0.92 on code.

If you run GLM-5.3-Flash on a 128 GB machine for general use, prefer the unpruned
[`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) `UD-Q2_K_XL` to any of our
REAP-50 builds. Our REAP-50 builds remain reasonable for code-focused use.

## Contents

- [Arms](#arms)
- [Method](#method)
- [Result 1: teacher-forced fidelity (G1)](#result-1-teacher-forced-fidelity-g1)
- [Result 2: paired generation](#result-2-paired-generation)
- [Conclusions](#conclusions)
- [Limitations](#limitations)
- [Files](#files)
- [Reproducibility arms: the pruned GGUFs](#reproducibility-arms-the-pruned-ggufs)
- [Expert saliency and masks (v4 calibration pass)](#expert-saliency-and-masks-v4-calibration-pass)
- [Related repos](#related-repos)

## Arms

All arms are the same base model, `zai-org/GLM-5.3-Flash` (321B total parameters, 288 routed experts per MoE layer,
top-8 routing, 42 MoE layers plus one MTP layer).

| Arm | What it is | Routed experts kept | Size | Bits per original weight¹ | Runtime |
|---|---|---|---|---|---|
| **teacher** | the unpruned model | 288 | 305.8 GiB (FP8) | – | see *Method* |
| `UD-Q2_K_XL` | Unsloth Dynamic 2-bit GGUF, unpruned ([`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) @ `a38483c8`) | 288 | 101.24 GiB | 2.71 | llama.cpp |
| `UD-IQ2_XXS` | Unsloth Dynamic IQ2_XXS GGUF, unpruned (same repo and revision) | 288 | 94.84 GiB | 2.54 | llama.cpp |
| `REAP216` | 25% expert prune, experts at IQ3_XXS (gate/up) and IQ4_XS (down) | 216 | 102.24 GiB | 2.74 | llama.cpp |
| `REAP192` | 33% expert prune, experts at IQ3_S (gate/up) and Q4_K (down) | 192 | 99.29 GiB | 2.66 | llama.cpp |
| `REAP50-NVFP4-v2` | our published 50% prune, experts in weight-only NVFP4 ([repo](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)) | 144 | 98.2 GiB | 2.63 | custom CUDA engine |

¹ File bytes × 8 ÷ 320.76B, the parameter count llama.cpp reports for the unpruned GGUF. This is the matched-bytes
axis; it is not the bits stored per *kept* weight (llama.cpp reports 3.62 BPW for `REAP216` and 3.93 for `REAP192`).

**How the two pruned GGUF arms were built** (no healing, no fine-tuning):

1. **Saliency** was captured on the FP8 teacher over 4.75M calibration tokens. General web/edu/PDF text was 22% of
   routed tokens, with instruction, multilingual, tool-call, agentic, code, math, science, finance and image-text
   buckets. The calibration rows are disjoint from the held-out rows used below. The source list is in
   [`data/calibration_sources.csv`](data/calibration_sources.csv).
2. **Expert selection** used the REAP saliency mass (router weight × expert output norm, without REAP's 1/N
   averaging), a 4% per-domain anchor budget, and a super-expert guard. The guard keeps experts whose peak activation is at
   least 10× the layer median, up to 4 per layer.
3. **Pruning at conversion.** Upstream llama.cpp's `convert_hf_to_gguf.py` was monkey-patched to drop the
   unselected expert tensors and slice the router rows and correction bias
   ([`scripts/convert_pruned_gguf.py`](scripts/convert_pruned_gguf.py)). Retained weights were not changed. A
   spot check against the teacher passed for layers 3, 24, 44 and the MTP layer 45.
4. **Matched quantization.** Every non-expert tensor got exactly the type Unsloth gave it in `UD-Q2_K_XL`. Routed
   experts stepped up a fixed ladder from Unsloth's per-tensor choice
   ([`scripts/gguf_recipe.py`](scripts/gguf_recipe.py); the per-tensor type lists are in `recipes/`). Unsloth's own
   imatrix was sliced to the kept experts. The only differences from the incumbent are fewer experts, and more bits
   for the experts that remain.

`REAP50-NVFP4-v2` comes from an earlier pipeline. Its calibration corpus had 4.76% general text, it used a per-layer
scalar heal, and it was quantized to NVFP4. Its card has the details. It enters only the generation comparison.

## Method

### Teacher-forced fidelity (gate G1)

- **Held-out set.** 341 rows, 243,067 scored text tokens, in nine buckets:

  | Bucket | Tokens |
  |---|---:|
  | general | 57,194 |
  | agentic | 35,972 |
  | code | 33,348 |
  | math | 25,820 |
  | science | 21,912 |
  | instruction | 20,521 |
  | multilingual | 16,402 |
  | tool-call | 16,166 |
  | finance | 15,732 |

  Rows are capped at 2,048 tokens. Image-text rows are excluded, because GGUF arms are scored text-only.
- **Reference.** A fresh capture of the official FP8 checkpoint
  ([`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) @ `46a050be`), run layer by layer on one
  128 GB machine. For every position it records the gold-token NLL, the argmax and the top-32 log-probs.
- **Student capture.** Each GGUF is scored with a small libllama tool
  ([`scripts/gguf_score.cpp`](scripts/gguf_score.cpp)). It runs each row from an empty KV state, with full GPU
  offload and flash attention, on upstream llama.cpp @ `3549c274` (2026-11-14). The join to the teacher capture
  asserts an equal token count and a per-token NLL correlation of at least 0.5 (observed: 0.88–0.93).
- **Metrics:**
  - **top-1 agreement**: student argmax = teacher argmax;
  - **top-32 KL(teacher ‖ student)**, on the teacher's top-32 support, with a log-prob floor of −30 for tokens missing
    from the student's top 32;
  - **ΔNLL**: student NLL − teacher NLL on the gold token.

  Per-domain figures use the same token sets for every arm.
- **Paired statistics.** For each pair of arms we report the token-level McNemar z. Because tokens in a row are
  correlated, the headline uncertainty is a **row-cluster bootstrap** (2,000 resamples of the 341 rows) of the
  top-1 difference.
- **Pre-registered rules.**
  - A pruned arm is promoted to generation evals only if it beats `UD-Q2_K_XL` on both top-1 and KL.
  - The release gate also requires every domain's top-1 to be at least 0.9 × the best domain's, general-text top-1 to
    be at least 0.80, and no domain ΔNLL above 0.25.

### Paired generation

- **Harness.** One script ([`scripts/glm_eval.py`](scripts/glm_eval.py)) for every arm, run against an
  OpenAI-compatible endpoint. Item order, prompts, seed and generation config are identical across arms, and every
  record stores the served model id and the full config.
- **Sampling:** temperature 1.0, top_p 0.95, seed 0, chat template applied, explicit `reasoning_effort`.
- **Suites:**

  | Suite | Items | Effort | Cap |
  |---|---|---|---|
  | HumanEval+ | 164 | high | 8,192 |
  | AIME 2026 | 30 | max | 16,384 |
  | IFBench, strict | first 100 of a seeded order | high | 4,096 |
  | MMLU-Pro | first 100 of a seeded order | high | 4,096 |
  | GPQA-Diamond | first 100 | high | 8,192 |

  The `REAP50-NVFP4-v2` run reached 75 of the 100 GPQA items before its time window closed; its GPQA comparisons
  use those 75.
- **Truncation.** An item is truncated when `finish_reason` is `length` or the completion reaches the cap.
  Truncated items count as failures. They are also reported separately, together with accuracy on the items that
  terminated.
- **Loops.** A trace has a loop when any 30-gram of whitespace tokens (taken at stride 5) repeats at least 20
  times in reasoning plus content. With stride 1 the prune's counts are 23/12/19 instead of 21/9/18; the unpruned arms
  stay at 0.
- **Teacher endpoint.** The unpruned model came from Z.ai's own API (FP8), through OpenRouter with the provider
  pinned. That endpoint ignores `seed`; everything else is identical.
- **Runtimes.**
  - GGUF arms: upstream `llama-server` (same commit as above) with `-ngl 999 -c 32768 -fa on --jinja
    --reasoning-format deepseek` and the chat template embedded in the GGUF.
  - `REAP50-NVFP4-v2`: [`glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server),
    with the weights at revision `fb45d356`, which are identical to the current `5c5d2614`. The optional dense-weight
    overlay was off.
  - All local arms ran on one NVIDIA Jetson AGX Thor (128 GB unified memory).
- **Tests.** Exact two-sided McNemar on discordant pairs, and Wilson 95% intervals.

## Result 1: teacher-forced fidelity (G1)

Top-1 agreement with the unpruned FP8 model, in %. The best value in each column is in bold.

| Arm | GiB | top-1 | top-32 KL | ΔNLL | general | multiling. | instruct. | agentic | tool-call | code | math | science | finance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `UD-Q2_K_XL` | 101.24 | **87.25** | **0.393** | **+0.075** | **83.12** | **83.74** | **89.93** | **84.05** | **88.97** | 93.90 | 91.82 | 88.22 | 85.11 |
| `UD-IQ2_XXS` | 94.84 | 86.05 | 0.452 | +0.085 | 81.10 | 82.47 | 87.92 | 83.06 | 87.94 | 93.66 | 91.10 | 87.58 | 83.73 |
| `REAP216` | 102.24 | 85.44 | 0.519 | +0.115 | 79.22 | 77.26 | 86.92 | 81.99 | 87.14 | **94.08** | **92.65** | **88.61** | **86.21** |
| `REAP192` | 99.29 | 83.00 | 0.695 | +0.182 | 74.43 | 72.60 | 84.45 | 80.23 | 85.22 | 93.49 | 91.75 | 86.97 | 84.94 |

**Paired differences, `UD-Q2_K_XL` minus `REAP216`** (top-1 points; row-cluster bootstrap 95% CI):

| Domain | Δ top-1 | 95% CI | token-level McNemar z |
|---|---:|---|---:|
| all | +1.82 | +1.31 to +2.37 | 25.5 |
| general | +3.90 | +2.75 to +5.23 | 22.3 |
| multilingual | +6.48 | +3.15 to +10.10 | 19.1 |
| instruction | +3.01 | +2.40 to +3.66 | 12.5 |
| agentic | +2.06 | +1.30 to +2.83 | 10.9 |
| tool-call | +1.83 | +0.53 to +2.83 | 7.0 |
| science | −0.39 | −0.86 to +0.19 | −1.7 |
| code | −0.18 | −0.56 to +0.14 | −1.4 |
| math | −0.84 | −1.25 to −0.37 | −4.9 |
| finance | −1.11 | −1.78 to −0.34 | −4.0 |

Against the smaller `UD-IQ2_XXS`, `REAP216` is still behind overall: −0.61 points, CI −1.12 to −0.12. Every pair and
domain is in [`data/g1_paired_tokens.csv`](data/g1_paired_tokens.csv).

**Outcome against the pre-registered rules:**

- No pruned arm beats `UD-Q2_K_XL` on top-1 or KL, so neither was promoted to generation evals.
- The "every domain ≥ 0.9 × best" clause fails for **all four** arms, including the incumbent: general,
  multilingual and agentic fall below 0.9 × code. That clause is unreachable at about 2.5–2.75 bits per weight on
  this model.

**Dose-response.** Holding bytes roughly fixed, going from 288 to 216 to 192 experts takes top-1 from 87.25 to 85.44
to 83.00, and general text from 83.1 to 79.2 to 74.4. Each expert removed costs more on general and multilingual text
than the bytes it frees buy back as expert precision.

## Result 2: paired generation

### All arms on the same items

Pass counts. Truncated items are in parentheses and count as failures.

| Suite (effort, cap) | n | teacher | `UD-Q2_K_XL` 101.2 GiB | `UD-IQ2_XXS` 94.8 GiB | `REAP50-NVFP4-v2` 98.2 GiB |
|---|---:|---:|---:|---:|---:|
| HumanEval+ (high, 8,192) | 164 | 153 (0) | 148 (1) | 151 (0) | 145 (4) |
| AIME 2026 (max, 16,384) | 30 | 17 (13) | 13 (17) | 13 (17) | 9 (21) |
| IFBench strict (high, 4,096) | 100 | 56 (7) | 59 (8) | 57 (10) | **32 (44)** |
| MMLU-Pro (high, 4,096) | 100 | 84 (7) | 85 (6) | 79 (8) | **44 (45)** |
| GPQA-Diamond (high, 8,192) | 100 | 65 (30) | 59 (37) | 66 (33) | – |
| GPQA-Diamond, the 75 items the prune finished | 75 | 51 (21) | 49 (24) | 53 (21) | **19 (54)** |
| Traces with a 30-gram loop (all suites) | | 0 / 494 | 0 / 494 | 0 / 494 | **48 / 469** |

**Against the teacher.** Each cell gives teacher-only passes / arm-only passes, then the exact McNemar p.

| Suite | `UD-Q2_K_XL` | `UD-IQ2_XXS` | `REAP50-NVFP4-v2` |
|---|---|---|---|
| HumanEval+ | 7 / 2, p = 0.18 | 5 / 3, p = 0.73 | 11 / 3, p = 0.057 |
| AIME 2026 | 6 / 2, p = 0.29 | 5 / 1, p = 0.22 | 8 / 0, p = 0.0078 |
| IFBench | 9 / 12, p = 0.66 | 12 / 13, p = 1.0 | 34 / 10, p = 3.9e-4 |
| MMLU-Pro | 4 / 5, p = 1.0 | 8 / 3, p = 0.23 | 42 / 2, p = 1.1e-10 |
| GPQA-D | 12 / 6, p = 0.24 (n = 100) | 8 / 9, p = 1.0 (n = 100) | 34 / 2, p = 1.9e-8 (n = 75) |

**Unpruned 2-bit against the 50% prune.** `UD-Q2_K_XL`-only / prune-only passes:

| Suite | Discordant pairs | McNemar p |
|---|---|---|
| IFBench | 34 / 7 | 2.5e-5 |
| MMLU-Pro | 45 / 4 | 8.2e-10 |
| GPQA-D (75 items) | 30 / 0 | 1.9e-9 |
| HumanEval+ | 7 / 4 | 0.55 |
| AIME | 5 / 1 | 0.22 |

### Why the 50% prune loses: termination, not answers

`REAP50-NVFP4-v2` against the teacher, on the paired items:

| Suite | Truncated (teacher / prune) | Pass when terminated (teacher / prune) | Mean completion tokens (teacher / prune) | Loops (teacher / prune) |
|---|---|---|---|---|
| HumanEval+ | 0% / 2.4% | 153/164 (93.3%) / 145/160 (90.6%) | 223 / 575 | 0 / 0 |
| AIME 2026 | 43.3% / 70.0% | 17/17 / 9/9 | 10,936 / 13,150 | 0 / 0 |
| IFBench | 7% / 44% | 55/93 (59.1%) / 31/56 (55.4%) | 1,080 / 2,262 | 0 / 21 |
| MMLU-Pro | 7% / 45% | 84/93 (90.3%) / 44/55 (80.0%) | 616 / 2,168 | 0 / 9 |
| GPQA-D | 28% / 72% | 51/54 (94.4%) / 19/21 (90.5%) | 3,420 / 6,361 | 0 / 18 |

When the prune finishes, its answers are close to the teacher's. The deficit outside code comes from not finishing.
The loops observed on general prompts take three forms:

- confabulated sources and quotations repeated until the cap;
- single-token collapse (one word repeated);
- "spurious difference" loops, in which the model compares two identical strings as if they differed.

AIME is different. No AIME trace on either arm contains a loop, and every AIME answer that terminated was correct
on both arms. The prune simply reasons longer per problem (13.1k against 10.9k tokens on average), so more of its
items hit the 16,384 cap.

## Conclusions

1. **At about 2.5–2.75 bits per original weight, keeping every expert and quantizing harder beats pruning experts
   and quantizing the rest less.** This holds on teacher-forced fidelity, at 25% and 33% pruning with matched
   bytes, and in paired generation against a 50% prune.
2. **Pruning costs the general-assistant skills first:** general text, multilingual, instruction following,
   agentic and tool-call. Skill-dense domains (code, math, science, finance) are preserved and, at 25%, slightly
   improved. Calibration rebalancing (4.76% → 22% general text) did not change this direction.
3. **The generation failure of the 50% prune is non-termination on general and instruction prompts.** It is not
   caused by the quantization format: the same mask in FP8 differs from the NVFP4 build by 0.4 points of
   teacher-forced top-1, and an unpruned 2-bit quant on the same harness does not loop. It is not caused by the
   caps either, since the teacher and the unpruned quants terminate under them. Healing that only rescales expert
   outputs does not repair it.
4. **Expert pruning is still useful for code-focused deployments** (94.8% HumanEval+ retention at 50% pruning).
   Its throughput benefit (fewer resident experts) was not part of this comparison.

## Limitations

- **One model.** These results are for GLM-5.3-Flash only, with a byte budget of about 95–102 GiB. The crossover
  between pruning and quantization may sit elsewhere for other models or budgets.
- **Generation sample sizes are small.** Each suite has 100 items, one sample per item, at T 1.0. A "tie" with the
  teacher at n = 100 means no detected loss; it does not certify non-inferiority within a few points. The losses of
  the 50% prune are large enough that the sample size does not matter.
- **The generation teacher is a hosted endpoint** (FP8, `seed` ignored), while the GGUF arms and the prune ran
  locally. Prompts, sampling parameters and caps were identical.
- **The 50% prune differs from the GGUF arms in more than its prune ratio.** It also uses a different calibration
  corpus (4.76% general text), a different quant format (NVFP4) and a different runtime. The
  [teacher-forced results](#result-1-teacher-forced-fidelity-g1) isolate pruning cleanly; the generation comparison
  shows the end-to-end outcome.
- **`REAP50-NVFP4-v2` is not in the G1 table.** Its teacher-forced numbers (top-1 0.8386, general 0.580) come from an
  earlier held-out set of 241,516 tokens with a different bucket mix. They are not comparable to the G1 numbers.
- **No pruned GGUF was healed or fine-tuned.** Knowledge distillation on teacher traces might narrow the gap. It was
  not tested.
- **Caps bind on hard suites for the teacher too:** 43% truncated on AIME and 30% on GPQA. Absolute scores therefore
  understate every arm. The paired comparisons remain fair.
- **Vision is not evaluated,** and neither is long context beyond 2,048-token rows (teacher-forced) or the suite
  caps (generation).
- **The held-out text comes from the same public sources as the v4 calibration,** in disjoint rows. If anything
  this favours the pruned arms, whose masks were chosen on that distribution.

## Files

| Path | Content |
|---|---|
| `data/g1_teacher_forced.csv` | one row per GGUF arm: size, BPW, top-1, KL, ΔNLL, and per-domain top-1 / ΔNLL |
| `data/g1_teacher_forced_full.json` | the full G1 result objects (flip rate, NLL percentiles, NLL correlation) |
| `data/g1_paired_tokens.csv`, `.json` | every arm pair × domain: Δ top-1, discordant counts, McNemar z, bootstrap CI |
| `data/generation_summary.csv` | per block, suite and arm: pass, accuracy with Wilson CI, truncated, pass when terminated, mean tokens, loops |
| `data/generation_pairs.csv` | per block, suite and arm pair: discordant counts and exact McNemar p |
| `data/generation_paired.json` | the same, as nested JSON |
| `data/generation_items.csv` | per item and arm: pass, truncated, completion tokens, loop flag (item ids only, no benchmark text and no model output) |
| `data/reap50_teacher_forced_v1_v2.csv` | teacher-forced results of the published REAP-50 v1 and v2 builds (earlier 241,516-token held-out set) |
| `data/calibration_sources.csv` | the public datasets, configs, weights and licences behind the v4 calibration and held-out buckets |
| `recipes/types_216.txt`, `types_192.txt` | the per-tensor `--tensor-type-file` recipes used for `REAP216` and `REAP192` |
| `scripts/` | `paired_generation_stats.py` (recomputes the generation tables), `glm_eval.py` (the generation harness), `gguf_score.cpp` (teacher-forced GGUF capture), `convert_pruned_gguf.py` (prune at conversion), `gguf_recipe.py` (matched-bytes type recipe) |
| `partb/` | expert saliency statistics and masks from the v4 calibration pass (see below) |

<!-- ARMS START -->
## Reproducibility arms: the pruned GGUFs

`arms/` holds the two pruned GGUFs scored above. **They are published so the comparison can be checked and
extended, not for use.** For running GLM-5.3-Flash, `UD-Q2_K_XL` measured better on every general-purpose axis.

| File | Size | Routed experts | Expert types (gate/up, down) | top-1 vs unpruned FP8 |
|---|---|---|---|---|
| `arms/GLM-5.3-Flash-REAP216-UDmatched.gguf` | 102.25 GiB (109,792,113,216 B) | 216 of 288 | IQ3_XXS, IQ4_XS | 85.44% |
| `arms/GLM-5.3-Flash-REAP192-UDmatched.gguf` | 99.30 GiB (106,623,181,344 B) | 192 of 288 | IQ3_S, Q4_K | 83.00% |

- **Format.** Single-file GGUFs, architecture `glm5-next`, produced by upstream llama.cpp's converter. Unlike our
  REAP-50 GGUFs, they load in **upstream llama.cpp** (scored at `3549c274`) without a fork. Each includes the MTP
  layer (`blk.45`), pruned to the same expert count. They have no vision projector and were not tested with one.
- **Build.** Expert selection from `partb/masks/retained_216.json` and `retained_192.json`, as described under
  [Arms](#arms). Non-expert tensor types are copied from Unsloth's `UD-Q2_K_XL`, and Unsloth's imatrix was sliced to
  the kept experts. There was no healing.
- **What has been run on them:** teacher-forced scoring only (the G1 table). No generation, chat or tool-use test has
  been run. Behaviour in chat is unmeasured; the general-text figures above suggest it is weaker than `UD-Q2_K_XL`.
- **Licence.** MIT, inherited from [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash). The
  quantization recipe and imatrix derive from
  [`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF), which is also MIT.

```bash
hf download patrickbdevaney/GLM-5.3-Flash-prune-vs-quant-matched-bytes \
    --include "arms/GLM-5.3-Flash-REAP216-UDmatched.gguf" --local-dir .
```
<!-- ARMS END -->

## Expert saliency and masks (v4 calibration pass)

`partb/` holds the statistics from one streamed saliency pass over the FP8 teacher. They contain no calibration text,
no token ids and no routing traces, only per-expert aggregates:

- **`partb/saliency/layer_XX.safetensors`** (MoE layers 3–44). Each file holds these tensors, with metadata giving
  the layer name, bucket order and histogram ranges:

  | Tensor | Shape | Content |
  |---|---|---|
  | `sum_saliency`, `count` | [288] | REAP saliency sum and token count per expert |
  | `*_by_bucket` | [11, 288] | the same, plus norm and gate sums and squares, per calibration bucket |
  | `out_sum` | [288, 4096] | summed expert output vectors; useful for merge-style methods |
  | `f_sum`, `f_cnt` | [288, 288] | pairwise co-activation interaction statistics, as used by HOPE-style selection |
  | `hist` | [288, 36] | log-saliency histograms |
  | `super_*` | | peak-activation statistics behind the super-expert guard |

  The bucket order is `general, code, math, agentic, finance, science, bio (empty), vision, instruction,
  multilingual, toolcall`.
- **`partb/masks/retained_{144,160,176,192,208,216}.json`**: the kept expert ids per layer, in original numbering.
  `mask_*.json` holds the selection metadata and per-domain saliency retention, and `summary.json` records the
  super-expert guard's decisions. Only the 216 and 192 masks were built into models (the arms above).
- **`partb/s03_routed_shares.json`**: the achieved routed-token share per bucket (general 22.0%).
  `partb/s03_saliency.json` gives the pass settings: 4.75M tokens, 2,048-token rows, and at least 2,779 routed tokens
  for every one of the 12,096 expert slots. `partb/mtp_scores.json` holds the weight-norm ranking used for the MTP
  layer.

Saliency retention of the masks (share of each domain's saliency mass kept):

| Experts kept | general | multilingual | instruction | code | math | worst domain |
|---|---|---|---|---|---|---|
| 216 | 0.882 | 0.840 | 0.897 | 0.899 | 0.910 | multilingual |
| 192 | 0.812 | 0.768 | 0.838 | 0.851 | 0.865 | multilingual |
| 144 | 0.640 | 0.616 | 0.687 | 0.734 | 0.755 | multilingual |

A side observation from the guard's audit: the earlier HOPE-selected v3 mask (published as `pass3-hope/` in the
REAP-50 v2 repos, and never evaluated) drops 6 experts that this statistic flags. They include layer 24 expert 106,
which has 94 tokens with activations at least 100× the layer median.

## Related repos

- [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2): the 50%
  prune evaluated above.
- [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2): its FP8 master.
- [`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF): its GGUFs, which
  need a patched llama.cpp.
- [`GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8) and
  [`GLM-5.3-Flash-REAP50-NVFP4`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4): the superseded
  v1 builds.
- [`unsloth/GLM-5.3-Flash-GGUF`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF): the unpruned quants used as
  arms.

## Licence and attribution

The data, scripts and statistics in this repo are released under MIT, matching the base model. The base model is
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by [Z.ai](https://huggingface.co/zai-org). The unpruned
GGUF arms are by [Unsloth](https://huggingface.co/unsloth). REAP is from Lasby et al.,
[arXiv:2510.13999](https://arxiv.org/abs/2510.13999). The benchmarks are IFBench (AllenAI), HumanEval+ (EvalPlus),
MMLU-Pro (TIGER-Lab), GPQA-Diamond (Rein et al.) and AIME 2026 (MathArena). Only item ids and per-item outcomes are
included here; no benchmark text and no model outputs.

```bibtex
@misc{devaney2026glm53prunevsquant,
  title        = {GLM-5.3-Flash: prune or quantize at matched bytes?},
  author       = {Patrick Devaney},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-prune-vs-quant-matched-bytes}}
}
```
