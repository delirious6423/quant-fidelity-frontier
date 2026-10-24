---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags: [reap, moe, pruning, glm5_next, jetson, thor, nvfp4, compressed-tensors]
---
# GLM-5.3-Flash REAP-50 (NVFP4)

50% of routed experts removed with **REAP**
(Router-weighted Expert Activation Pruning, arXiv:2510.13999), calibrated on a
permissively-licensed multi-domain corpus that includes real image-text pairs.

| | |
|---|---|
| Base | `zai-org/GLM-5.3-Flash` (MIT, FP8 E4M3, 128x128 block scales) |
| Experts | 288 -> 144 per layer, top-8 routing unchanged |
| Size | **98.2 GiB** v2 (repo root) / **93.6 GiB** v3 (`pass3-hope/`), NVFP4 compressed-tensors — both fit the 117 GiB Thor envelope |
| Healed | yes |
| MTP block | present (layer 45, 144 experts; see *Corrections (2026-10-24)*) |


> **Card corrected 2026-10-01.** This card previously carried the FP8 card's title and size
> (160.6 GiB), so this repo appeared to hold FP8 weights. The weights were always NVFP4; only the
> card was wrong.

## Corrections (2026-10-24)

This card previously described the v3 (`pass3-hope/`) weights more favourably than the build logs
support. **No weights were changed**; only the description is corrected.

1. **v3 was not healed with measured or per-expert gains.** The healing stage detected that the
   available measured gains (router-replay scalars and per-expert coefficients) had been fitted
   against the v2 keep-set, not v3's. That check was meant to stop the run; an over-broad
   exception handler turned it into a warning, and v3 was healed with the fallback
   **first-moment per-layer scalar** (median gain 0.766, range 0.667-0.853, 42 layers) with
   **zero per-expert coefficients**. On v2 data this estimator gave 0.7149 where router replay
   measured 0.9111, i.e. it over-corrects by roughly 30%. Earlier text describing v3's healing as
   "per-expert", or as the measured correction, was wrong.
2. **v3 has not been evaluated end to end.** No teacher-forced or generative evaluation has been
   run on the v3 weights, so there is no evidence that v3 is better than, equal to, or worse than
   v2. What v3 changes (HOPE selection, `protect_frac = 0.04`) is design intent, not a measured
   improvement.
3. **The v3 NVFP4 build also quantises the KDA attention q/k/v projections**
   (`nvfp4_attn_mode="kda"`). On v2 this exact configuration was measured at **-0.0023 top-1
   agreement (z = 5.8)** against the build that keeps attention in BF16, and it was deliberately
   not shipped for v2. The v2 NVFP4 at the repo root keeps attention in BF16 (98.2 GiB); the v3
   NVFP4 does not (93.6 GiB), so it carries that measured cost in addition to the healing issue.
4. **The MTP block (layer 45) is present** in both the v2 and v3 weights, pruned to 144 experts
   like every other MoE layer. It has no activation saliency (`transformers` does not run it), so
   its experts were ranked by weight norm. Earlier text saying the MTP block was excluded was
   wrong for these weights.
5. **The best proven 50% recipe is still v2** (repo root): the pass-2 mask with the measured
   per-layer scalar heal, FP8 top-1 agreement **0.84249** against the unpruned model on 241,516
   held-out tokens. Use v2 unless you specifically want to experiment with v3.
6. **Planned:** re-heal v3 with the keep-set check made fatal (so gains are refitted on v3's own
   keep-set), then a paired v3-vs-v2 evaluation on the same held-out tokens. This note will be
   replaced with those results.

## Versions — which weights are which

Three selection passes have shipped. Each is a **different mask**, not a re-run of the same one,
and all keep **144 of 288 routed experts** (REAP50). What differs is *which* 144.

**Older weights are never overwritten.** The subdirectory tells you the version.

| version | method | healing | location in this repo |
|---|---|---|---|
| **v2** | refined REAP scalar mask | per-layer scalar (per-expert reverted 2026-08-29, see *Corrections*) | **repo root** (the files beside this card) |
| **v3** | **HOPE** — interaction-aware, `protect_frac=0.04` | first-moment per-layer scalar fallback (median 0.766), **not** the measured gain; 0 per-expert — see *Corrections (2026-10-24)* | subdirectory **`pass3-hope/`** |

> The repo name's `-v2` suffix is from the pass-2 release and is now just a name. Read the
> **path**, not the repo title: root = v2, `pass3-hope/` = v3.

v1 (the first published REAP mask) is superseded and not carried here; it remains in the separate
`GLM-5.3-Flash-REAP50-NVFP4` repo.

### What v3 changed, and why

v2 ranked each expert alone and took the top-k. [arXiv 2609.18916] shows that is exactly HOPE with
the off-diagonal of `F` zeroed — two experts that duplicate each other are cheap to drop together,
two that complement each other are not, and a per-expert ranking cannot tell the difference. v3
minimises `pᵀFp`, the output error a prune set actually causes, over the same accumulators.

v2 also ran with **no domain floor**, and ballast is its worst-retained domain (top-1 agreement 0.580). v3 sets
`protect_frac = 0.04`, measured off this run's own accumulators at the knee of worst-domain
retention against interaction cost — not guessed, and not inherited from MiMo's 0.08, which sits
where marginal efficiency is 0.43.

v3 is calibrated on **866,678,064 routed slots** (42 layers x 10 chunks). Its healing used the
first-moment fallback (median gain 0.766, 42 layers, none skipped), **not** the measured
router-replay gain that v2 uses; see *Corrections (2026-10-24)*.

### Honest comparison

v1 -> v2 was first reported here as **zero to within noise** end to end (top-1 agreement 0.8370 ->
0.8369, SE 0.00075 on 241,516 held-out tokens). That comparison was confounded: v1 shipped scalar
healing and v2 shipped per-expert healing. **Holding healing fixed, the v2 mask is worth +0.00545
top-1 agreement over v1 (McNemar z = 9.3)**, and the v2 FP8 parent with the per-layer scalar it now
ships scores **0.84249** (this NVFP4 build: 0.83857). See *Corrections* at the end of this card.
**v3 has not yet been evaluated end to end** — when it has, the
numbers will be published here. Until then, treat v3 as "better-motivated selection, unproven
head-to-head, healed with a fallback estimator", not as a measured improvement.

### Caveats that apply to v3

- **Healing is the first-moment fallback**, not the measured gain, and v3 is **not evaluated**
  end to end. See *Corrections (2026-10-24)*.
- **No bio coverage.** The bio domain saw **0** of 866,678,064 routed slots, because every bio
  source is excluded by licence or clinical scope (`camel-ai/biology`, `tattabio/OG`, `PubMedQA`).
  Selection scored the 7 live domains. This is a property of the calibration corpus, not a bug, and
  no amount of recomputation changes it. **Do not rely on this model for biomedical work.**
- **Long-context assurance is at 8192 tokens**, not the intended 16384. A single 16384-token
  forward needs >93 GiB — attention is quadratic and one forward is indivisible, so no amount of
  blocking helps on the build host. 8192 is still 4x the 2048 tokens the mask was calibrated at.

Full method write-up: **[glm-5.3-reap/wiki/31-pass3-hope-method.md](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/31-pass3-hope-method.md)**

### Download a specific version

```bash
# v3 (HOPE) - 93.6 GiB, 62 shards
hf download patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
    --include "pass3-hope/*" --local-dir ./glm53-reap50-nvfp4-v3

# v2 - repo root
hf download patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
    --exclude "pass3-hope/*" --local-dir ./glm53-reap50-nvfp4-v2
```

v3 NVFP4 is **93.6 GiB**, which fits the 117 GiB Jetson AGX Thor envelope with 23 GiB spare. It is
smaller than v2's 98.2 GiB because it also quantises the KDA q/k/v attention projections, which
v2 deliberately leaves in BF16 (see *Corrections (2026-10-24)*).

## Why FP8 and not BF16

The upstream release is **FP8**, not BF16. Routed experts are stored per-expert with their own
`weight_scale_inv` block scales, so pruning is deleting whole tensors - **lossless on every
retained weight**. The 642 GB BF16 repo elsewhere on the Hub is a dequantised upcast carrying
no additional information.

## Calibration

Mixture weighted for a coding/agentic model that stays empirically grounded: agentic 24%,
code 21%, math 15%, multimodal 15%, science+bio 10%, finance 8%, ballast 7%.
Permissive licences only, so this checkpoint keeps the base model's MIT lineage.

Vision is first-class: the vision tower contains no MoE and is untouched, but image tokens
route through the same expert pool as text, so text-only calibration would have deleted
vision-serving experts with certainty. Real image-text pairs were asserted present.

## Evaluation status

v2 (repo root): teacher-forced evaluation below. v3 (`pass3-hope/`): not yet evaluated end to end.
No generative benchmarks have been run on either.

<!-- EVAL:BEGIN -->
## Measured against the unpruned teacher

Teacher-forced paired evaluation on **241,516 held-out tokens** the calibration never
saw, scored against unpruned `zai-org/GLM-5.3-Flash` on identical inputs.

| | |
|---|---|
| **Top-1 agreement** | **0.839** |
| ΔNLL (student − teacher), mean | +0.1835 |
| ΔNLL, median | +0.0017 |
| Top-k KL (teacher ‖ student) | 0.6725 |
| Teacher NLL / student NLL | 1.018 / 1.201 |

| domain | tokens | top-1 agreement | ΔNLL |
|---|---|---|---|
| agentic | 69,202 | **0.869** | +0.153 |
| math | 66,558 | **0.916** | +0.024 |
| science | 45,034 | **0.827** | +0.129 |
| finance | 22,215 | **0.748** | +0.191 |
| ballast | 22,154 | **0.580** | +0.996 |
| code | 15,821 | **0.917** | +0.055 |
| vision | 532 | *n too small* | *n too small* |

`1 − flip_rate` against the unpruned model is the same quantity aggressively-quantised releases
quote as "retains X% of top-1 accuracy", so this number is comparable to a heavily-quantised GGUF
of the same base — provided both are measured against that same unpruned reference.

**Image-placeholder positions are excluded**, as the training objective excludes them. After that
exclusion the vision bucket is too small to support a conclusion: vision is **unmeasured here**,
which is not the same as measured-and-fine. Its per-domain saliency retention (0.682, mid-pack of
seven domains) says the mask did not strip vision-serving experts; whether the survivors suffice
is untested.

**This is teacher-forced agreement, not capability.** It measures how far the student moved from
the teacher on ground-truth prefixes. It does not measure whether the model is *smart* — only
generative benchmarks do, and none has been run.
<!-- EVAL:END -->

## What this prune costs, and what it does not

REAP ranks experts on saliency pooled across the calibration mixture, so the cost is **not**
spread evenly across capabilities. Measured on this checkpoint - the fraction of the expert
output each domain actually relies on that survived the prune:

| domain | retained |
|---|---|
| agentic / tool use | 0.747 |
| code | 0.728 |
| science | 0.720 |
| maths | 0.713 |
| vision (image-text) | 0.682 |
| finance | 0.651 |
| general / ballast | **0.487** |

**The damage is concentrated where retrieval can repair it.** Generic factual ballast is the
worst-retained bucket, and it is also the single most RAG-recoverable capability: a markdown
corpus and a retriever substitute for memorised trivia almost perfectly. The best-retained
buckets - agentic behaviour, code, science, maths - are the ones retrieval **cannot** restore,
because you cannot retrieve your way to reasoning, tool use, or code synthesis.

If you are deploying this, pair it with retrieval. That is not a workaround for a defect; it is
the shape the trade was made in. A uniformly-quantised model of the same footprint spends its
degradation budget evenly, including on the capabilities retrieval cannot give back.

Two honest caveats. Low retention can also mean a domain uses experts *diffusely* rather than
that its capability was removed - generic web text spreads across many experts while REAP keeps
concentrated ones. And the per-domain spread was a **consequence** of the calibration mixture,
not a targeted design: this is mixture-shaped preservation, not surgery.

## Method: strengths and gaps

**Where this is above common practice**

- Calibration is 5.5M tokens across seven use-case-weighted domains, against the 128-512 generic
  samples typical of pruning and quantisation work.
- **Real image-text pairs** in calibration, not text descriptions of images. Measured effect:
  vision retained 0.682, at the cross-domain mean. Text-only calibration deletes
  vision-serving experts with near-certainty.
- Saliency capture was **verified against the model source** - the router gate excludes
  `e_score_correction_bias`, which is the detail most implementations get silently wrong.
- Per-token router scores were cached, making post-prune routing exactly replayable offline.
- For v2 (repo root), the healing correction was **measured, not derived**. The standard
  first-moment estimator over-corrects by ~30% on this architecture because it ignores
  `norm_topk_prob` renormalisation; that error was found and fixed only because of the router
  cache. (v3 was healed with that first-moment estimator; see *Corrections (2026-10-24)*.)
- The calibration budget was validated by a split-half stopping rule on retained saliency mass,
  rather than assumed.

**Where it falls short, stated plainly**

- **Healing rescales experts; it does not retrain them.** Any *fixed rescaling* has a ceiling,
  and that ceiling is now measured rather than assumed: 96.6% of an
  expert's output energy is token-dependent residual rather than its mean, so the ~0.27 residual
  that remains is information deleted with the pruned experts, not error a better fit could
  remove. Recovering it means layer-local distillation, which needs a backward pass that does not
  fit the hardware this was built on.
- **No expert merging** (REAM/EEP), rejected on cost.
- **Evaluation is teacher-forced dNLL and top-1 agreement, not generative benchmarks.** That is
  the largest gap: dNLL does not fully predict agentic or coding capability, which is what this
  model is for. Generation from the unpruned teacher costs ~110 s/token on one device and was
  out of reach.
- **Long context is uncalibrated.** Calibration sequences were capped at 2048 tokens against a
  1M-token context window.
- The calibration mixture was specified in *samples* but acts in *tokens*; document lengths
  differ by more than 10x, so realised domain weights differed substantially from intended.
- A single prune ratio was materialised. No 40/50/60 ablation with evaluation behind it.

## Known limitations

- **The MTP (multi-token-prediction) block at layer 45 is present**, pruned to 144 experts. `transformers`' `Glm5NextForConditionalGeneration` does not instantiate it, so no activation saliency exists for it and its experts were ranked by weight norm. (Corrected 2026-10-24: this line previously said the block was excluded.)
- REAP has no published data above 50% compression; this checkpoint sits at the validated
  ceiling, not beyond it.
- Expect **factual-recall** regression before reasoning or coding regression. That is the
  measured failure mode of expert pruning on this architecture family: the closest published
  analogue (`cerebras/Kimi-Linear-REAP-35B-A3B`, same KDA + full-attention stack) loses 3.4
  points on FRAMES at only 30% pruning while code and maths hold flat.
- Healing (v2, repo root) is an output-scale correction applied exactly to the F32 block scales, median gain **0.9096**. It was **measured**, not derived: post-prune routing is replayed from a cached router-score trace, because the first-moment estimator that pass 1 used ignores that `norm_topk_prob` renormalises the surviving top-8 and over-corrects by ~30%. It is *not* distillation and does not recover lost knowledge.
- A **per-expert** least-squares refinement of this correction was built, shipped, and then **reverted**. It reduced held-out reconstruction residual in 41 of 42 layers and was still **11.8σ worse end-to-end** (top-1 agreement 0.83693 vs the scalar's 0.84249, worse on every metric and in every sufficiently-sampled domain). Under the measured near-orthogonality the coefficient reduces to `c_j = (gate mass before pruning) / (gate mass after)`, so it suppresses exactly the experts the *pruned* router leans on hardest - 62% of coefficients landed below their layer scalar, the worst by 3.3×. Lower reconstruction error, worse model. Recorded here because the technique is more attractive on paper than in measurement; see the repository's `research/HEALING_ABLATION.md`.
- Routing is disrupted more than expert count suggests: the retained experts carry ~0.99x the
  routing mass an average expert would, because REAP preserves rare-but-strong experts over
  common-but-weak ones.

## Serving on Jetson Thor

Use the **cutlass** fused-MoE backend (the Marlin FP4 MoE kernel faults at >=256 experts) and
`TRITON_MLA` for the 11 MLA+DSA layers (FLASHINFER is invalid for MLA).

<!-- CORRECTIONS:BEGIN -->
## Corrections to this repository

**2026-08-29 — the healing correction was replaced, and the weights changed.** If you downloaded
this checkpoint before this date you have the superseded version; re-pull to get the current one.

This checkpoint originally shipped a **per-expert** least-squares healing correction: one
coefficient per retained expert, fitted in closed form to reproduce the unpruned layer's output
under post-prune routing. It reduced held-out reconstruction residual in 41 of 42 layers, which is
why it shipped.

An end-to-end ablation then measured it against the **per-layer scalar** it had replaced — same
checkpoint, same 241,516 held-out tokens, same cached teacher, one variable moved:

| metric | per-expert (was shipped) | per-layer scalar (ships now) | Δ |
|---|---|---|---|
| top-1 agreement | 0.83693 | **0.84249** | **+0.00556** |
| ΔNLL vs teacher | 0.19396 | **0.17563** | −0.01833 |
| top-k KL | 0.69388 | **0.65030** | −0.04358 |

Those figures are measured on the **FP8 parent** this checkpoint is quantised from, which is where the healing correction lives. This repository's own measured numbers are in the evaluation section above; the NVFP4 build moved from 0.83294 to **0.83857** across the same correction.

Better in every sufficiently-sampled domain. Both arms score the identical token set, so the
correct test is paired: McNemar over the 13,020 discordant tokens (5,839 the per-expert arm gets
right and the scalar does not, 7,181 the other way) gives **χ² = 138.1, z = 11.8**. Healing is a
multiply on the F32 block scales and is exactly invertible, so the fix was a rescale, not a
re-prune — no expert was re-selected and the mask is unchanged.

**Why it lost.** Under the measured near-orthogonality of expert outputs (off-diagonal mass 1.9%)
the coefficient reduces to `c_j = (gate mass expert j received before pruning) / (gate mass it
receives after)`. The intended reading was that an expert promoted into the top-8 by pruning is
doing work it never did before and should be shrunk. But it is doing that work *because the expert
that used to do it was deleted* — damping the substitute does not bring the original back, it
leaves a hole. 62% of the 5,760 coefficients landed below their layer's scalar; 45 experts were
suppressed more than 2×, the worst by 3.3×, and by construction those are exactly the experts the
pruned router depends on most.

**A second correction follows from the same measurement.** Pass 1 shipped scalar healing and
pass 2 shipped per-expert, so the two were never comparable. Holding healing fixed, the pass-2
mask is worth **+0.00545** over pass 1 (McNemar z = 9.3) — not the ≈0 previously reported here. The two
changes were real effects of opposite sign that had cancelled into a null.

Full analysis, including why a held-out reconstruction proxy improved while the model got worse:
`research/HEALING_ABLATION.md` in the pipeline repository.
<!-- CORRECTIONS:END -->
