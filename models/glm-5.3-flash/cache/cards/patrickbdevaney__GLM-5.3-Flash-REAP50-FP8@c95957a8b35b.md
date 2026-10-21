---
license: mit
base_model: zai-org/GLM-5.3-Flash
tags: [reap, moe, pruning, glm5_next, jetson, thor]
---

# GLM-5.3-Flash REAP-50 (FP8)

> ## Corrected 2026-08-28 — healing scalar re-fit
>
> **This checkpoint has been revised.** The original upload applied a first-moment healing gain
> (median **0.6933**) to every retained expert's output. That estimator ignores that
> `norm_topk_prob` renormalises the surviving top-8, so the router already returns essentially all
> the pruned gate mass by itself — **measured: gate mass 2.5000 before pruning and 2.5000 after**.
> The derivation therefore prescribed a ~39% shrink where the true output inflation is ~10%.
>
> Re-measured by replaying post-prune routing from a cached router-score trace over 2.3M tokens,
> the correct gain is median **0.9111** (per-layer 0.846–0.950).
> Every layer's block scales have been multiplied by `measured / shipped`
> (median **1.3121**, range 1.0719–1.5260), correcting a systematic
> under-scaling of the entire MoE pathway relative to attention, the shared experts and the
> residual stream.
>
> The fix is exact and lossless: it scales the F32 `weight_scale_inv` tensors, never the FP8
> values, so no weight was requantised. 6,048 block scales were corrected — exactly
> 42 layers x 144 retained experts.
>
> **If you downloaded the earlier revision, re-pull.** The previous state remains in this repo's
> git history, and is recoverable by dividing the block scales by the factors above.
>
> Not yet propagated to the NVFP4 sibling repo, which still carries the original gain.


50% of routed experts removed with **REAP**
(Router-weighted Expert Activation Pruning, arXiv:2510.13999), calibrated on a
permissively-licensed multi-domain corpus that includes real image-text pairs.

| | |
|---|---|
| Base | `zai-org/GLM-5.3-Flash` (MIT, FP8 E4M3, 128x128 block scales) |
| Experts | 288 -> 144 per layer, top-8 routing unchanged |
| Size | 157.0 GiB (FP8) |
| Healed | yes — **re-fit 2026-08-28**, measured gain 0.9111 |
| MTP block | excluded (see below) |

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

## Evaluation status: NONE

**This checkpoint has not been evaluated.** No benchmark has been run against it - not coding,
not agentic, not vision, not knowledge. What has been verified is *structural*: expert counts
match the config, routers are sliced to the retained set, every tensor loads, the vision tower
is untouched, and the MTP block is cleanly absent.

The pruning itself measured **1.29x better than random** at retaining expert output
contribution (saliency mass 0.643 against 0.50 for random pruning at the same ratio). That says
the criterion selected well. It does **not** say the model is good.

Treat this as a research artifact pending evaluation, not a drop-in replacement.

## Known limitations

- **The MTP (multi-token-prediction) block at layer 45 is excluded.** `transformers`'
  `Glm5NextForConditionalGeneration` does not instantiate it, so the pruning path cannot see
  it. Its original tensors are archived unmodified rather than inconsistently pruned.
- REAP has no published data above 50% compression; this checkpoint sits at the validated
  ceiling, not beyond it.
- Expect **factual-recall** regression before reasoning or coding regression. That is the
  measured failure mode of expert pruning on this architecture family: the closest published
  analogue (`cerebras/Kimi-Linear-REAP-35B-A3B`, same KDA + full-attention stack) loses 3.4
  points on FRAMES at only 30% pruning while code and maths hold flat.
- Healing is a **first-moment output-scale correction** derived from the calibration saliency
  (median gain 0.696, applied exactly to the F32 block scales). It is *not* distillation and
  does not attempt to recover lost knowledge.
- Routing is disrupted more than expert count suggests: the retained experts carry ~0.90x the
  routing mass an average expert would, because REAP preserves rare-but-strong experts over
  common-but-weak ones.

## Serving on Jetson Thor

Use the **cutlass** fused-MoE backend (the Marlin FP4 MoE kernel faults at >=256 experts) and
`TRITON_MLA` for the 11 MLA+DSA layers (FLASHINFER is invalid for MLA).
