---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - reap
  - moe
  - expert-pruning
  - glm5_next
  - fp8
---

# GLM-5.3-Flash-REAP50-FP8-v2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE)
[![Base model](https://img.shields.io/badge/base-zai--org%2FGLM--5.3--Flash-blue)](https://huggingface.co/zai-org/GLM-5.3-Flash)

A 50% expert-pruned version of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), kept in the
upstream FP8 format. It is the master checkpoint from which the NVFP4 and GGUF siblings are built, and it is the
highest-fidelity version of the pruned model. Against the unpruned model it keeps **top-1 agreement of 0.842** on
241,516 held-out tokens, scored teacher-forced.

The files at the repo root are **v2**, the recommended version. The `pass3-hope/` subfolder holds an experimental
**v3** that has not been evaluated; see [Which version should I use](#which-version-should-i-use). At 160.6 GiB, this
checkpoint does not fit on a 128 GB device. For that, use
[`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2).

## Model overview

| | |
|---|---|
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash): MIT, 321.3B parameters by safetensors count, 305.8 GiB in FP8 |
| Architecture | `Glm5NextForConditionalGeneration` (`glm5_next`). 45 decoder layers: 34 KDA linear-attention layers and 11 MLA layers with DSA sparse attention. Layers 0–2 are dense MLP, layers 3–44 are MoE. Hidden size 4096. Includes a 24-block vision tower. |
| Parameters | 165.5B after pruning, by safetensors count. Active parameters were not computed. |
| Routed experts | 144 of 288 kept in every MoE layer. Top-8 routing and 1 shared expert, both unchanged. |
| MTP block | Present as layer 45, pruned to 144 experts (`num_nextn_predict_layers: 1`) |
| Context length | 1,048,576 (`max_position_embeddings`, unchanged). Calibration used 2,048-token sequences. |
| Precision | FP8 E4M3 with 128×128 block scales (F32 `weight_scale_inv`), as upstream, for these tensors: routed and shared experts (including the MTP block's), the MLA projections, and dense layers 0–2. BF16, as upstream, for KDA attention, the DSA indexer, routers, embeddings, `lm_head`, the vision tower and the hyper-connection weights. |
| Size on disk | 160.6 GiB (172,416,620,840 bytes, 62 safetensors shards) |
| Licence | MIT, inherited from the base model |

## Which version should I use

Three expert-selection passes exist: v1, v2 and v3. Each is a different set of 144 experts per layer. Use **v2**: it is
the only selection that has been both evaluated and fixed after evaluation.

| Weights | Location | Selection | Healing | Size | Top-1 vs unpruned | Use |
|---|---|---|---|---|---|---|
| **v2 FP8** | **this repo, root** | pass-2 REAP | measured per-layer scalar | 160.6 GiB | **0.8425** | **Recommended master, when memory allows** |
| v2 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2), root | pass-2 REAP | measured per-layer scalar | 98.2 GiB | 0.8386 | Recommended for 128 GB devices |
| v3 FP8 | this repo, `pass3-hope/` | HOPE, `protect_frac` 0.04 | first-moment fallback | 160.6 GiB | not evaluated | Experimental |
| v3 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2), `pass3-hope/` | HOPE, `protect_frac` 0.04 | first-moment fallback | 93.6 GiB | not evaluated | Experimental |
| v1 FP8 | [`GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8) | pass-1 REAP | measured scalar (re-fitted 2026-08-28) | 157.0 GiB | 0.8370 | Superseded; no MTP block |
| v1 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4) | pass-1 REAP | superseded first-moment gain | 95.9 GiB | not evaluated | Superseded; do not use |

GGUF builds for llama.cpp, made from the v2 FP8 weights, are in [`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF);
see that card for their details.

The `-v2` in this repo's name dates from the pass-2 release. To tell the versions apart, go by the path: the root is v2
and `pass3-hope/` is v3.

### About v3 (`pass3-hope/`)

v3 selects experts with HOPE, which minimises the output error of the whole prune set, including interactions between
experts. v2 ranks each expert on its own. v3 also reserves a domain floor (`protect_frac = 0.04`). That is the design
intent; it has not been shown to help. Treat v3 as experimental:

1. **Healing used the fallback estimator.** The measured gains on file had been fitted to the v2 keep-set. The check
   that should have stopped the run only logged a warning, so v3 was healed with the first-moment per-layer scalar
   (median 0.766, range 0.667–0.853, 42 layers) and no per-expert coefficients. On v2 data this estimator gave 0.7149
   where router replay measured 0.9111, an over-correction of about 30%.
2. **It has not been evaluated.** No teacher-forced or generative evaluation has been run on v3.

The v3 FP8 weights keep the same precision layout as v2. Only the NVFP4 v3 build also quantises the KDA q/k/v
projections. The v3 FP8 is the same size as v2 because both keep 144 equally sized experts per layer, but they are
different weights: 59 of the 62 shards differ. The `pass3-hope/` folder has no `preprocessor_config.json` or
`processor_config.json`; copy them from the repo root for image input.

## Quickstart

**Download.**

```bash
# v2 (recommended): repo root only
hf download patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2 \
    --exclude "pass3-hope/*" --local-dir ./glm53-reap50-fp8-v2

# v3 (experimental, unevaluated)
hf download patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2 \
    --include "pass3-hope/*" --local-dir ./glm53-reap50-fp8-v3
```

**Inference: untested.** This checkpoint has not been served by any runtime. It has been loaded only layer by layer, by
the pipeline's streaming evaluation harness. Loading it with stock `transformers`, vLLM or SGLang has not been tested,
so no serve command is given. For a version that has been served, see
[`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2).

## Evaluation

**Method.** Paired teacher-forced comparison against the unpruned `zai-org/GLM-5.3-Flash` (FP8) on identical inputs:
241,516 held-out tokens that calibration never saw. Of the 338,113 positions captured, 96,597 image-placeholder positions
are excluded, as the training objective excludes them. Top-k KL uses the teacher's top 32 logits. The source is
`artifacts/eval/eval_glm-5.3-flash-reap50-fp8-pass2.json` in the pipeline repo
([`glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap); the artifact is local and not in the public repo).

| Metric | This repo (v2 FP8) | v2 NVFP4 | v1 FP8 |
|---|---|---|---|
| **Top-1 agreement** | **0.8425** | 0.8386 | 0.8370 |
| ΔNLL, mean (student − teacher) | +0.1756 | +0.1835 | +0.1979 |
| ΔNLL, median | +0.0012 | +0.0017 | +0.0015 |
| Top-32 KL (teacher ‖ student) | 0.6503 | 0.6725 | 0.6948 |
| Teacher NLL / student NLL | 1.018 / 1.194 | 1.018 / 1.201 | 1.018 / 1.216 |

Per domain, for this repo:

| Domain | Tokens | Top-1 agreement | ΔNLL |
|---|---|---|---|
| agentic | 69,202 | 0.873 | +0.137 |
| math | 66,558 | 0.920 | +0.022 |
| science | 45,034 | 0.830 | +0.125 |
| finance | 22,215 | 0.755 | +0.181 |
| general ("ballast") | 22,154 | 0.580 | +0.989 |
| code | 15,821 | 0.919 | +0.052 |
| vision (text after images) | 532 | too few tokens to report | — |

**Paired tests** (same tokens; `artifacts/eval/mask_paired_test.json` and `healing_paired_test.json`):

| Comparison | Δ top-1 | McNemar |
|---|---|---|
| v2 mask vs v1 mask, healing held at the per-layer scalar | +0.00545 | z = 9.3 |
| Per-layer scalar heal (shipped) vs per-expert heal (reverted) | +0.00556 | z = 11.8 |

**What this does and does not show.** Teacher-forced agreement measures how far the pruned model's next-token
predictions moved from the unpruned model's on ground-truth text. It is not a capability benchmark. Vision quality is
unmeasured: after placeholders are excluded, only 532 text tokens remain in the vision bucket. No generative benchmark
has been run on this checkpoint.

## Hardware and memory

The build machine is an NVIDIA Jetson AGX Thor (128 GB unified memory, `sm_110a`). This checkpoint is larger than that
memory, so it ran there only through the layer-streaming evaluation harness. It has not been run on any other hardware
and has not been served. Speed and memory measurements exist only for the NVFP4 sibling.

## How it was made

1. **Source.** The upstream FP8 release, which stores each routed expert as its own tensors with 128×128 block scales.
   Pruning deletes whole expert tensors and slices the router rows. The retained FP8 weight values are copied
   unchanged; healing changes only their F32 block scales.
2. **Calibration.** 5,500,928 tokens (2,686 samples) across seven domains: agentic, code, math, science, finance,
   general, and real image-text pairs. Image tokens route through the same experts as text. The mixture was set in
   samples but acts in tokens; measured early in the pass, the token shares were agentic 31.6%, math 29.9%,
   science 21.3%, multimodal 6.2%, code 5.7%, general 4.9% and finance 0.5%. Only permissively licensed sources were
   used. Biomedical sources were excluded on licence or clinical-scope grounds, so there is no biomedical calibration
   data.
3. **Selection.** REAP (router-weighted expert activation pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999))
   kept 144 of 288 experts in each of the 42 MoE decoder layers. MTP layer 45 is not run by `transformers`, so it has no
   activation saliency; its experts were ranked by weight norm.
4. **Healing.** One output-scale gain per layer, multiplied into the experts' F32 block scales. The gain was measured by
   replaying post-prune routing from a cached router-score trace (median about 0.91, range 0.834–0.950). The cheaper
   first-moment estimate over-corrects by about 30% here, because it ignores the `norm_topk_prob` renormalisation of
   the surviving top-8. A per-expert refinement was shipped on 2026-08-29 and reverted the same day; see the
   changelog. Healing is a rescale, not retraining.

Method write-ups in the pipeline repo: [healing](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/70-healing.md),
[evaluation](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-evaluation.md),
[healing ablation](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/research/HEALING_ABLATION.md),
[v3 / HOPE](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/31-pass3-hope-method.md).

**Side metadata files (corrected 2026-11-06).** The weights and `config.json` are authoritative. Each corrected file
carries a `_note` key explaining the change:

- Root `adapters/first_moment_gains.json` records the per-expert heal that was reverted on 2026-08-29. It is kept for the
  record and marked as not applied; the weights carry the per-layer scalar.
- Root `reap_metadata.json` now says `mtp_preserved: true` (it previously said `false`), matching `config.json` and the
  layer-45 tensors.
- `pass3-hope/reap_metadata.json` no longer repeats v2's figures. It gives the v3 heal as the first-moment scalar
  (median 0.766, from `pass3-hope/adapters/first_moment_gains.json`), and sets the fields that were never recorded for v3
  to `null`.

## Limitations

- **The cost of pruning is uneven across domains.** General-knowledge text is the worst-preserved domain (top-1
  agreement 0.580, against 0.92 for code and math). Expect factual-recall regressions before reasoning or coding ones.
- **No biomedical calibration data.** Do not rely on this model for biomedical work.
- **Long context is not calibrated.** Calibration used 2,048-token sequences.
- **The evaluation is teacher-forced only**, and vision is unmeasured.
- **No tested runtime** for this FP8 checkpoint.
- **v3 (`pass3-hope/`) is unevaluated** and was healed with the fallback estimator.

## Changelog and corrections

- **2026-08-29**: v2 published. It first shipped with per-expert healing coefficients. An end-to-end ablation found them
  worse than the per-layer scalar (top-1 0.83693 vs 0.84249, z = 11.8), so the weights were re-uploaded the same day
  with the scalar. If you downloaded before that re-upload, download again.
- **2026-08-31**: `preprocessor_config.json` and `processor_config.json` added.
- **2026-10-01**: v3 added under `pass3-hope/`.
- **2026-10-12**: the healing column corrected to the per-layer scalar. The v1 → v2 comparison reconciled: first
  reported as no difference, it measures +0.00545 once healing is held fixed.
- **2026-10-24**: the card corrected to describe only measured results. The v3 description now covers the fallback
  healing and the missing evaluation. Earlier text saying the MTP block was excluded was wrong for v2 and v3.
- **2026-11-06**: card rewritten and verified against repo files and build artifacts; a 2026-10-26 automated rewrite
  contained errors.

No weights have changed since the 2026-08-29 re-upload (root) and the 2026-10-01 upload (`pass3-hope/`).

## License and attribution

Released under the MIT licence, the base model's licence (see `LICENSE`). The base model is
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by [Z.ai](https://huggingface.co/zai-org); all credit for
the architecture and pre-training belongs to its authors. The pruning criterion is REAP: Lasby et al., *REAP the
Experts: Why Pruning Prevails for One-Shot MoE Compression*, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999).

## Citation

```bibtex
@misc{devaney2026glm53reap50,
  title        = {GLM-5.3-Flash-REAP50: 50\% expert-pruned GLM-5.3-Flash in FP8 and NVFP4},
  author       = {Patrick Devaney},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2}}
}
```
