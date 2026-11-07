---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
new_version: patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - reap
  - moe
  - expert-pruning
  - glm5_next
  - fp8
---

# GLM-5.3-Flash-REAP50-FP8 (v1, superseded)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE)
[![Superseded by](https://img.shields.io/badge/superseded_by-FP8--v2-lightgrey)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)

The first published 50% expert-pruned version of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash),
kept in the upstream FP8 format. It is kept as a reproducible baseline; it is not the recommended version. Against the
unpruned model it keeps **top-1 agreement of 0.837** on 241,516 held-out tokens, scored teacher-forced.

**For new work, use [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2)**
(0.842, with the MTP block), or
[`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2) on a 128 GB device.

## Model overview

| | |
|---|---|
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash): MIT, 321.3B parameters by safetensors count, 305.8 GiB in FP8 |
| Architecture | `Glm5NextForConditionalGeneration` (`glm5_next`). 45 decoder layers: 34 KDA linear-attention layers and 11 MLA layers with DSA sparse attention. Layers 0–2 are dense MLP, layers 3–44 are MoE. Hidden size 4096. Includes a 24-block vision tower. |
| Parameters | 161.7B after pruning, by safetensors count. Active parameters were not computed. |
| Routed experts | 144 of 288 kept in every MoE layer. Top-8 routing and 1 shared expert, both unchanged. |
| MTP block | **Not included.** There are no layer-45 tensors, and `num_nextn_predict_layers` is 0. |
| Context length | 1,048,576 (`max_position_embeddings`, unchanged). Calibration used 2,048-token sequences. |
| Precision | FP8 E4M3 with 128×128 block scales (F32 `weight_scale_inv`), as upstream, for these tensors: routed and shared experts, the MLA projections, and dense layers 0–2. BF16, as upstream, for KDA attention, the DSA indexer, routers, embeddings, `lm_head`, the vision tower and the hyper-connection weights. |
| Size on disk | 157.0 GiB (168,549,039,624 bytes, 62 safetensors shards) |
| Licence | MIT, inherited from the base model |

## Which version should I use

Three expert-selection passes exist: v1 (this repo), v2 and v3. Each is a different set of 144 experts per layer. The v1
and v2 masks agree on about 91% of experts. Use **v2**: it is the only selection that has been both evaluated and fixed
after evaluation.

| Weights | Location | Selection | Healing | Size | Top-1 vs unpruned | Use |
|---|---|---|---|---|---|---|
| v2 FP8 | [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2), root | pass-2 REAP | measured per-layer scalar | 160.6 GiB | 0.8425 | **Recommended master, when memory allows** |
| v2 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2), root | pass-2 REAP | measured per-layer scalar | 98.2 GiB | 0.8386 | **Recommended for 128 GB devices** |
| v3 FP8 | [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2), `pass3-hope/` | HOPE, `protect_frac` 0.04 | first-moment fallback | 160.6 GiB | not evaluated | Experimental |
| v3 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2), `pass3-hope/` | HOPE, `protect_frac` 0.04 | first-moment fallback | 93.6 GiB | not evaluated | Experimental |
| **v1 FP8** | **this repo** | pass-1 REAP | measured scalar (re-fitted 2026-08-28) | 157.0 GiB | **0.8370** | Superseded; no MTP block |
| v1 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4) | pass-1 REAP | superseded first-moment gain | 95.9 GiB | not evaluated | Superseded; do not use |

GGUF builds for llama.cpp, made from the v2 FP8 weights, are in [`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF);
see that card for their details.

## Quickstart

```bash
hf download patrickbdevaney/GLM-5.3-Flash-REAP50-FP8 --local-dir ./glm53-reap50-fp8-v1
```

**Inference: untested.** This checkpoint has not been served by any runtime. It has been loaded only layer by layer, by
the pipeline's streaming evaluation harness. Loading it with stock `transformers`, vLLM or SGLang has not been tested,
so no serve command is given. The repo has no `preprocessor_config.json` or `processor_config.json`; for image input,
take them from [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2).

## Evaluation

**Method.** Paired teacher-forced comparison against the unpruned `zai-org/GLM-5.3-Flash` (FP8) on identical inputs:
241,516 held-out tokens that calibration never saw. Of the 338,113 positions captured, 96,597 image-placeholder positions
are excluded. Top-k KL uses the teacher's top 32 logits. The scored weights are this repo's weights after the
2026-08-28 re-fit. The source is `artifacts/eval/eval_glm-5.3-flash-reap50-fp8.json` in the pipeline repo
([`glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap); the artifact is local and not in the public repo).

| Metric | This repo (v1 FP8) | v2 FP8 | v2 NVFP4 |
|---|---|---|---|
| **Top-1 agreement** | **0.8370** | 0.8425 | 0.8386 |
| ΔNLL, mean (student − teacher) | +0.1979 | +0.1756 | +0.1835 |
| ΔNLL, median | +0.0015 | +0.0012 | +0.0017 |
| Top-32 KL (teacher ‖ student) | 0.6948 | 0.6503 | 0.6725 |
| Teacher NLL / student NLL | 1.018 / 1.216 | 1.018 / 1.194 | 1.018 / 1.201 |

Per domain, for this repo:

| Domain | Tokens | Top-1 agreement | ΔNLL |
|---|---|---|---|
| agentic | 69,202 | 0.863 | +0.182 |
| math | 66,558 | 0.919 | +0.026 |
| science | 45,034 | 0.829 | +0.138 |
| finance | 22,215 | 0.741 | +0.220 |
| general ("ballast") | 22,154 | 0.572 | +1.021 |
| code | 15,821 | 0.921 | +0.048 |
| vision (text after images) | 532 | too few tokens to report | — |

With healing held at the per-layer scalar, the v2 mask beats this v1 mask by **+0.00545 top-1** on the same tokens
(McNemar z = 9.3; `artifacts/eval/mask_paired_test.json`).

**What this does and does not show.** Teacher-forced agreement measures how far the pruned model's next-token
predictions moved from the unpruned model's on ground-truth text. It is not a capability benchmark. Vision quality is
unmeasured. No generative benchmark has been run on this checkpoint.

## Hardware and memory

The build machine is an NVIDIA Jetson AGX Thor (128 GB unified memory, `sm_110a`). This checkpoint is larger than that
memory, so it ran there only through the layer-streaming evaluation harness. It has not been run on any other hardware
and has not been served.

## How it was made

1. **Source.** The upstream FP8 release. Pruning deletes whole expert tensors and slices the router rows. The retained
   FP8 weight values are copied unchanged; healing changes only their F32 block scales.
2. **Calibration.** 256 samples, about 0.52M tokens, drawn from a permissively licensed multi-domain mixture that
   included image-text pairs. That is roughly a tenth of v2's 5.5M-token budget.
3. **Selection.** REAP (router-weighted expert activation pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999))
   kept 144 of 288 experts in each of the 42 MoE decoder layers, retaining 0.643 of expert saliency mass, against 0.50
   for random selection at the same ratio. The MTP block (layer 45) was left out.
4. **Healing.** As first uploaded, each layer's expert outputs were scaled by a first-moment gain (median 0.69). That
   estimator ignores `norm_topk_prob` renormalisation; measured gate mass is 2.5000 both before and after pruning. On
   2026-08-28 the gain was re-measured by replaying post-prune routing from a cached router-score trace (median 0.9111,
   per-layer 0.846–0.950). The block scales were multiplied by `measured / shipped` (median 1.3121, range
   1.0719–1.5260). Only F32 scales changed; no weight was re-quantised.

`adapters/first_moment_gains.json` in this repo still lists the original first-moment gains, not the re-fitted ones.

## Limitations

- **Superseded.** The v2 mask is measurably better (+0.00545 top-1, z = 9.3), and v2 keeps the MTP block.
- **No MTP block**, so native multi-token-prediction drafting is not possible with these weights.
- **The cost of pruning is uneven across domains.** General-knowledge text is the worst-preserved domain (top-1
  agreement 0.572, against 0.92 for code and math).
- **Long context is not calibrated** (2,048-token sequences). The evaluation is teacher-forced only, and vision is
  unmeasured.
- **No tested runtime.**

## Changelog and corrections

- **2026-08-28**: published. Later that day the healing gain was re-fitted (first-moment median 0.69 → measured 0.9111)
  and the weights were re-uploaded. If you downloaded before the re-fit, download again.
- **2026-08-28**: the paired evaluation first reported top-1 0.674, because image-placeholder positions were scored.
  With them excluded, the result is the 0.8370 above.
- **2026-11-06**: card rewritten and verified against repo files and build artifacts; a 2026-10-26 automated rewrite
  contained errors.

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
  howpublished = {\url{https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8}}
}
```
