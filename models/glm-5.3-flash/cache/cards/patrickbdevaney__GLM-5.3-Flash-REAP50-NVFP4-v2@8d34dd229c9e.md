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
  - nvfp4
  - compressed-tensors
  - jetson-thor
---

# GLM-5.3-Flash-REAP50-NVFP4-v2

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE)
[![Base model](https://img.shields.io/badge/base-zai--org%2FGLM--5.3--Flash-blue)](https://huggingface.co/zai-org/GLM-5.3-Flash)

A 50% expert-pruned version of [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash), with the
routed and shared experts stored as weight-only NVFP4. At 98.2 GiB it is the version of this model built to fit a single
128 GB device. It was built and tested on an NVIDIA Jetson AGX Thor. Against the unpruned model it keeps
**top-1 agreement of 0.839** on 241,516 held-out tokens, scored teacher-forced.

> **Evaluation update (2026-11).** In paired generation against the unpruned model, this checkpoint keeps 94.8% of the HumanEval+ score, but it hits the token cap on 44–72% of general-knowledge and instruction prompts. For example, MMLU-Pro is 44 against 84 out of 100. For general use on a 128 GB machine, the unpruned [`UD-Q2_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) GGUF (101 GiB) measured better. See [Evaluation findings (2026-11)](#evaluation-findings-2026-11).

The files at the repo root are **v2**, the recommended version. The `pass3-hope/` subfolder holds an experimental
**v3** that has not been evaluated; see [Which version should I use](#which-version-should-i-use).

## Model overview

| | |
|---|---|
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash): MIT, 321.3B parameters by safetensors count, 305.8 GiB in FP8 |
| Architecture | `Glm5NextForConditionalGeneration` (`glm5_next`). 45 decoder layers: 34 KDA linear-attention layers and 11 MLA layers with DSA sparse attention. Layers 0–2 are dense MLP, layers 3–44 are MoE. Hidden size 4096. Includes a 24-block vision tower. |
| Parameters | 165.5B after pruning. This is the element count of the FP8 sibling, which has the same tensors; the Hub's count for this repo counts packed bytes, not weights. Active parameters were not computed. |
| Routed experts | 144 of 288 kept in every MoE layer. Top-8 routing and 1 shared expert, both unchanged. |
| MTP block | Present as layer 45, pruned to 144 experts (`num_nextn_predict_layers: 1`) |
| Context length | 1,048,576 (`max_position_embeddings`, unchanged). Calibration used 2,048-token sequences. |
| Precision | Routed and shared experts, including the MTP block's, are NVFP4: weight-only, group size 16, FP8 E4M3 group scales and an F32 global scale. Everything else is BF16: KDA and MLA attention, the DSA indexer, dense layers 0–2, routers, embeddings, `lm_head`, the vision tower and the hyper-connection weights. Small state tensors are F32. There are no activation scales and no `input_scale` tensors. |
| Format | `compressed-tensors`, `nvfp4-pack-quantized` |
| Size on disk | 98.2 GiB (105,393,931,276 bytes, 62 safetensors shards) |
| Licence | MIT, inherited from the base model |

## Which version should I use

Three expert-selection passes exist: v1, v2 and v3. Each is a different set of 144 experts per layer. Use **v2**: it is
the only selection that has been both evaluated and fixed after evaluation.

| Weights | Location | Selection | Healing | Size | Top-1 vs unpruned | Use |
|---|---|---|---|---|---|---|
| **v2 NVFP4** | **this repo, root** | pass-2 REAP | measured per-layer scalar | 98.2 GiB | **0.8386** | **Best pruned build for 128 GB devices; code-focused use (see findings)** |
| v2 FP8 | [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2), root | pass-2 REAP | measured per-layer scalar | 160.6 GiB | 0.8425 | Recommended master, when memory allows |
| v3 NVFP4 | this repo, `pass3-hope/` | HOPE, `protect_frac` 0.04 | first-moment fallback | 93.6 GiB | not evaluated | Experimental |
| v3 FP8 | [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2), `pass3-hope/` | HOPE, `protect_frac` 0.04 | first-moment fallback | 160.6 GiB | not evaluated | Experimental |
| v1 FP8 | [`GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8) | pass-1 REAP | measured scalar (re-fitted 2026-08-28) | 157.0 GiB | 0.8370 | Superseded; no MTP block |
| v1 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4) | pass-1 REAP | superseded first-moment gain | 95.9 GiB | not evaluated | Superseded; do not use |

GGUF builds for llama.cpp, made from the v2 FP8 weights, are in [`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF);
see that card for their details.

The `-v2` in this repo's name dates from the pass-2 release. To tell the versions apart, go by the path: the root is v2
and `pass3-hope/` is v3.

### About v3 (`pass3-hope/`)

v3 selects experts with HOPE, which minimises the output error of the whole prune set, including interactions between
experts. v2 ranks each expert on its own. v3 also reserves a domain floor (`protect_frac = 0.04`). That is the design
intent; it has not been shown to help. Two build defects mean you should not treat v3 as an upgrade:

1. **Healing used the fallback estimator.** The measured gains on file had been fitted to the v2 keep-set. The check
   that should have stopped the run only logged a warning, so v3 was healed with the first-moment per-layer scalar
   (median 0.766, range 0.667–0.853, 42 layers) and no per-expert coefficients. On v2 data this estimator gave 0.7149
   where router replay measured 0.9111, an over-correction of about 30%.
2. **The v3 NVFP4 build also quantises the KDA q/k/v projections** (102 tensors) to NVFP4, which is why it is 4.6 GiB
   smaller. On the v2 mask, quantising exactly these tensors cost **−0.0023 top-1** (McNemar z = 5.8). For that reason
   it was not shipped for v2.

v3 has had no teacher-forced or generative evaluation, so nothing is known about whether it is better or worse than
v2. The `pass3-hope/` folder has no `preprocessor_config.json` or `processor_config.json`; copy them from the repo root
for image input.

## Quickstart

**Download.**

```bash
# v2 (recommended): repo root only
hf download patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
    --exclude "pass3-hope/*" --local-dir ./glm53-reap50-nvfp4-v2

# v3 (experimental, unevaluated)
hf download patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2 \
    --include "pass3-hope/*" --local-dir ./glm53-reap50-nvfp4-v3
```

**Serve on Jetson AGX Thor.** These weights have been served only by the author's pure-CUDA engine,
[`glm-5.3-flash-cuda-server`](https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server), which targets `sm_110a`
and supports only the v2 NVFP4 weights. Its README covers the build requirements and the optional dense-weight overlay.
In outline:

```bash
git clone https://github.com/patrickbdevaney/glm-5.3-flash-cuda-server && cd glm-5.3-flash-cuda-server
bash scripts/build.sh
scripts/run_full.sh --ckpt ./glm53-reap50-nvfp4-v2 --ctx 32768 --port 8080   # OpenAI-compatible API
```

Set `reasoning_effort` to `low` or `high` in each request. The server documents that at the default (Max), HumanEval
generations ran past 8,192 tokens without terminating.

**Not tested:** loading with stock `transformers`, vLLM or SGLang on any data-center GPU. No serve command for those
runtimes is given here because none has been run. A stock vLLM 0.32 nightly image on Jetson AGX Thor stopped before
loading weights: it has no sparse-MLA attention backend for `sm_110`. The 2026-11-14 `config.json` change (see the
changelog) adds the `quantization_config.ignore` entries that compressed-tensors loaders need for the BF16 dense MLP
and the MTP `eh_proj`. Whether vLLM then loads and serves the model correctly is unverified.

## Evaluation findings (2026-11)

**Summary.** In paired generation against the unpruned GLM-5.3-Flash, this checkpoint holds up on code but fails to finish on
general-knowledge and instruction prompts. The answers it does finish are close to the unpruned model's. The deficit
comes from hitting the token cap, often in a repetition loop. The evidence points to the expert prune, which kept too little
general-text capacity: general-text top-1 agreement is 0.58, against 0.92 on code, in teacher-forced scoring outside
the serving engine. An unpruned 2-bit quant run through the same harness does not loop.

| Suite (effort, token cap) | Paired items | Unpruned model | REAP50-NVFP4-v2 | Retention | McNemar p |
|---|---:|---:|---:|---:|---:|
| HumanEval+ (high, 8,192) | 164 | 153 (93.3%) | 145 (88.4%) | 94.8% | 0.057 |
| AIME 2026 (max, 16,384) | 30 | 17 (56.7%) | 9 (30.0%) | 52.9% | 0.0078 |
| IFBench strict (high, 4,096) | 100 | 56 (56.0%) | 32 (32.0%) | 57.1% | 3.9e-4 |
| MMLU-Pro (high, 4,096) | 100 | 84 (84.0%) | 44 (44.0%) | 52.4% | 1.1e-10 |
| GPQA-Diamond (high, 8,192) | 75 | 51 (68.0%) | 19 (25.3%) | 37.3% | 1.9e-8 |

| Suite | Hit the token cap (unpruned / prune) | Accuracy when it finished (unpruned / prune) | Traces with a repetition loop (unpruned / prune) |
|---|---|---|---|
| HumanEval+ | 0% / 2.4% | 93.3% / 90.6% | 0 / 0 |
| AIME 2026 | 43% / 70% | 100% / 100% (17 and 9 items) | 0 / 0 |
| IFBench | 7% / 44% | 59.1% / 55.4% | 0 / 21 |
| MMLU-Pro | 7% / 45% | 90.3% / 80.0% | 0 / 9 |
| GPQA-Diamond | 28% / 72% | 94.4% / 90.5% | 0 / 18 |

Same prompts, items, sampling (T 1.0, top_p 0.95, seed 0), reasoning effort and token caps for both models. The unpruned model was Z.ai's FP8 API endpoint; `REAP50-NVFP4-v2` (revision `fb45d356`, weights identical to the current revision) ran on `glm-5.3-flash-cuda-server` on a Jetson AGX Thor with the dense overlay off. A trace counts as a loop when a 30-word sequence repeats at least 20 times. Truncated items score as failures. Per-item data, scripts and the full method are in [the findings repo](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-prune-vs-quant-matched-bytes).

**Matched-bytes comparison.** On the same items, Unsloth's *unpruned* [`UD-Q2_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) GGUF (101.2 GiB) is not significantly different from the unpruned model on any of the five suites (paired McNemar p ≥ 0.18; IFBench 59, MMLU-Pro 85, GPQA 59/100, HumanEval+ 148, AIME 13) and never loops. Teacher-forced on 243,067 held-out tokens, it agrees with the unpruned model on 87.25% of top-1 tokens. A 25%-pruned GGUF at the same size reaches 85.44%, and a 33%-pruned one 83.00%. At this byte budget, keeping every expert and quantizing harder beats removing experts.

**What this means in practice:**
- **Code-focused use:** the v2 mask is reasonable. The NVFP4 build kept 94.8% of the unpruned HumanEval+ score, with no loops.
- **General assistant, instruction-following or knowledge work on a 128 GB machine:** use the unpruned
  [`UD-Q2_K_XL`](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) GGUF instead.
- **If you serve this checkpoint anyway:** set a token cap and treat a capped reply as a failure. In an early
  smoke run, one looping trace ran to a 30,000-token cap without stopping, so a larger cap is unlikely to help.

Full results, per-item data and scripts: [GLM-5.3-Flash-prune-vs-quant-matched-bytes](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-prune-vs-quant-matched-bytes).

## Evaluation (teacher-forced)

**Method.** Paired teacher-forced comparison against the unpruned `zai-org/GLM-5.3-Flash` (FP8) on identical inputs:
241,516 held-out tokens that calibration never saw. Of the 338,113 positions captured, 96,597 image-placeholder positions
are excluded, as the training objective excludes them. Top-k KL uses the teacher's top 32 logits. The source is
`artifacts/eval/eval_glm-5.3-flash-reap50-nvfp4-pass2.json` in the pipeline repo
([`glm-5.3-reap`](https://github.com/patrickbdevaney/glm-5.3-reap); the artifact is local and not in the public repo).

| Metric | This repo (v2 NVFP4) | v2 FP8 parent | v1 FP8 |
|---|---|---|---|
| **Top-1 agreement** | **0.8386** | 0.8425 | 0.8370 |
| ΔNLL, mean (student − teacher) | +0.1835 | +0.1756 | +0.1979 |
| ΔNLL, median | +0.0017 | +0.0012 | +0.0015 |
| Top-32 KL (teacher ‖ student) | 0.6725 | 0.6503 | 0.6948 |
| Teacher NLL / student NLL | 1.018 / 1.201 | 1.018 / 1.194 | 1.018 / 1.216 |

Per domain, for this repo:

| Domain | Tokens | Top-1 agreement | ΔNLL |
|---|---|---|---|
| agentic | 69,202 | 0.869 | +0.153 |
| math | 66,558 | 0.916 | +0.024 |
| science | 45,034 | 0.827 | +0.129 |
| finance | 22,215 | 0.748 | +0.191 |
| general ("ballast") | 22,154 | 0.580 | +0.996 |
| code | 15,821 | 0.917 | +0.055 |
| vision (text after images) | 532 | too few tokens to report | — |

**Paired tests** (same tokens; artifacts in `artifacts/eval/`):

- With healing held at the per-layer scalar, the v2 mask beats the v1 mask by **+0.00545 top-1** (McNemar z = 9.3, FP8).
- The per-layer scalar heal beats the per-expert heal it replaced by **+0.00556 top-1** (z = 11.8, FP8).
- Quantising more of attention to NVFP4 on the v2 mask: KDA q/k/v −0.0023 (z = 5.8); `o_proj` plus the MLA
  projections −0.0055 (z = 12.3); both −0.0067 (z = 14.5). None of these builds was shipped at the root.

**What this does and does not show.** Teacher-forced agreement measures how far the pruned model's next-token
predictions moved from the unpruned model's on ground-truth text. It is not a capability benchmark. Vision quality is
unmeasured: after placeholders are excluded, only 532 text tokens remain in the vision bucket. The paired generative comparison with the
unpruned model is in [Evaluation findings (2026-11)](#evaluation-findings-2026-11).

**Served-engine checks.** These were run through the CUDA server above with its local NVFP4 dense-weight overlay
switched on. The overlay is not part of this repo, and against the BF16 dense path it measured +1.56% perplexity over
47,195 tokens. These checks show that the served model works; they are not a quality claim for this checkpoint:

| Check | Result | Settings |
|---|---|---|
| HumanEval pass@1 | 150/164 (91.5%) | `reasoning_effort=high`, 4,096-token budget, 7 truncated |
| BFCL `live_multiple` | 73/100 | `reasoning_effort=high`, 2,048-token budget, 4 truncated |
| Needle in a haystack | 9/9 found | 32K, 64K and 128K tokens × depths 0.1 / 0.5 / 0.9 |

The truncated runs hit their token budget, so those scores are lower bounds.

## Hardware and memory

These weights have been run only on an **NVIDIA Jetson AGX Thor** (128 GB unified memory, `sm_110a`). Two kinds of run
were made: the teacher-forced evaluation above, which streams layers through the pipeline's harness, and serving through
`glm-5.3-flash-cuda-server`. The following was measured with that server, at batch size 1 and with a single stream:

| | Measured |
|---|---|
| Decode, checkpoint weights only | 8.36 tok/s |
| Decode, with the local dense NVFP4 overlay | 11.86–11.91 tok/s |
| Resident memory, with overlay, 16K context allocated | 101.85 GiB |
| Native MTP speculative decoding | net slowdown (best 0.95×), so disabled |

From the server's documentation (estimated, not measured): only the 11 MLA layers keep a KV cache. A stream costs a
fixed 145.56 MiB of KDA state plus about 22.5 KB of latent cache per token (11 layers × 2,048 bytes), which is about 2.9
GiB at 128K context.

## How it was made

1. **Source.** The upstream FP8 release, which stores each routed expert as its own tensors with 128×128 block scales.
   Pruning deletes whole expert tensors and slices the router rows, so retained weights are not changed.
2. **Calibration.** 5,500,928 tokens (2,686 samples) across seven domains: agentic, code, math, science, finance,
   general, and real image-text pairs. Image tokens route through the same experts as text. The mixture was set in
   samples but acts in tokens; measured early in the pass, the token shares were agentic 31.6%, math 29.9%,
   science 21.3%, multimodal 6.2%, code 5.7%, general 4.9% and finance 0.5%. Only permissively licensed sources were
   used. Biomedical sources were excluded on licence or clinical-scope grounds, so there is no biomedical calibration
   data.
3. **Selection.** REAP (router-weighted expert activation pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999))
   kept 144 of 288 experts in each of the 42 MoE decoder layers. The router gate was read without
   `e_score_correction_bias`. MTP layer 45 is not run by `transformers`, so it has no activation saliency; its experts
   were ranked by weight norm.
4. **Healing.** One output-scale gain per layer, applied to the expert block scales. The gain was measured by replaying
   post-prune routing from a cached router-score trace (median about 0.91, range 0.834–0.950). The cheaper first-moment
   estimate over-corrects by about 30% here, because it ignores the `norm_topk_prob` renormalisation of the surviving
   top-8. A per-expert refinement was shipped on 2026-08-29 and reverted the same day; see the changelog. Healing is a
   rescale, not retraining.
5. **Quantisation.** Routed and shared expert weights were converted from the healed FP8 to NVFP4, weight-only.
   The FP8 MLA projections and dense layers 0–2 were dequantised to BF16. Tensors that are BF16 upstream stay BF16.

Method write-ups in the pipeline repo: [healing](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/70-healing.md),
[quantisation](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/60-quantization.md),
[evaluation](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/97-evaluation.md),
[healing ablation](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/research/HEALING_ABLATION.md),
[v3 / HOPE](https://github.com/patrickbdevaney/glm-5.3-reap/blob/main/wiki/31-pass3-hope-method.md).

## Limitations

- **The cost of pruning is uneven across domains.** General-knowledge text is the worst-preserved domain (top-1
  agreement 0.580, against 0.92 for code and math). Expect factual-recall regressions before reasoning or coding ones,
  and consider pairing the model with retrieval.
- **No biomedical calibration data.** Do not rely on this model for biomedical work.
- **Long context is not calibrated.** Calibration used 2,048-token sequences. The only long-context evidence is the 9/9
  needle-in-a-haystack result through the CUDA server, up to 128K tokens.
- **Fails to finish on general and instruction prompts.** In paired generation, 44–72% of IFBench, MMLU-Pro and
  GPQA items hit the token cap (7–28% for the unpruned model), with repetition loops in 9–24% of them. Code is largely
  unaffected. See [Evaluation findings (2026-11)](#evaluation-findings-2026-11). Vision is unmeasured.
- **An unpruned 2-bit quant of similar size beats it** for general use; see the findings.
- **One runtime, one device.** Tested only with the custom CUDA server on Jetson AGX Thor.
- **v3 (`pass3-hope/`) is unevaluated** and carries the two defects described above.

## Changelog and corrections

- **2026-08-29**: v2 published. It first shipped with per-expert healing coefficients. An end-to-end ablation found them
  worse than the per-layer scalar (FP8 top-1 0.83693 vs 0.84249, z = 11.8), so the weights were re-uploaded the same day
  with the scalar. This NVFP4 build moved from 0.83294 to 0.83857. If you downloaded before that re-upload, download
  again.
- **2026-08-31**: `preprocessor_config.json` and `processor_config.json` added.
- **2026-10-01**: v3 added under `pass3-hope/`. The card had carried the FP8 card's title and size; the weights were
  always NVFP4.
- **2026-10-24**: the card corrected to describe only measured results. The v3 description now covers the fallback
  healing, the missing evaluation and the KDA q/k/v quantisation. Earlier text saying the MTP block was excluded was
  wrong for v2 and v3.
- **2026-11-06**: card rewritten and verified against repo files and build artifacts; a 2026-10-26 automated rewrite
  contained errors.
- **2026-11-14**: `config.json` only. Added `quantization_config.ignore` rules for the BF16 dense MLP of layers 0–2
  (`gate_proj`, `up_proj`, `down_proj`) and the MTP `eh_proj`, which were never NVFP4. Weights are unchanged and
  revision `5c5d2614` serves the same tensors as `fb45d356`. (This entry was missing from the card.)
- **2026-11-23**: added *Evaluation findings (2026-11)*, the paired generation comparison with the unpruned model, and
  the matched-bytes result. The recommendation is narrowed to code-focused use.

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
  howpublished = {\url{https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2}}
}
```
