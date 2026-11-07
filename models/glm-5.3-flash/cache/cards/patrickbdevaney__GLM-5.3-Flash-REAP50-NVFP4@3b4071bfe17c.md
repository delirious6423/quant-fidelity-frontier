---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
new_version: patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - reap
  - moe
  - expert-pruning
  - glm5_next
  - nvfp4
  - compressed-tensors
---

# GLM-5.3-Flash-REAP50-NVFP4 (v1, superseded; do not use)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE)
[![Superseded by](https://img.shields.io/badge/superseded_by-NVFP4--v2-lightgrey)](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)

The first published NVFP4 build of a 50% expert-pruned [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash).
It is kept only as a record. **It carries a known healing defect, and it has never been evaluated or served.**

**Use [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2)
instead.** That repo holds 98.2 GiB of weights with top-1 agreement of 0.839 against the unpruned model, and they have
been served on Jetson AGX Thor.

## The known defect

This build was quantised from the v1 FP8 master **before** that master's healing gain was corrected. Every retained
expert's output is therefore scaled by a first-moment gain (median 0.69) where the measured value is 0.9111. That makes
the MoE pathway about 0.76× too small relative to attention, the shared experts and the residual stream, in all 42 MoE
layers. The FP8 master was fixed on 2026-08-28 by rescaling its block scales. This repo was never corrected; the
corrected NVFP4 build is v2.

## Model overview

| | |
|---|---|
| Base model | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash): MIT, 321.3B parameters by safetensors count, 305.8 GiB in FP8 |
| Architecture | `Glm5NextForConditionalGeneration` (`glm5_next`). 45 decoder layers: 34 KDA linear-attention layers and 11 MLA layers with DSA sparse attention. Layers 0–2 are dense MLP, layers 3–44 are MoE. Hidden size 4096. Includes a 24-block vision tower. |
| Parameters | 161.7B after pruning. This is the element count of the v1 FP8 master, which has the same tensors; the Hub's count for this repo counts packed bytes, not weights. |
| Routed experts | 144 of 288 kept in every MoE layer. Top-8 routing and 1 shared expert, both unchanged. |
| MTP block | **Not included.** There are no layer-45 tensors, and `num_nextn_predict_layers` is 0. |
| Context length | 1,048,576 (`max_position_embeddings`, unchanged) |
| Precision | Routed and shared experts are NVFP4: weight-only, group size 16, FP8 E4M3 group scales and an F32 global scale. Everything else is BF16: KDA and MLA attention, the DSA indexer, dense layers 0–2, routers, embeddings, `lm_head`, the vision tower and the hyper-connection weights. There are no `input_scale` tensors. |
| Format | `compressed-tensors`, `nvfp4-pack-quantized` |
| Size on disk | 95.9 GiB (103,022,994,376 bytes, 62 safetensors shards) |
| Licence | MIT, inherited from the base model |

## Which version should I use

| Weights | Location | Selection | Healing | Size | Top-1 vs unpruned | Use |
|---|---|---|---|---|---|---|
| v2 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2), root | pass-2 REAP | measured per-layer scalar | 98.2 GiB | 0.8386 | **Recommended for 128 GB devices** |
| v2 FP8 | [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2), root | pass-2 REAP | measured per-layer scalar | 160.6 GiB | 0.8425 | **Recommended master, when memory allows** |
| v3 NVFP4 | [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2), `pass3-hope/` | HOPE, `protect_frac` 0.04 | first-moment fallback | 93.6 GiB | not evaluated | Experimental |
| v3 FP8 | [`GLM-5.3-Flash-REAP50-FP8-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8-v2), `pass3-hope/` | HOPE, `protect_frac` 0.04 | first-moment fallback | 160.6 GiB | not evaluated | Experimental |
| v1 FP8 | [`GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8) | pass-1 REAP | measured scalar (re-fitted 2026-08-28) | 157.0 GiB | 0.8370 | Superseded; no MTP block |
| **v1 NVFP4** | **this repo** | pass-1 REAP | superseded first-moment gain | 95.9 GiB | not evaluated | **Superseded; do not use** |

GGUF builds for llama.cpp, made from the v2 FP8 weights, are in [`GLM-5.3-Flash-REAP50-GGUF`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-GGUF);
see that card for their details.

## Quickstart

There is none. This build has not been loaded by any runtime, and it is not supported by the author's Thor CUDA server,
which supports only the v2 NVFP4 weights. The repo also has no `preprocessor_config.json` or `processor_config.json`.
Use [`GLM-5.3-Flash-REAP50-NVFP4-v2`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4-v2).

## Evaluation

**Not evaluated.** No teacher-forced or generative evaluation has been run on these NVFP4 weights. The v1 FP8 master
(0.8370) was evaluated only after its healing re-fit, so that result does not describe this build, which carries the
pre-fix gain.

## Hardware and memory

The build machine is an NVIDIA Jetson AGX Thor (128 GB unified memory, `sm_110a`). These weights have not been run on it
or on any other hardware.

## How it was made

Same selection as [`GLM-5.3-Flash-REAP50-FP8`](https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-FP8) (v1).
That is REAP ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999)) on 256 calibration samples, about 0.52M tokens,
keeping 144 of 288 experts in each of the 42 MoE decoder layers, with the MTP block left out. Routed and shared expert
weights were then converted to weight-only NVFP4 from the FP8 master, still carrying its first-moment healing gain. The
FP8 MLA projections and dense layers 0–2 were dequantised to BF16.

## Limitations

- **Healing defect** (see above): the MoE output is mis-scaled in every MoE layer.
- **Unevaluated and never run.**
- **No MTP block.**
- Superseded by v2 on every measured axis.

## Changelog and corrections

- **2026-08-28**: published. Later that day the card was flagged: this build carries the superseded healing gain, and
  the FP8 master was corrected instead.
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
  howpublished = {\url{https://huggingface.co/patrickbdevaney/GLM-5.3-Flash-REAP50-NVFP4}}
}
```
