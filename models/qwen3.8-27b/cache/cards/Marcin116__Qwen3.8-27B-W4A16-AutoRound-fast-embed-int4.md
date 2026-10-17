---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - qwen3.8
  - w4a16
  - int4
  - compressed-tensors
  - dflash2
  - rtx3090
---

# Qwen3.8-27B-W4A16-AutoRound-fast-embed-int4

A complete, single-repository serving checkpoint for Qwen3.8-27B on Ampere GPUs.
This is the assembled form of the `syvai/qwen3.8-27b-3090-fast-variant` companion
package over a W4A16 AutoRound base, so users do not need a second model download or
hardlink/overlay step.

> **Variant — INT4 token embeddings.** This is a re-quantized variant of
> [`born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast`](https://huggingface.co/born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast)
> with the token-embedding group (`model.language_model.embed_tokens`) reduced from
> **INT8 to INT4** (same group-128 symmetric pack-quantized layout). Everything else is
> byte-identical to that source. See the “Variant” note below. The `born2bewild` repo is
> the untouched original (INT8 embedding).

## Provenance

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
- Quantization: AutoRound-style symmetric INT4, group size 128, packaged as
  `compressed-tensors` (`pack-quantized`).
- Companion optimization: [`syvai/qwen3.8-27b-3090-fast-variant`](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant).
- External drafter used in the validated profile:
  [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16).

The fast variant keeps the GatedDeltaNet recurrence-control projections in BF16 and
uses the companion's optimized INT4 MTP/lm-head tensors plus token embeddings.
The vision tower remains higher precision. This repository contains all model shards,
configuration, tokenizer, MTP extra tensors, and `mtp_draft_vocab_ids.pt` required by
`from_pretrained()`; it is not an official Qwen release.

## Variant: INT4 token embeddings

This repository is derived from [`born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast`](https://huggingface.co/born2bewild/Qwen3.8-27B-W4A16-AutoRound-fast).
The **only** change is the token-embedding group, re-quantized from **INT8 to INT4**:

| Tensor (shard `model-00006-of-00007.safetensors`) | born2bewild (original) | this repo (variant) |
|---|---|---|
| `model.language_model.embed_tokens.weight_packed` | INT8 `[248320, 1280]` | INT4 `[248320, 640]` |
| `model.language_model.embed_tokens.weight_scale`  | BF16 `[248320, 40]` | BF16 `[248320, 40]` (recomputed) |
| `model.language_model.embed_tokens.weight_shape`  | I64 `[2]` | I64 `[2]` (unchanged; stores the logical shape `[248320, 5120]`) |
| `model.language_model.norm.weight` | BF16 | BF16 (byte-identical) |

- Same pack-quantized, group-128 symmetric layout (vLLM/syvai-compatible). `weight_shape` is unchanged so the loader reconstructs the full `[248320, 5120]` matrix.
- `config.json` marks the group: `group_2.targets=["re:.*embed_tokens$"]`, `weights={type:int, symmetric, group_size:128, num_bits:4}`.
- All other shards, tokenizer, MTP tensors, `mtp_draft_vocab_ids.pt`, and every other tensor are byte-identical to the `born2bewild` source. The `born2bewild` repo is untouched (it keeps the INT8 embedding).

**Size:** INT4 embedding saves ~0.635 GB on that tensor → whole repo **15.18 GB** (vs 16.47 GB for the original INT8-embedding repo). A BF16-embedding variant would instead be ~17.7 GB.

**Measured quantization error** (full `embed_tokens` dequantized vs the BF16 ground-truth weights, symmetric per-group error; cosine over the full 248320×5120 matrix):

| vs BF16 ground truth | rel. RMS error | cosine similarity |
|---|---|---|
| INT8 (original) | 0.67 % | 0.999978 |
| INT4 (this repo) | 11.77 % | 0.993120 |

The INT4 embedding's *relative* error is ~18× the INT8's, but the absolute magnitude is still small (per-token-vector RMS error ≈ 0.0009 vs a value RMS of ≈ 0.0073; vector cosine ≈ 0.993). Because embeddings are read only once per token (not per depth layer) and the rest of the network is W4A16, this is an order of magnitude below the error of the many INT4 linear layers — the practical impact on output quality is expected to be minor.

## vLLM

This checkpoint is intended for vLLM with the DFlash2 backport and the Qwen3.8
compressed-tensors support. The Club3090 DFLASH15 profile uses:

- TP=2, RTX 3090-class Ampere;
- W4A16 weights, BF16 KV, FlashAttention;
- `SPEC_N=15`, lookup-augmented DFlash2, and single-stream serving;
- `GPU_MEMORY_UTILIZATION=0.85` on WSL2/PCIe rigs.

The DFlash2 drafter is a separate checkpoint and is not embedded in this repository.

## Local evidence

On one 2x RTX 3090 WSL2 rig, the target plus DFLASH15 measured approximately 105.94
narrative and 188.06 code decode tokens/s, with a validated 244,320-token ceiling.
These are rig-specific measurements, not a universal performance claim. The exact
launcher, patch pins, benchmark logs, and caveats are maintained in the Club3090 PR.

## License and attribution

Apache-2.0. Redistribution must preserve this notice and the upstream Qwen and
AutoRound attributions. Quantization and serving artifacts are provided as-is;
please validate quality and hardware compatibility for your workload.
