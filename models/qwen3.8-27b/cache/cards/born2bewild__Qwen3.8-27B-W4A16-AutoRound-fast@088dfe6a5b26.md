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

# Qwen3.8-27B-W4A16-AutoRound-fast

A complete, single-repository serving checkpoint for Qwen3.8-27B on Ampere GPUs.
This is the assembled form of the `syvai/qwen3.8-27b-3090-fast-variant` companion
package over a W4A16 AutoRound base, so users do not need a second model download or
hardlink/overlay step.

## Provenance

- Base model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B).
- Quantization: AutoRound-style symmetric INT4, group size 128, packaged as
  `compressed-tensors` (`pack-quantized`).
- Companion optimization: [`syvai/qwen3.8-27b-3090-fast-variant`](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant).
- External drafter used in the validated profile:
  [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16).

The fast variant keeps the GatedDeltaNet recurrence-control projections in BF16 and
uses the companion's optimized INT4 MTP/lm-head tensors plus INT8 token embeddings.
The vision tower remains higher precision. This repository contains all model shards,
configuration, tokenizer, MTP extra tensors, and `mtp_draft_vocab_ids.pt` required by
`from_pretrained()`; it is not an official Qwen release.

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
