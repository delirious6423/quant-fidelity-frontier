---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- tenstorrent
- qwen3_5
- bfloat16
- exponent-search
---

# Qwen3.8-27B — Emax/Emax−1 values in BF16

This is an experimental derivative of **Qwen/Qwen3.8-27B**, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Selected weights were modified on
2026-09-21 using offline, weight-only exponent search for Tenstorrent BFP4_B
and BFP8_B. The files remain ordinary BF16 safetensors, approximately 55.6 GB.
This is not a compressed INT4 checkpoint or an upstream Qwen release.

## Intended loader

Prepared for `mando2222/qwen3.8-27b-dflash2-p300x2-q4kv` at revision
`9a39bc16ca0a9d560f06caa2111db6c25fa0b524`, using its existing dtypes and
4-chip tensor-parallel layout. The bundle's `q4kv` setting is about the KV
cache and is unchanged. Its DFlash2 draft checkpoint is also unchanged.

For each group of 16 outputs per input channel, evaluate the ordinary maximum
exponent and maximum exponent minus one, quantize with TT mantissa rounding and
saturation, and keep the candidate with smaller unweighted squared weight error.
Ties keep the ordinary exponent. Store the resulting representable values in
BF16. There is no calibration, inference, training, GPTQ, or activation scaling.

Modified matrices:

| Target text-model weights | Bundle format (unchanged) |
| --- | --- |
| MLP gate and up | BFP4_B |
| MLP down | BFP8_B |
| Full-attention Q, K, V, O | BFP8_B |
| GDN QKV, Z and output projections | BFP8_B |
| Language-model output head | BFP8_B |

Embeddings, norms, convolutions, small GDN A/B projections, vision parameters,
and MTP parameters are unchanged. The small GDN A/B projections are excluded
because their fused per-device widths cross 16-element exponent boundaries.
The source configuration, tokenizer, tensor names, shapes and BF16 storage
dtypes are preserved. `tt_bfp_quantization.json` records every modified tensor,
source/output hashes, group choices, reconstruction errors and validation.

## Validation and limits

Every changed group is checked for no increase in squared weight error versus
ordinary quantization. Every exported value is exactly representable in BF16
and survives ordinary max-exponent re-quantization for the audited grouping.
Every serialized tensor is hash-checked; untargeted tensors match the source.

No full-model TT inference, perplexity, task accuracy, or performance benchmark
has been run on this checkpoint. Lower weight error does not establish improved
model accuracy. BF16 is a convenient transport approximation: changing runtime
dtypes, grouping, sharding, fusions or quantization code requires revalidation.
Use a fresh TT tensor cache when switching checkpoints.

## Use with the existing bundle

On the serving host, authenticate to Hugging Face if this repository is private.
Use the same `HF_HOME` / `HF_HUB_CACHE` for all these commands. Download
`prepare_bundle.py` from this repository, then substitute this repository's ID
and exact commit below:

```bash
export CHECKPOINT_REPO='YOUR_NAMESPACE/Qwen3.8-27B-emax-minus-one-bf16'
export CHECKPOINT_REV='EXACT_40_CHARACTER_COMMIT'
tt-model pull mando2222/qwen3.8-27b-dflash2-p300x2-q4kv@9a39bc16ca0a9d560f06caa2111db6c25fa0b524
hf download "$CHECKPOINT_REPO" --revision "$CHECKPOINT_REV"
hf download incoai/Qwen3.8-27B-DFlash2 --revision dedf8df68adfb1afeaf7b7480c0a0243108177b4
python prepare_bundle.py "$CHECKPOINT_REPO" --revision "$CHECKPOINT_REV" --out qwen38-emaxm1-bundle
tt-model serve --print qwen38-emaxm1-bundle/tt_kernel_manifest.json
tt-model serve qwen38-emaxm1-bundle/tt_kernel_manifest.json
```

The helper creates a local manifest with a separate cache name and exact
snapshot selection. It preserves the image, model code, runtime precision,
KV-cache format and DFlash2 checkpoint. It does not upload or start a server.

## Reproduction and license

`export_weights.py` is the bounded-memory export driver. It uses the offline
`tt_bfp_quant` package from tt-metal branch `tcheda/offline-bfp-quantization`,
commit `75052cb05887623a82e173b2a46d3d6b4b78ab45`, with exponent candidates
`[0, -1]`. The adjacent audit manifest records the pinned source and scope.

The original Apache-2.0 `LICENSE` and original model card (`UPSTREAM_README.md`)
are retained. Copyright 2026 Alibaba Cloud is retained in the license.
The original card describes the upstream model; its published evaluation
results must not be interpreted as measurements of this modified checkpoint.
Upstream `crc32.txt` is omitted because those checksums no longer describe the
changed shards. Use `SHA256SUMS` for this export.
