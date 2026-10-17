---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gmlx
tags:
- gguf
- speculative-decoding
- multi-token-prediction
- draft-model
- apple-silicon
---

# Qwen3.8-Flash-Next MTP draft head (GGUF)

Standalone GGUF packagings of the Qwen3.8-Flash-Next multi-token-prediction
(MTP) head, for speculative decoding with [gmlx](https://github.com/asher/gmlx)
on macOS / Apple Silicon.

Initial GGUF conversions of this model (unsloth and others) drop the
mtp.* tensors: llama.cpp does not yet implement the qwen4exp MTP head, so its
converter discards them, leaving no self-drafter. These files restore the
head. The tensors were extracted from the official bf16 safetensors
release (Qwen/Qwen3.8-Flash-Next, revision main) and re-encoded, unmodified;
either file pairs with any quantization of the base model GGUF.

## Files

| file | codecs | size | sha256 |
|---|---|---|---|
| Qwen3.8-Flash-Next-mtp-drafter-Q8_0.gguf | q8_0 throughout, norms f32 | 2.6 GiB | f2d08341...a80865a |
| Qwen3.8-Flash-Next-mtp-drafter-Q4_K_L.gguf | expert gate/up q4_k, expert down + dense q8_0, norms f32 | 1.8 GiB | ff01c084...fa266fb |

One MTP layer each (attention + 512-expert MoE + hyper-connection mixer)
plus fc/norm glue, 33 tensors. In Q4_K_L the expert down projections stay
q8_0 because their 640-wide rows do not divide the 256-wide q4_k
superblock. Full sha256 values and per-tensor shard sources are in the
.provenance.json next to each file.

## Usage

Requires gmlx. llama.cpp is not expected to load these files; the model
support they would need does not exist there as of 2026-08-26.

Place one file next to the base model GGUF and gmlx autodetects it:

    gmlx serve <base-model>.gguf --speculative

Notes:

- Place exactly one drafter file next to the base model. gmlx detects it
  by GGUF header (arch qwen4exp-mtp), not by filename; with two present
  the lexically first wins.
- Works with any base quantization (UD-Q3_K_XL, UD-Q4_K_XL, ...); the
  drafter comes from the bf16 source, not a quantized base.
- MTP acceptance on Qwen models is markedly better with thinking mode
  enabled.
- Q8_0 is the reference companion; Q4_K_L saves ~1 GiB at lower
  drafter precision.

## Attribution and license

Source weights: Qwen/Qwen3.8-Flash-Next (Qwen team, Alibaba Cloud),
Apache-2.0. These files repackage a subset of those weights and inherit
that license. Pairs with unsloth/Qwen3.8-Flash-Next-GGUF or any other
conversion. Extraction and encoding via mlx-kquant; the q4_k wire bytes
are bit-identical to the reference GGUF codec.
