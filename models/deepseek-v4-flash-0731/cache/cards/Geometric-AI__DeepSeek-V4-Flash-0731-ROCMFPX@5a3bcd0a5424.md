---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - gguf
  - deepseek
  - rocm
  - strix-halo
  - quantized
---

# DeepSeek-V4-Flash-0731 — ROCmFPX (uniform baseline)

A ROCmFPX GGUF quantization of **DeepSeek-V4-Flash-0731**, built to run the full target
locally on AMD Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S) in unified memory.

**This is a baseline, deliberately.** The format assignment here is uniform by tensor role
— the same public ROCmFPX recipe, applied to the new checkpoint. It exists so that our own
adaptive-format work has an honest "before" to be measured against, on the same checkpoint,
with the same harness. It is not the interesting artifact; it is the control.

## What's in it

| role | qtype | name | block | bits/weight | tensors |
|---|---|---|---|---|---|
| attention, dense | 101 | `Q4_0_ROCMFP4_FAST` | 17 B / 32 | 4.25 | 660 |
| down projections | 104 | `Q3_0_ROCMFPX` | 14 B / 32 | 3.50 | 43 |
| gate / up projections | 107 | `Q2_0_ROCMFP2` | 10 B / 32 | 2.50 | 86 |
| passthrough (norms, embeddings, router) | — | F32 / Q6_K | — | — | 539 |

1328 tensors, single file, no companion sidecar required. The qtype histogram is an exact
match to the published preview-era ROCmFPX artifact, which is what makes it a fair control:
the format is held fixed and only the checkpoint differs.

## What it is not

- **Not the Lucebox artifact.** [Lucebox's published
  ROCMFPX GGUF](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-ROCMFPX) is built from the
  *preview* checkpoint. This is an independent build from `0731`, matching their format.
- **Not adaptive.** No learned codebooks, no per-expert format selection. Those land in
  separate repos.
- **Not imatrix-calibrated.** There is no calibration input at all: the assignment is fixed
  by role in the export plan. If you are used to seeing `quantize.imatrix.*` keys in a GGUF
  of this family, their absence here is correct and deliberate — see below.

## Metadata provenance

This artifact was assembled using a metadata template taken from a published GGUF of the
same family, and it initially inherited that file's `quantize.imatrix.*` keys — which
described an imatrix calibration on someone else's build machine that played no part in
producing these weights. Those keys have been removed. What remains:

```
general.name          = DeepSeek-V4-Flash-0731-ROCMFPX
geoquant.source_model = deepseek-ai/DeepSeek-V4-Flash-0731
geoquant.format       = ROCmFPX uniform 101/104/107
geoquant.calibration  = none
```

If a GGUF's metadata claims a method that did not produce it, every downstream comparison
built on it is quietly wrong. Worth checking on any quant, not just this one.

## Speculative decode

Pair with the drafter for DSpark speculative decode:
**[DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF](https://huggingface.co/Geometric-AI/DeepSeek-V4-Flash-0731-DSpark-Drafter-GGUF)**
— extracted from this checkpoint's integrated MTP head, so it is matched to this target.

## Measurements

**Not published here yet.** Throughput, prefill, accept rate and quality for this baseline
and for the adaptive variants are being measured under one protocol on one box, and will be
reported together. A number measured here and compared against a number quoted from
somewhere else is not a comparison, so we would rather wait.

## Integrity

```
sha256  24cacd61f17bd189807d2ea51aadbde81ff15f69c4f6e358c34952bbe58dbea8
file    ds4-0731-uniform.gguf  (102,320,631,200 bytes)
```

## License

MIT, inherited from the base model.
