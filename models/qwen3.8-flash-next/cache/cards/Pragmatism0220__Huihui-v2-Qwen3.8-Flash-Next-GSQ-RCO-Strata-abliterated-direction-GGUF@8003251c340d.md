---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
license: apache-2.0
tags:
  - gguf
  - gsq
  - rco
  - quantization
  - mixed-precision
  - ist-daslab
  - moe
  - multimodal
  - vision
---

# Huihui-v2-Qwen3.8-Flash-Next-GSQ-RCO-Strata-abliterated-direction-GGUF

A single-file GGUF **control vector** containing the **refusal direction** that
[huihui-ai](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF)'s
*abliterated* releases of [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
remove from the model weights. Instead of committing to baked-in abliterated weights,
you can load this direction at runtime — on any GGUF quant of the **base** model —
and dial the strength yourself (including off).

| | |
|---|---|
| File | `Huihui-Qwen3.8-Flash-Next-refusal-direction-r-v2.gguf` |
| Size | 483,520 B |
| sha256 | `ac870b06535f809b0e546436f629751e8d425977a3a186c18c601b38651a7fc0` |
| Format | GGUF v3 · `general.architecture = controlvector` |
| Tensors | 47 × `direction.<l>` · F32 · `[2560]` · byte-identical copies · unit norm |
| Metadata | `controlvector.model_hint = qwen4exp` · `controlvector.layer_count = 47` |

## What this is

huihui-ai publishes abliterated variants of Qwen3.8-Flash-Next in two strengths:

- **v1** — `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated` (bf16 safetensors + GGUF)
- **v2** — `huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF` (UD-quantized GGUF only)

This repository extracts the direction those releases operate on, as a standalone
control vector:

- the direction is extracted at full precision from the **bf16 v1 checkpoint**
  (v2 ships only in quantized form, so bf16 is the highest-precision source);
- it is then validated, in situ, against the **published v2 GGUF tensors**
  (see the precision report below) — v2 uses the *same* direction at ≈ 1.78× v1's strength.

## How it works

1. **Background.** Refusal behavior in LLMs is largely mediated by a single direction
   in the residual stream (Arditi et al., 2024). "Abliterating" a model means removing
   that direction from its weights.

2. **What the abliterated releases actually change.** Diffing huihui's releases against
   `Qwen/Qwen3.8-Flash-Next` shows that exactly **96 "writer" matrices** are edited
   (all other tensors — including the MTP head — are byte-identical to the base):

   | role | tensors | shape |
   |---|---|---|
   | full-attention `o_proj` | 12 | 2560 × 6144 |
   | linear-attention (GDN) `out_proj` | 36 | 2560 × 6144 |
   | shared-expert `down_proj` | 48 | 2560 × 640 |

   These are the matrices that write into the residual stream.

3. **The edits are rank-one and share one direction.** For each edited tensor,
   ΔW = W<sub>abliterated</sub> − W<sub>base</sub> concentrates **99.83 %** (median) of its
   energy in the top singular component, i.e. ΔW ≈ −s·r·vᵀ. The leading vector **r**
   (2560-dim, output/residual-stream space) is the *same* across all 96 writers — that
   vector is the refusal direction, and it is what this file contains (unit-normalized).

4. **Strengths, as measured in the published weights.**
   With s defined as the removal factor of the r-component in each writer
   (s = 1 → component fully removed; s > 1 → component inverted):

   | release | s (96 writers, mean ± std) | r-component after edit |
   |---|---|---|
   | v1 | 0.99969 ± 0.00022 | +0.0003 (≈ 0, removed) |
   | v2 | 1.77989 ± 0.00023 | −0.78 (inverted) |

   v2 therefore applies the **same direction at 1.780× v1's strength**, flipping the
   component to −0.78 of its original value.

## Usage

Two supported engines, two different conventions — make sure you use the right
one for each.

### Strata (easiest)

[Strata](https://github.com/Niko1221/Strata) ships an engine based on the
[`Cudila/llama.cpp` `cvec-hooks`](https://github.com/Cudila/llama.cpp/tree/cvec-hooks)
fork, which applies control vectors in **projection mode** by default for this:
`h -= scale·(h·v)v` per layer. In that convention a **positive** scale is the
fraction of the direction to remove, so `1.0` = full removal:

```bash
./setup.sh --setup --model IQ3_S \
  --experimental-speed-projection /path/to/Huihui-Qwen3.8-Flash-Next-refusal-direction-r-v2.gguf
```

Setup writes `--control-vector-scaled <file>:1.0 --control-vector-layer-range 4 44
--cvec-mode project --cvec-dir per-layer` into the engine config (`--cvec-mode`
and `--cvec-dir` exist only in that fork, not in stock llama.cpp). Note: Strata's
own built-in "speed projection" vector is a *different* per-layer direction; this
file replaces it.

### llama.cpp

Stock llama.cpp applies control vectors **additively** (`h += scale·direction`),
and — as of master `620d222` and release `v0.6.0` — **silently ignores them for
the `qwen4exp` architecture**, because `src/models/qwen4exp.cpp` never calls
`build_cvec()` (verified; every other architecture does). Apply the two-line
patch from this repo first:

```bash
cd llama.cpp
git apply llama.cpp-qwen4exp-controlvector.patch
cmake -B build && cmake --build build -j --target llama-server
```

Then (note the **negative** scale — additive convention):

```bash
llama-server -m <base Qwen3.8-Flash-Next GGUF> \
  --control-vector-scaled /path/to/Huihui-Qwen3.8-Flash-Next-refusal-direction-r-v2.gguf:-1.0
```

Practical notes:

- **Apply it to the base model** (e.g. from `unsloth/Qwen3.8-Flash-Next-GGUF`),
  not to an already-abliterated release — otherwise the effects stack.
- **Sign.** Additive mode (stock): the vector points *along* the refusal
  direction, so a **negative** scale suppresses it. Projection mode
  (cvec-hooks fork / Strata): **positive** scale = fraction removed. If refusal
  gets *stronger*, you are in the other convention — flip the sign.
- **Scale.** Additive: `-1.0` ≈ v1-like removal, `-1.78` ≈ v2-like inversion
  (measured weight-space factors; runtime scales don't map 1:1, tune on your
  own prompts). Projection: `1.0` = remove the component entirely.
- `--control-vector-layer-range START END` (inclusive) is optional; the file
  covers layers 1..47 and all 47 per-layer vectors are byte-identical, so
  restricting the range only weakens the effect. Omitting the flags (or
  scale 0) leaves the model untouched.
- Very old builds used `--control-vector-file <fname>`.

#### Known issue

Control vectors are **silently ignored** for the `qwen4exp` architecture in
stock llama.cpp (`src/models/qwen4exp.cpp` never calls `build_cvec()`; verified
on master `620d222` and release `v0.6.0`). The `Cudila/llama.cpp` `cvec-hooks`
branch (used by Strata) already contains the fix at `qwen4exp.cpp:505`; the
patch in this repo backports it — the main-graph line is byte-identical to the
fork's, and the optional MTP line mirrors upstream `deepseek4.cpp:1478` (it is
currently a harmless no-op in stock, since no direction tensor exists for the
MTP layer index).

## Precision report

All numbers are weight-space measurements computed directly from the published tensors.

**Extraction quality** (96 writer tensors, bf16 v1 vs bf16 base):

| metric | value |
|---|---|
| rank-one energy of ΔW (median / min) | 99.833 % / 99.787 % |
| relative edit ‖ΔW‖ / ‖W‖ (median / max) | 3.86 % / 4.86 % |

**Cross-check vs an independent extraction** (third-party control-vector GGUF of the
same direction, sha256 `59f9ae8b4372107a…`):

| metric | value |
|---|---|
| cos(this, reference) | 0.99999998 |
| angle | 0.011° |

**In-situ validation vs the published v2** (UD-Q4_K_XL GGUF, all 96 writers):

| metric | value |
|---|---|
| global direction cos(this, v2-measured) | 0.9999967 |
| measurement noise floor (same pipeline on v1, whose answer is known) | 0.9999919 |
| per-writer direction cos (min / mean) | 0.99993 / 0.99997 |
| v2 strength s₂ (96 writers) | 1.77989 ± 0.00023 |

The deviation between this file's direction and what v2 actually encodes (3.3 × 10⁻⁶)
is ≈ 2.5× **below** the measurement pipeline's own noise floor (8.1 × 10⁻⁶): within
measurement resolution, v2 uses the direction in this file, at 1.78× v1's strength.

**Scope of these claims.** The numbers above characterize *weights*. No behavioral
benchmarks (refusal rates, capability evals) are included in this repository.

## Provenance & credits

- Base model: [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- GSQ RCO GGUF: [`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
- Extraction source (bf16): [`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated)
- Validation target (v2): [`huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF`](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-v2-GGUF)
- Pristine GGUF baseline: [`unsloth/Qwen3.8-Flash-Next-GGUF`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)
- Method background: Arditi et al., *Refusal in Language Models Is Mediated by a Single
  Direction*, NeurIPS 2024.

## License

This file is a direction vector derived from the releases above. Please comply with
the upstream licenses (Qwen base model, huihui-ai releases) when redistributing or
using it, and pick this repository's license accordingly.
