---
license: mit
library_name: mlx
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags: [mlx, glm5_next, quantized, 4-bit, 6-bit, mixed-precision, apple-silicon]
---

# GLM-5.3-Flash-MLX-mixed-4_6bit

**Routed experts at 4-bit, everything else at 6-bit.** Named after PipeNetwork's
`mixed-4_8bit` convention: the first number is the routed experts, the second is
every other quantised layer.

An MLX quantisation of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)
that sits between the two builds people usually choose from:

| build | routed experts | attention, shared experts, embeddings, lm_head | size |
|---|---|---|---|
| uniform 4-bit | 4 | 4 | ~178 GB |
| **this build** | **4** | **6** | **179.7 GB** |
| mixed-4/8 | 4 | 8 | ~182 GB |
| uniform 6-bit | 6 | 6 | ~256 GB |

## Why this split

97% of GLM-5.3-Flash's parameters are routed experts, so expert precision
decides the **size**. But only 8 experts run per token, while attention, shared
experts, embeddings and `lm_head` run on **every** token — so their precision
decides how many bytes each decode step reads, and therefore **speed**.

On PipeNetwork's published measurements for this model, uniform 6-bit is
statistically indistinguishable from 8-bit (ΔNLL +0.0011, 95% CI [−0.0017,
+0.0038]). That makes 6-bit the natural floor for the always-on layers, and it
reads ~15% fewer bytes per token than mixed-4/8 at essentially the same size.

## Recipe

- Routed experts (`switch_mlp`): **4-bit**, group 64
- Everything else quantisable — KDA projections, MLA low-rank projections,
  absorbed `kv_b`, shared experts, dense MLPs, embeddings, `lm_head`: **6-bit**, group 64
- Lightning-indexer projections: **8-bit** (block-selection errors compound)
- Kept as stored: MoE router and correction bias, mHC arrays and KDA decay
  parameters (float32), convolutions, norms, vision tower (bfloat16)
- Dropped: the multi-token-prediction layer

Converted from the **FP8 release** (`F8_E4M3` with 128×128 block scales), one
layer at a time, so the full source is never held in memory.

## Verification

Each quantised tensor was dequantised and compared against an **independent**
dequantisation of the FP8 source: e4m3 decoded by hand in NumPy from the bit
layout, block scales read from the raw safetensors bytes. The reference shares no
code with MLX's converter, so agreement is not the converter checking itself.

Sampled layers [3, 12, 23, 34, 44] (routed experts 0, 150 and 287 of each), with the same
tensors from the mixed-4/8 build measured the same way for comparison:

| tensor group | this build | error | min corr | mixed-4/8 | error | min corr |
|---|---|---|---|---|---|---|
| routed experts | 4-bit | 9.28% | 0.9955 | 4-bit | 9.28% | 0.9956 |
| shared experts | 6-bit | 2.28% | 0.9997 | 8-bit | 0.73% | 1.0000 |
| attention projection | 6-bit | 2.38% | 0.9996 | 8-bit | 0.74% | 1.0000 |

Relative L2 error of the dequantised weight against the float reference.
The 6-bit layers carry roughly a quarter of 4-bit's error, which is what two
extra bits predict; FP8-sourced and bf16-sourced tensors land at the same error at
the same bit width, so the FP8 dequantisation adds no error of its own.

## Measured against mixed-4/8, same machine

Mac Studio M3 Ultra, 512 GB. Both builds served through `mlx_vlm.server` with a
wired memory limit, `reasoning_effort: low`, temperature 0.

**Decode**, five prompts (prose, code, Lebanese Arabic, a scheduling question,
creative), three runs each, prompt cache cold:

| prompt | mixed-4/8 | this build | |
|---|---|---|---|
| prose | 28.2 tok/s | 30.2 tok/s | 1.07× |
| code | 28.0 | 30.1 | 1.07× |
| scheduling | 28.3 | 30.1 | 1.06× |
| creative | 28.5 | 30.3 | 1.06× |

About **7% faster decode**, close to the ~9% that 15% fewer bytes per token
predicts once fixed per-token overhead is accounted for. Time to first token was
unchanged (0.23 s). Prefill was not compared head-to-head; it is compute-bound
rather than bandwidth-bound, so no gain is expected there.

**Task accuracy**, a private 64-scenario assistant suite (calendar, tasks,
messages, diarized voice notes; English, Lebanese Arabic in Arabic script, and
Latin-script Arabizi), two trials each, graded on end state:

| | mixed-4/8 | this build |
|---|---|---|
| overall | 76.6% | **76.6%** |
| Arabic subset | 83.3% | 79.6% |

Identical overall. The Arabic gap is a single scenario whose answer was correct
but phrased with a diacritic the grader did not accept. Outputs are not
token-identical to mixed-4/8 at temperature 0 — different weights choose
differently at near-ties — but the differences read as rewording, not
different content.

**Not measured:** perplexity. Weight-level and task-level results both point to
quality close to mixed-4/8, but no NLL number exists for this build yet.

## Running it

Needs a runtime with `glm5_next` support. It was built with, and verified
against, the corrected GLM-5.3-Flash MLX classes used for PipeNetwork's builds
([PipeNetwork/glm53-flash-mlx](https://github.com/PipeNetwork/glm53-flash-mlx)).

Single-stream decode on an M3 Ultra with that runtime, measured on a live
`mlx_vlm.server` at temperature 0: **30.5 tok/s eager, 33.7 tok/s with the
fused KDA decode path** from
[PipeNetwork/glm53-flash-mlx#2](https://github.com/PipeNetwork/glm53-flash-mlx/pull/2)
(`GLM_FUSED_KDA_KERNEL=1`, a Metal kernel from mlx-vlm #2105; the pure-MLX
compiled step `GLM_FUSE_DECODE=1` gives 32.1). Both are bit-identical to the
eager path. The compiled step for upstream mlx-vlm is
[Blaizzy/mlx-vlm#2255](https://github.com/Blaizzy/mlx-vlm/pull/2255).

## Credits

Base model © Z.ai, MIT licensed. Conversion recipe and runtime classes follow
PipeNetwork's GLM-5.3-Flash MLX work; their published quality table motivated
this split.
