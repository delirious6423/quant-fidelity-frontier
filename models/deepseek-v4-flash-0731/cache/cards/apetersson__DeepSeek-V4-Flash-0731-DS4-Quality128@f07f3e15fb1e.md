---
license: mit
library_name: gguf
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: apetersson
tags:
  - gguf
  - quantized
  - deepseek
  - deepseek-v4
  - deepseek-v4-flash
  - moe
  - mixture-of-experts
  - mxfp4
  - iq2_xxs
  - q2_k
  - ds4
  - dspark
  - apple-silicon
  - metal
---

# DeepSeek V4 Flash 0731 — DS4 Quality128

> **Clean official weights, exact MXFP4 experts, maximum resident quality.**

> **Required DS4 version:** this model is not compatible with an older DS4
> build. Its native MXFP4 tensors require
> [a recent ds4 version from the main branch](https://github.com/antirez/ds4).
> That version runs both the target model and the supplied DSpark support model.

> **Validation status:** conversion and CPU structural validation are complete.
> Metal generation, DSpark acceptance, throughput, peak-memory and actual
> one-million-token-context tests for this exact artifact are still pending.

This is a quality-first DS4 package of the official
[`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
checkpoint. It is designed to keep the target model—and optionally its real
three-stage DSpark speculative drafter—resident on a 128 GB Apple Silicon host.

This model was quantized directly from the original official checkpoint. No
behavioral weight edit was applied before quantization. All model weights were
regenerated from the official local FP8 checkpoint; no tensor values were
copied from another GGUF or quantized model.

## Artifacts

| File | Purpose | Bytes | GiB | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `DeepSeek-V4-Flash-0731-DS4-Quality128.gguf` | Authoritative 43-layer target model | `102,826,238,912` | `95.7644` | `efcbf786154c2aec61511785d04a4c8d98ee42500b60ff8a810717fc0a69d1d3` |
| `DeepSeek-V4-Flash-0731-DS4-Quality128-DSpark-support.gguf` | Three-stage speculative drafter; not standalone | `7,297,737,120` | `6.7965` | `393b807aa0a409f1c006f0e700bbb91a80d3de28daa56c222edf944250075c66` |
| **Combined GGUFs** | — | **`110,123,976,032`** | **`102.5609`** | — |

`BUILD_MANIFEST.json`, `PROVENANCE.md` and `SHA256SUMS` provide the full
machine-readable build record and integrity inventory.

## Quantization profile

### Main target model

| Tensor class | Quantization/storage | Rationale |
| --- | --- | --- |
| Routed gate/up/down on layers 10, 14, 30, 34, 37, 38, 39, 40, 41 and 42 | Exact native `MXFP4` | Preserves source packed I8 codes and F8_E8M0 scales byte-for-byte on sensitivity-selected layers. |
| Routed gate/up on the other 33 MoE layers | `IQ2_XXS`, importance-matrix calibrated | Applies the most aggressive compression to the largest tensor bank. |
| Routed down on the other 33 MoE layers | `Q2_K`, importance-matrix calibrated | More conservative two-bit storage on the projection that writes expert output to the residual stream. |
| Attention projections | `Q8_0` | Protects a dense path used for every token. |
| Shared experts | `Q8_0` | Protects the expert path active for every token. |
| Vocabulary/output head | `Q8_0` | Protects final-logit fidelity. |
| Indexer `attn_q_b` tensors on layers 2, 4, …, 42 | forced `F16` | Preserves a small, sensitive compressed-attention component. |
| Remaining control, normalization, routing and auxiliary tensors | template-declared `F16`/`F32`/`I32` | Avoids forcing small or numerically sensitive tensors into the low-bit expert rules. |

Observed main-GGUF inventory from strict DS4 inspection:

| GGUF type | Tensors |
| --- | ---: |
| `F32` | 492 |
| `F16` | 359 |
| `I32` | 3 |
| `Q8_0` | 345 |
| `IQ2_XXS` | 66 |
| `Q2_K` | 33 |
| `MXFP4` | 30 |
| **Total** | **1,328** |

The target GGUF is version 3, describes approximately 284.33 billion logical
parameters and retains the checkpoint's declared 1,048,576-token training
context. That declaration is not evidence that a one-million-token inference
run fits or remains robust on a particular machine.

### DSpark support model

This is the actual 0731 three-stage DSpark module (`mtp.0`–`mtp.2`), not the
legacy single-stage MTP attachment. The target model remains authoritative and
verifies speculative proposals.

| Parameter | Value |
| --- | --- |
| Stages | 3 |
| Proposal block size | 5 |
| Target layers | 40, 41, 42 |
| Markov rank | 256 |
| Noise token ID | 128799 |
| Routed gate/up | `IQ2_XXS` (6 tensors) |
| Routed down | exact native `MXFP4` (3 tensors) |
| Dense projections | `Q8_0` (31 tensors) |
| Control/auxiliary tensors | 7 `F16` + 34 `F32` tensors |
| Total | 81 tensors |

The support GGUF describes approximately 19.85 billion logical parameters and
must be loaded alongside its matching target GGUF.

## Importance calibration

The routed-expert calibration source is the 0731-native matrix from
[`ox-ox/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/ox-ox/DeepSeek-V4-Flash-0731-GGUF):

- Repository revision: `6d58a3a36030c3ccb969bb5759fc6ae08cd299f8`
- File: `imatrix/DeepSeek-V4-Flash-0731-chat-v2-routed-moe-ds4-1p5m.dat`
- Size: `450,892,654` bytes
- SHA-256: `6fce7674df701de544e5d3351aab04e67602eddeafeb48cf70e77ebe47239eb4`
- Coverage: exactly 129 routed target tensors (43 layers × gate/up/down), with exact name and vector-dimension validation

The build ran strict imatrix validation. Preserved native MXFP4 tensors do not
consume imatrix data because they are copied from the official FP8 checkpoint's
packed source representation rather than requantized.

The published matrix has no native DSpark entries. The support-model importer
therefore made deterministic target-layer proxy aliases:

- `mtp.0` ← target layer 40
- `mtp.1` ← target layer 41
- `mtp.2` ← target layer 42

The extended matrix contains 138 entries and has SHA-256
`689b446ed2e2657ebcb69a6516781ee0444e07641aeb6125233fb4e6fe7cbce3`.
This is an explicitly recorded proxy, not a fresh activation capture from the
DSpark drafter.

## Weight provenance and metadata template

The sole source of weight values was the local copy of
`deepseek-ai/DeepSeek-V4-Flash-0731` at official revision
`9e165c30e2704aec5d9d593cce3eebd58bbef1cb`. All 48 FP8 weight shards—
`166,886,535,336` bytes in total—were fully SHA-256 verified and guarded
against mutation throughout conversion.

The exact-0731 GGUF from
[`antirez/deepseek-v4-gguf`](https://huggingface.co/antirez/deepseek-v4-gguf)
was used only as a bounded metadata, tokenizer, tensor-order and shape template:

- Repository revision: `1cd7b564460821938add0475a60b942c409295e0`
- Template LFS SHA-256: `ca22ae2f838e14077c22bc1c1417b71b45b5e5a3687bd96c2ac6e17fdb6261c0`
- Template Xet object: `7da16e1025c1b856490c29c341f4e467d15cb195389c70383dade5e6108799ac`
- Verified 64 MiB header SHA-256: `f0e1d5e8f3b008402aa6eb32cada3873dd926c8bf5e7d00d7788eec65f09dd6d`
- Template weight values used: **no**

## Runtime requirement

Use [a recent ds4 version from the main branch](https://github.com/antirez/ds4).
This is required because the main GGUF contains 30 native MXFP4 routed-expert
tensors. An older runtime without the MXFP4 loader and Metal kernels now in
`main` is not compatible, even if it can parse the GGUF header.

Install or update a `main`-branch checkout:

```bash
git clone https://github.com/antirez/ds4.git
cd ds4
git switch main
git pull --ff-only
make
```

Build note: the [`d516d4e` quantizer PR](https://github.com/apetersson/ds4-omlx/tree/d516d4eeb82c454aeb2831af1b1961801d6b571b),
tracked as [`antirez/ds4#642`](https://github.com/antirez/ds4/issues/642),
was needed to create the preserved-MXFP4 DSpark GGUF. Users do **not** need that
PR to run either supplied file.

Do not substitute `antirez/ds4` `main`, an older DS4 binary or a generic GGUF
runtime. Container parsing alone does not demonstrate correct native MXFP4 or
mixed DSpark execution.

## Running with DS4

Target-only Metal inference:

```bash
./ds4 --metal \
  -m DeepSeek-V4-Flash-0731-DS4-Quality128.gguf
```

Greedy DSpark inference:

```bash
./ds4 --metal \
  -m DeepSeek-V4-Flash-0731-DS4-Quality128.gguf \
  --mtp DeepSeek-V4-Flash-0731-DS4-Quality128-DSpark-support.gguf \
  --dspark --temp 0
```

DSpark is opt-in, accelerates generation rather than prefill, and can be neutral
or slower when proposal acceptance is low. Sampled decoding does not use DSpark
proposals in the pinned runtime.

Structural inspection:

```bash
./ds4 --cpu --inspect \
  -m DeepSeek-V4-Flash-0731-DS4-Quality128.gguf \
  --mtp DeepSeek-V4-Flash-0731-DS4-Quality128-DSpark-support.gguf \
  --dspark-strict
```

## Memory planning on a 128 GB Mac

The target machine has 128 GiB (`137,438,953,472` bytes) of unified physical
memory, but the more relevant ceiling for a large Metal allocation is its
approximately `121.60 GiB` `recommendedMaxWorkingSetSize`. The remaining
roughly `6.40 GiB` is not extra model capacity; macOS, applications, drivers and
untracked transient allocations still need memory. A configuration at or below
128 GiB can therefore be unsafe when it is above—or too close to—the Metal
working-set recommendation.

### Fixed weight cost

| Loaded weights | GiB | Share of 128 GiB | Share of 121.60 GiB Metal recommendation |
| --- | ---: | ---: | ---: |
| Target model only | `95.7644` | `74.8%` | `78.8%` |
| Target + DSpark support | `102.5609` | `80.1%` | `84.3%` |

These are GGUF payload sizes, before KV cache, indexed-attention scratch,
prefill workspace, verifier state and other runtime allocations. DS4's planner
uses an approximately `97.63 GiB` resident span for the main model after its
mapping/alignment accounting, rather than treating the main file size as the
entire live allocation.

### Context-length scaling at prefill chunk 1,024

`--ctx` is the total prompt-plus-completion capacity. `--prefill-chunk` is the
maximum prompt microbatch DS4 processes at once; it is the relevant “batch
size” for this single-session memory calculation. A larger chunk can improve
prefill throughput, but indexed-attention scratch grows with both context length
and chunk size.

The following are conservative planning estimates. “Margin” is remaining space
under the `121.60 GiB` Metal recommendation, not free system RAM.

| Context | DSpark off: total | Off: margin | DSpark on: total | On: margin |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | `97.77 GiB` | `23.83 GiB` | `104.57 GiB` | `17.03 GiB` |
| 32,768 | `98.05 GiB` | `23.55 GiB` | `104.85 GiB` | `16.75 GiB` |
| 131,072 | `98.99 GiB` | `22.61 GiB` | `105.79 GiB` | `15.81 GiB` |
| 262,144 | `100.24 GiB` | `21.36 GiB` | `107.04 GiB` | `14.56 GiB` |
| 524,288 | `102.75 GiB` | `18.85 GiB` | `109.55 GiB` | `12.05 GiB` |
| 1,048,576 | `107.77 GiB` | `13.83 GiB` | `114.56 GiB` | `7.04 GiB` |

At ordinary 4K–128K contexts, the model weights dominate and chunk 1,024 leaves
substantial planned margin. At 512K and especially 1M, the compressed KV and
context-by-chunk attention workspace become material. DSpark adds approximately
`6.80 GiB` at every context length because its support model remains resident
during prefill even though speculative decoding only accelerates generation.

### Prefill batch-size effect at one-million-token context

This table holds `--ctx 1048576` constant and varies `--prefill-chunk`. Totals
are the conservative envelope of DS4's resident-span/context estimator and the
completed release planning model. Percentages use all 128 GiB of physical RAM;
the Metal margin remains the safer operational measure.

| Prefill chunk | DSpark off total | Off: 128 GiB used | Off: Metal margin | DSpark on total | On: 128 GiB used | On: Metal margin |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | `106.20 GiB` | `83.0%` | `15.40 GiB` | `113.00 GiB` | `88.3%` | `8.60 GiB` |
| 512 | `106.72 GiB` | `83.4%` | `14.88 GiB` | `113.52 GiB` | `88.7%` | `8.08 GiB` |
| 1,024 | `107.77 GiB` | `84.2%` | `13.83 GiB` | `114.56 GiB` | `89.5%` | `7.04 GiB` |
| 2,048 | `110.30 GiB` | `86.2%` | `11.30 GiB` | `117.09 GiB` | `91.5%` | `4.51 GiB` |
| 4,096 | `116.44 GiB` | `91.0%` | `5.16 GiB` | `123.24 GiB` | `96.3%` | **`−1.64 GiB`** |

The 4,096/DSpark combination exceeds Metal's recommendation despite fitting
numerically inside 128 GiB and should not be treated as resident-safe. The
2,048/DSpark combination is also tight: its `4.51 GiB` planned Metal margin can
be consumed by omitted driver and verifier peaks. Chunk 1,024 is the sensible
first DSpark experiment at 1M; target-only chunk 2,048 is the more conservative
maximum-quality starting point. Chunks 256–512 provide more margin at the cost
of more prefill iterations and likely lower prompt-processing throughput.

Target-only 1M starting point:

```bash
./ds4 --metal \
  -m DeepSeek-V4-Flash-0731-DS4-Quality128.gguf \
  --ctx 1048576 --prefill-chunk 2048
```

DSpark 1M starting point, only after unloading other large applications and
measuring the target-only peak:

```bash
./ds4 --metal \
  -m DeepSeek-V4-Flash-0731-DS4-Quality128.gguf \
  --ctx 1048576 --prefill-chunk 1024 \
  --mtp DeepSeek-V4-Flash-0731-DS4-Quality128-DSpark-support.gguf \
  --dspark --temp 0
```

### Multiple sessions and server batching

The tables describe one resident session. They do not mean that `N` concurrent
requests cost exactly `N` times the displayed total: weights and some graph
workspace are shared, while each resident session needs its own KV/context
buffers and live request state. DS4 reports the aggregate minimum context-buffer
request at server startup. For multi-user serving, start with one session,
measure the process and system peak, then raise concurrency one session at a
time; do not infer a safe concurrency count from the GGUF sizes alone.

All figures above are planning estimates, not measurements of this exact
artifact. Metal load, peak-memory capture and an actual 1,048,576-token run
remain required. Keep several GiB of additional operational margin, watch
memory pressure rather than only Activity Monitor's process RSS, and expect
other active models or large applications to invalidate the table.

## Reproducibility

- Build tooling revision: `e24746463a0a3e79036dd9c0472deac6ce704f08`
- Build driver SHA-256: `cb1569b909de010b4aaa2a1b1b7ed8e277bcdf41e32585f1651005b41e81567f`
- Build-time DS4/quantizer revision: `d516d4eeb82c454aeb2831af1b1961801d6b571b`
- oMLX revision: `76352ed2363e42b2146243463875756e683a640d`
- Profile SHA-256: `bc7f6470a0cb6208addf2ac52806d976108517119b411023da3fd91b492b218c`
- Build run ID: `20260801-161944-54912`

The tooling worktree was intentionally dirty and is cryptographically described
in `BUILD_MANIFEST.json`, alongside the complete profile, source-shard hashes,
calibration/template identities and exact converter commands.

## Validation status

Completed on 2026-08-01:

- focused build-driver and finalization test suites: 40/40 passed;
- every one of the 48 official source weight shards fully SHA-256 verified;
- strict routed-imatrix name and vector-dimension coverage passed;
- main GGUF: exact size, 1,328 tensors, exact name set and exact type histogram;
- DSpark GGUF: exact size, 81 tensors, three stages and target layers 40/41/42;
- strict DSpark binding: 81 tensors, 0 missing, 0 invalid and 0 metadata errors;
- `ds4 --cpu --inspect --dspark-strict` passed;
- byte-for-byte regeneration passed for all 30 main and all three DSpark native
  MXFP4 tensors; and
- the original build payloads passed their recorded SHA-256 checks before
  atomic publication; this README was added afterward and independently added
  to `SHA256SUMS`.

Still required before making runtime, quality or performance claims:

- successful Metal load and deterministic generation with DSpark disabled;
- successful Metal generation with DSpark enabled and lossless target agreement;
- proposal acceptance rate and accepted tokens per target step;
- prompt-processing and generation tokens/second;
- measured peak unified memory and practical context limits on the target host;
- an actual 1,048,576-token context run; and
- capability/perplexity comparisons against the official FP8 source.

## Limitations and responsible use

Ultra-low-bit expert quantization can reduce reasoning, factuality, style
fidelity and long-context robustness even when dense paths and selected experts
are protected. Native MXFP4 Metal kernels and the mixed DSpark path are newer
than DS4's mature Q4_K path. This package does not guarantee correctness,
neutrality, safety, regulatory compliance or a particular response style.
Evaluate it for the intended workload and apply appropriate access controls.

## License and attribution

The upstream repository and weights are MIT licensed. This quantized derivative
retains that license. Credit DeepSeek-AI for the original model, ox-ox for the
0731 routed-expert importance matrix, antirez for DS4 and the exact-0731
metadata recipe, and apetersson for the DS4 fork, conversion profile and release
tooling.
