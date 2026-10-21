---
license: other
license_name: deepseek
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
- gguf
- rocmfp
- rocmfpx
- strix-halo
- gfx1151
- amd
- deepseek-v4
- deepseek-v4-0731
- moe
- imatrix
- abliterated
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 — Strix Halo ROCmFPx GGUF

> [!WARNING]
> This is a custom GGUF for **[Ember](https://github.com/otheru-ai/ember)**, an
> ROCmFPx-aware DeepSeek-V4 runtime for AMD Strix Halo (`gfx1151`). It uses
> custom tensor types that mainline `llama.cpp`, LM Studio, and Ollama do not
> implement. Loading it with an incompatible runtime may fail or produce invalid
> output.

An abliterated, importance-matrix-calibrated quantization of the official
`deepseek-ai/DeepSeek-V4-Flash-0731` release, plus its matching DSpark drafter.

> [!IMPORTANT]
> This is the text-only model. Ember's default download is now the vision-capable
> [DeepSeek-V4-Flash-Vision-Exp](https://huggingface.co/otheru/DeepSeek-V4-Flash-Vision-Strix-Halo-GGUF),
> which also serves text and ships an image-calibrated importance matrix.
>
> This artifact is not deprecated and has not been deleted. Set
> `EMBER_MODEL_REPO` and `EMBER_MODEL_REVISION` to keep serving it. Its DSpark
> drafter is the one the vision release ships with, because it measured better
> than the vision checkpoint's own.
>
> The two models' text quality has not been compared, so neither supersedes the
> other on evidence.

## Ember performance benchmark

Measured 2026-08-22 on one AMD Ryzen AI Max+ 395 with Radeon 8060S (`gfx1151`)
and 128 GB unified memory, against **the artifacts published here**
(target SHA-256 `a936e0a5…`, drafter `1a01c80e…`, both re-verified against the
local copies before the run). Runtime was [Ember](https://github.com/otheru-ai/ember)
release `2026.8.22` in `ember-rocm:7.14`, with speculative decoding enabled and
per-token instrumentation disabled.

> [!IMPORTANT]
> These measure the production serving configuration: sparse prefill plus DSpark
> speculative decode, one request at a time. They are not exact-prefill reference
> numbers and they do not measure quality against BF16.

![Ember sparse-prefill throughput by context length](./benchmarks/ember-2026-08-22/prefill-throughput.svg)

| Evaluated prompt tokens | Samples | Median prefill tok/s | Observed range | Median wall time |
|---:|---:|---:|---:|---:|
| 154 | 3 | 209.6 | 206.4–210.9 | 0.736 s |
| 538 | 3 | 319.2 | 313.2–320.5 | 1.686 s |
| 2,074 | 3 | **386.8** | 365.4–391.1 | 5.364 s |
| 8,218 | 3 | 337.2 | 331.6–337.9 | 24.370 s |
| 16,410 | 2 | 310.7 | 310.0–311.4 | 52.812 s |
| 32,794 | 2 | 282.3 | 282.0–282.6 | 116.165 s |

Every scaling prompt reported zero restored prefix tokens, so each pass evaluated
the full prompt. First runs at each shape are included in the medians.

![Ember DSpark decode throughput](./benchmarks/ember-2026-08-22/decode-throughput.svg)

Three greedy 256-token samples produced **34.16 tok/s median decode**
(33.67–34.16) at **0.989 median DSpark acceptance** — one of the three samples
reached 1.000 — with speculation active on 3/3.

This release carries two gfx1151 kernel changes: a branch-free UE4M3 scale
decode and per-translation-unit selection of WGP vs CU mode. Against the
previous published figure of 33.60 tok/s (2026-08-10, commit `8872221`,
same two model files), that is **+1.7% median decode**.

> [!NOTE]
> Acceptance is workload-dependent. A greedy 256-token continuation is highly
> predictable and accepts nearly everything; separate mixed free-form sampling at
> temperature 0.6 measured ~0.52 mean acceptance on the same pair. Both are real
> and describe different workloads. The kernel changes in this release are
> numerically inert: the differential validator reports the autoregressive,
> DSpark, disk and resident-batch paths all token-exact.

The [full benchmark bundle](./benchmarks/ember-2026-08-22/README.md) contains the
harness, raw JSONL, summary and environment JSON, both SVG sources, and the
exact model and runtime identifiers. The
[2026-08-10 bundle](./benchmarks/ember-2026-08-10/README.md) is retained for
comparison.

### Quantization quality — not yet characterized

No perplexity, no KL-divergence against BF16, no capability benchmark has been
run on this artifact. The numbers above are **throughput only**. The abliteration
has been validated behaviourally (see below), and the tensor layout has been
validated structurally, but the quality cost of the 2.58 BPW quantization itself
is currently unmeasured. Treat it as unknown rather than as acceptable.

## Artifacts

Verify any local copy against the SHA-256 values below.

| file | bytes | GiB | SHA-256 |
| --- | ---: | ---: | --- |
| `DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf` | `91,547,243,200` | `85.2600` | `a936e0a514385c8ae964c0f42263a4314a34fbc6efea9d9aced5320f320a3d54` |
| `DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf` | `10,897,111,840` | `10.1487` | `1a01c80eceae302bcc1d70836759ee97974d7983c5084ef43f6ef772a8970ae6` |

Target + matching drafter is `102,444,355,040` bytes (`95.41 GiB`) resident.

## The abliteration band

The edit is restricted to the band where the refusal direction actually
separates — blocks 10 through 42, plus the 3 DSpark stages, leaving the first
ten blocks untouched.

| | |
| --- | --- |
| edited tensors | **36** (blocks 10–42 + 3 MTP) |
| strength (lambda) | **3.5** |
| SRA rank | **0** (none) |
| row-norm preserve | **yes** |
| direction | **rank-1, published, SHA-256 pinned** |

Blocks 0–9 are left untouched. Early layers carry general features and show the
weakest harmful/harmless separation, so projecting there costs capability
without adding bypass. Abliterating all 46 attention output-B tensors instead of
36 is known to damage agentic coherence: public tuning work on this model family
found the all-layers variant reaches full refusal bypass but degrades
instruction following, with the characteristic symptom being the model emitting
its skill catalog in response to a bare greeting.

The refusal direction is the published rank-1 vector, SHA-256
`6e4d8a8f3aa9e21795faab2c5b14d29b019acdf2ddbfbd8238430458a5837fe0`
(MIT, credit drowzeys/keys), rather than a locally extracted per-layer set.

Per-tensor relative-Frobenius edit magnitude: min `0.0494`, median `0.0566`,
max `0.0668`.

## Verification of the abliteration

Each block's `attn_output_b` was probed directly in the shipped weights by
measuring the refusal-direction component against an isotropic baseline
(`||r^T W|| / (||W||_F / sqrt(n_out))`). An untouched tensor sits near 1.0.

| artifact | blocks 0–9 | blocks 10–42 | verdict |
| --- | ---: | ---: | --- |
| non-abliterated reference | 1.01 | 1.05 | clean |
| **this target** | **0.93–1.17** | **mean 2.163** | band edit confirmed |
| **this drafter** | — | 2.501 / 2.541 / 2.559 | all 3 stages edited |

## Behavioural validation

136 cases, all recorded, greedy (temperature 0, top_p 1, seed 42), reasoning
pass disabled:

| suite | n | result |
| --- | ---: | --- |
| refusal bypass (held-out harmful) | 96 | **0 refusals (0.0%)** |
| over-refusal (XSTest-style safe-but-alarming) | 32 | 0 strict, 1 soft (3.1%) |
| coherence / agentic | 8 | **7/8** |

The bypass set is disjoint from the prompts used to derive the refusal
direction. The greeting-versus-skill-catalog case — the symptom that motivated
this build — **passes**. The single coherence failure is a verbosity bound: the
model answered correctly but wrapped the answer in a code fence, exceeding a
word cap.

This is a self-consistent harness, comparable across builds from this project
only. It is not a sealed judge suite and it does not measure capability.

## Structural validation

The target was checked against a per-tensor type contract extracted from a GGUF
known to run on the target runtime: architecture, tensor count, block range,
pinned metadata, and the type of every tensor class.

```text
1,328 tensors, 41 tensor classes, all types match
```

Type histogram — target: `Q4_0_ROCMFP4_FAST` 574, `F32` 535, `Q2_0_ROCMFPX` 129,
`Q4_0_ROCMFP4` 43, `Q8_0` 43, `I32` 3, `Q6_K` 1.
Drafter (82 tensors): `F32` 46, `Q8_0` 25, `Q4_0_ROCMFP4_FAST` 9, `F16` 2.

## The DSpark drafter

The drafter is a **separate GGUF**, not part of the 43-block target: architecture
`deepseek4-dflash-draft`, 3 blocks, each a full 256-expert MoE block. It is
attached at runtime and the two files are never merged.

```text
DFLASH_DS4_SPEC=1
DFLASH_DS4_DRAFT=/path/to/...-Abliterated-DSpark-draft-4.25bpw.gguf
DFLASH_DS4_SPEC_Q=4
DFLASH_DS4_FUSED_VERIFY=1
```

> [!NOTE]
> **Pairing the abliterated drafter is optional.** It was measured against the
> stock drafter on the same target, 16 prompts (8 benign / 8 refusal-triggering),
> greedy:
>
> | drafter | spec runs | mean acceptance | benign | harmful |
> | --- | ---: | ---: | ---: | ---: |
> | abliterated (matched) | 16/16 | 0.520 | 0.605 | 0.435 |
> | stock (mismatched) | 16/16 | 0.508 | 0.578 | 0.438 |
>
> The two are indistinguishable, including on refusal-triggering prompts where a
> mismatch was predicted to hurt most. **No throughput benefit is claimed for the
> matched drafter.** It is published because a drafter that models the same
> distribution as its target is the defensible default, not because it measured
> faster. A stock DSpark drafter works.
>
> Speculative decoding is disabled above `DFLASH_DS4_SPEC_MAX_CTX`.

Note also that the drafter is *not* interchangeable with a `ds4 --mtp` DSpark
support GGUF: that artifact keeps `mtp.*` tensor names, while this runtime
resolves only `blk.*`, `dflash.*` and `output_*`. They share no tensor names.

## Quantization recipe

The base preset is `Q4_0_ROCMFP4_STRIX_LEAN`, adapted for DeepSeek-V4's fused
MLA tensor names:

- all 129 routed-expert gate/up/down tensors: affine `Q2_0_ROCMFP2`
  (custom type 107, 2.5 BPW)
- all 43 fused `attn_kv` tensors: dual-scale `Q4_0_ROCMFP4`
  (custom type 100)
- 574 remaining dense, attention, shared-expert, indexer, and output tensors:
  `Q4_0_ROCMFP4_FAST` (custom type 101)
- all 43 `attn_output_b` tensors: `Q8_0`
- token embedding: `Q6_K`
- norms, biases, scales, and sinks: `F32`/`I32` as appropriate

The affine expert format decodes each block as:

```text
value = code * scale - offset, where code is one of {0, 1, 2, 3}
```

Its packed blocks remain 2.5 BPW while using both stored scale bytes as an
affine scale and offset rather than two symmetric half-block scales.

## 0731-specific importance matrix

The routed-MoE matrix was regenerated against the official 0731 weights by
collecting expert activations on ROCm:

- calibration artifact:
  [`DeepSeek-V4-Flash-0731-chat-v2-routed-moe-ds4-rocm.dat`](./DeepSeek-V4-Flash-0731-chat-v2-routed-moe-ds4-rocm.dat)
- 4,692 rendered prompts
- 2,897,869 calibration tokens
- 747,650,202 routed-expert observations
- 202,186 quantizer chunks
- complete 129/129 gate/up/down expert-tensor coverage
- no default, zero, duplicate, non-finite, missing, or trailing entries
- imatrix SHA-256:
  `9095857d3f47f94ad32326ff167e92c50b9a23c6176a1ef68d5af0253c67b3f5`

The GGUF embeds the matrix filename, dataset identifier, chunk count, and
entry count in its quantization metadata. The downloadable `.dat` uses the
legacy llama.cpp importance-matrix binary format accepted by the ROCmFPx
quantizer used for this build.

## Runtime validation

Checks this recipe passes end to end:

- header and all 1,328 tensor descriptors parsed successfully
- every special tensor matched the intended type; no missing or unexpected
  overrides
- final tensor end offset exactly equals the physical file size
- the custom Ember ROCm runtime loaded all tensors into an 86,886.9 MiB GPU
  buffer on `gfx1151`
- health, ordinary chat completion, structured tool-call generation, and a
  tool-result continuation all passed
- DSpark speculative decoding initialized and ran during the smoke tests

The benchmark figures above were measured on this artifact and identify the
exact runtime, sampler, and drafter configuration.

## Runtime

Built for and tested with **[Ember](https://github.com/otheru-ai/ember)**
(`ember-dflash`). Any runtime used instead must implement:

- the DeepSeek-V4 (`deepseek4`) graph
- custom ROCmFPx types 100, 101, and 107
- the affine type-107 CPU reference path and ROCm kernels
- the official DeepSeek-V4-0731 chat template, including DSML tool calls

The model runs without speculative decoding. To use the separate DSpark
drafter, the tested runtime configuration was:

```text
DFLASH_DS4_SPEC=1
DFLASH_DS4_DRAFT=/path/to/DeepSeek-V4-Flash-0731-Abliterated-DSpark-draft-4.25bpw.gguf
DFLASH_DS4_SPEC_Q=4
DFLASH_DS4_FUSED_VERIFY=1
```

A matching abliterated drafter is now published in this repository (see
Artifacts). The upstream original is
[`deepseek-ai/DeepSeek-V4-Flash-DSpark`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark),
and the GGUF used during validation came from
[`Lucebox/DeepSeek-V4-Flash-DSpark-Drafter-GGUF`](https://huggingface.co/Lucebox/DeepSeek-V4-Flash-DSpark-Drafter-GGUF).

## Provenance

```text
deepseek-ai/DeepSeek-V4-Flash-0731
revision 9e165c30e2704aec5d9d593cce3eebd58bbef1cb
  -> rank-1 refusal projection, blocks 10-42 + 3 MTP, lambda 3.5
     direction 6e4d8a8f...37fe0 (MIT, drowzeys/keys)
  -> BF16 GGUF (modular DeepSeek-V4 converter)
  -> 0731 ROCm activation collection (imatrix)
  -> affine expert + STRIX_LEAN ROCmFPx quantization
  -> per-tensor type contract check against a known-good GGUF
```

The base model's improved 0731 Jinja chat template is retained in the GGUF.

## License

This artifact inherits the DeepSeek model license. See the base model card and
license for its terms.
