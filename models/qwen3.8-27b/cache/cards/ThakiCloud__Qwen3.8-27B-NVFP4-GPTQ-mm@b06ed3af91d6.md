---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - nvfp4
  - fp4
  - quantized
  - llm-compressor
  - gptq
  - vision-language
  - vllm
pipeline_tag: image-text-to-text
base_model_relation: quantized
library_name: transformers
language:
- en
- ko
---

# Qwen3.8-27B-NVFP4-GPTQ-mm

> ### ⚠️ There is a better build of this model — `ThakiCloud/Qwen3.8-27B-NVFP4-FP8ATTN`
>
> This checkpoint quantizes **only the MLP** and leaves every attention projection at
> bf16. Our mixed build puts attention at FP8 instead, and on the same B200 harness it is
> **smaller and faster at once**: 22.90 GB against 30.14 GB, and 1.641x over bf16 at
> saturation against this build's 1.488x (1.675x with `--kv-cache-dtype fp8`). Neither
> build separates from bf16 on paired MMMU.
>
> This repo stays up because the pair is the evidence for a calibration null result, and
> because the six-arm comparison needs it. **For serving, take the mixed build.**


NVFP4 (4-bit weights and activations) quantization of the **MLP stack** of `Qwen/Qwen3.8-27B`, produced with
**llm-compressor** using a GPTQ oneshot over 1,024 calibration samples calibrated on
**text **and images****. **30.14 GB** on disk against the 55.59 GB bf16 original.

### What is actually 4-bit here (corrected 2026-08-19)

⛔ **An earlier revision of this card said the vision tower was the only thing left at
original precision. That was wrong, and it understated what stayed in bf16 by a factor of
four.** The ignore list holds **511** modules, and only 110 of them are the vision tower:

| kept at bf16 | count |
|---|---|
| `model.language_model.*.linear_attn.*` (48 layers x 7 projections) | 336 |
| `model.language_model.*.self_attn.{q,k,v,o}_proj` (16 layers x 4) | 64 |
| `lm_head` | 1 |
| `model.visual.*` (vision tower) | 110 |

So this checkpoint is **NVFP4 on the MLP stack, bf16 everywhere else** — every attention
projection in the language model is untouched. That, not the vision tower, is the main
reason the reduction is 1.84x instead of the ~3x a blanket 4-bit pass would give.

The measured numbers below are unaffected: they were taken on this checkpoint as it is.
What changes is what you should call it. If you are comparing against
`RadixArk/Qwen3.8-27B-NVFP4`, note that theirs is a **different recipe** — `modelopt`
mixed precision that puts all 208 attention projections at **FP8** and the MLP at NVFP4
with an FP8 KV cache, rather than leaving attention at bf16. Those two checkpoints are not
the same thing under the same name, and a size or score comparison between them is a
comparison of recipes.

## This repo is one half of a pair

`ThakiCloud/Qwen3.8-27B-NVFP4-GPTQ-txt` is the **same model, same recipe, same bit width,
same file size** — calibrated on text only instead. Nothing else differs. `quantize_meta.json`
in each repo records the difference as `multimodal: true`.

The pair exists because of a result we got on a different vision-language model: with the
recipe held fixed, moving from a text-only calibration set to one that includes images
moved MMMU by **1.89pp** on that model. On *this* one the two calibrations are
indistinguishable (see Measured) — so the pair is published as a null result rather
than a winner. Publishing only one would have implied a difference we did not find.

## Measured

Single B200. Serving config **identical across every arm and echoed by the harness rather
than assumed**: `max_num_seqs=256`, `max_model_len=32768`, `gpu_memory_utilization=0.90`,
`CompilationMode.VLLM_COMPILE` + `CUDAGraphMode.FULL_AND_PIECEWISE`.

### Throughput — 1.44x at batch 1, still 1.49x at saturation

Ladder 1/8/32/128, three repeats per level, median. Output tokens/s, in=2048 out=256.

| concurrency | `Qwen/Qwen3.8-27B` (bf16) | this build | speedup |
|---|---|---|---|
| 1 | 86.5 | 124.4 | **1.44x** |
| 8 | 565.4 | 811.4 | **1.44x** |
| 32 | 1,382.4 | 2,013.4 | **1.46x** |
| 128 | 2,141.4 | 3,189.0 | **1.49x** |

The gain holds at saturation, which is where serving actually lives — a single-stream
figure alone would not tell you that. The sibling build, calibrated separately, lands
within 0.002 of these ratios at every level.

### Quality — no measurable change against bf16

MMMU validation, multiple-choice only (53 open-ended items skipped rather than guessed at),
`max_new_tokens=16384`, temperature 0, one prompt shared by every arm. Scored by code, not
by an LLM judge. **Paired**: the table below is restricted to the 246 items that produced a
verdict in all three arms, because arms lose different items to truncation and unpaired
accuracies are not differenceable.

| build | MMMU-val MC (246 paired) | vs bf16 | McNemar |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` (bf16, 55.59 GB) | 0.8455 | — | — |
| this build (30.14 GB) | 0.8537 | +0.8pp | p = 0.754 (10 discordant pairs) |

**There is no detectable quality difference on MMMU.** Nor is there one between the two
calibrations. That statement is narrower than it used to be — see the math section below. Read that as "quantization did not break it", not as a gain — 4-bit does not
add knowledge, and a difference this size is well inside the noise at n=246.

⚠️ **n=246 can show the absence of a large effect, not of a small one.** MMMU only.

⛔ **An earlier revision of this card listed bf16 MMMU at 0.4511. That number is withdrawn.**
It came from a harness that capped generation at 512 tokens on a model that spends 12,000+
characters reasoning before it answers, so it scored truncated reasoning as wrong answers.
Chance on 4-way multiple choice is 0.25; a "score" near it should have been read as a broken
measurement, and for a day it was not. The same run had a judge-scored task return **exactly
0.0** — `lmms_eval/tasks/mmmu/utils.py` converts a judge failure into `judge_score = 0` with
nothing but a log line, so a missing judge server is indistinguishable from a model that
gets everything wrong. Both numbers are gone; these replace them.

Answer extraction failed on **0** items across all three arms.

### Quality — math reasoning moves on the sibling build (added 2026-08-24)

The MMMU result above stands within its scope. On another axis, one moved — and it was
measured on `ThakiCloud/Qwen3.8-27B-NVFP4-GPTQ-txt`, **not on this build**.

⚠️ **Read this as transferred evidence, not as a measurement of this checkpoint.** The two
repos are the same recipe, the same bit width and the same file size, differing only in the
calibration set, and MMMU could not separate them. That makes the transfer plausible. It
does not make it measured. We have not run GSM8K on this build.

**GSM8K, full 1,319-item set, greedy, vLLM v0.27.1 server + lm-eval 0.4.12, identical
serving config on both arms — measured on the `-txt` sibling:**

| build | strict-match | flexible-extract |
|---|---|---|
| `Qwen/Qwen3.8-27B` (bf16) | 0.5413 | 0.5451 |
| `…-NVFP4-GPTQ-txt` (sibling) | 0.4936 (**−4.77pp**) | 0.5239 (−2.12pp) |

Two-proportion test on strict-match: z = −2.45, **p = 0.014**. An independent rerun of the
same comparison returned −4.32pp (p = 0.026), so this is not a single-run artifact.

**Korean and general reasoning show no such loss** on the sibling (n = 600 per task): HAERAE
−1.01pp (p = 0.680), KoBEST −0.25pp (p = 0.926), belebele-ko +0.83pp (p = 0.625),
ARC-Challenge +0.50pp, IFEval prompt-strict +0.92pp.

**So the honest summary is narrower than "no measurable change".** On MMMU, Korean,
commonsense reasoning and instruction following we cannot distinguish this recipe from bf16.
On GSM8K we can, and it is worse. This card previously listed math in neither the measured
nor the excluded column — that blind spot is what this section closes. If your workload is
math or multi-step arithmetic reasoning, measure before you switch; and if you need the
number for *this* build specifically, it does not exist yet.

Ledgers: `2026-08-22-qwen38-27b-precision.json`, `2026-08-22-qwen38-multi-benchmark.json`.

## Kernel path

A 4-bit checkpoint can hold NVFP4 weights and still expand them to bf16 for every matmul —
slower than bf16, saving only memory. Verified on B200 (SM100) with vLLM 0.27.1 that this
build reaches a native FP4 kernel:

```
KERNEL backend markers=['mm_fp4_sm100a'] fallback=False is_fp4=True VERDICT=native
```

⚠️ **Blackwell only.** On Hopper (H200/H100) vLLM falls back to weight-only Marlin
emulation, which we measured at roughly 0.85x of bf16 on a comparable model. NVFP4 is not
a speed win on SM90.

## Usage

```bash
vllm serve ThakiCloud/Qwen3.8-27B-NVFP4-GPTQ-mm --max-model-len 32768
```

## Provenance

`quantize_meta.json` ships in this repo: GPTQ oneshot, 1,024 samples, 2556 s, and the
calibration flag that distinguishes this build from its sibling. Check the card against it.

## Scope

One model, one recipe, one engine version, one GPU generation for the kernel claim.
Directly evaluated on **MMMU and ChartQA only**. Math reasoning is covered by transfer from
the `-txt` sibling (see above), not by measurement on this build. Code, long-context,
multilingual and video behaviour are unevaluated and unclaimed.
