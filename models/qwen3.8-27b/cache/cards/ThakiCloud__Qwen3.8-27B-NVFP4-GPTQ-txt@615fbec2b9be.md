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

# Qwen3.8-27B-NVFP4-GPTQ-txt

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
**text only**. **30.14 GB** on disk against the 55.59 GB bf16 original.

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

`ThakiCloud/Qwen3.8-27B-NVFP4-GPTQ-mm` is the **same model, same recipe, same bit width,
same file size** — calibrated on text and images instead. Nothing else differs. `quantize_meta.json`
in each repo records the difference as `multimodal: false`.

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
| 1 | 86.5 | 126.3 | **1.46x** |
| 8 | 565.4 | 814.4 | **1.44x** |
| 32 | 1,382.4 | 2,013.1 | **1.46x** |
| 128 | 2,141.4 | 3,186.2 | **1.49x** |

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
| this build (30.14 GB) | 0.8577 | +1.2pp | p = 0.453 (7 discordant pairs) |

**There is no detectable quality difference.** Nor is there one between the two
calibrations. Read that as "quantization did not break it", not as a gain — 4-bit does not
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

### Quality — math reasoning does move (added 2026-08-22)

The MMMU result above stands, and so does its caveat: n=246 cannot see a small effect.
We went looking on other axes and found one.

**GSM8K, full 1,319-item set, greedy, vLLM v0.27.1 server + lm-eval 0.4.12,
identical serving config on both arms:**

| build | strict-match | flexible-extract |
|---|---|---|
| `Qwen/Qwen3.8-27B` (bf16) | 0.5413 | 0.5451 |
| this build | 0.4936 (**−4.77pp**) | 0.5239 (−2.12pp) |

Two-proportion test on strict-match: z = −2.45, **p = 0.014**. An independent rerun of the
same comparison returned −4.32pp (p = 0.026), so this is not a single-run artifact.
Flexible-extract moves −2.12pp (p = 0.275), inside noise. That the gap appears under strict
but not flexible extraction points at **format adherence rather than answer correctness** —
but that reading comes from the difference between two metrics and is not itself measured.

**Korean and general reasoning show no such loss** (n = 600 per task, same harness):

| task | bf16 | this build | delta | p |
|---|---|---|---|---|
| HAERAE | 0.7709 | 0.7608 | −1.01pp | 0.680 |
| KoBEST | 0.6770 | 0.6745 | −0.25pp | 0.926 |
| belebele-ko | 0.9000 | 0.9083 | +0.83pp | 0.625 |
| ARC-Challenge | 0.5700 | 0.5750 | +0.50pp | — |
| IFEval prompt-strict | 0.3050 | 0.3142 | +0.92pp | — |

**So the honest summary is narrower than "no measurable change".** On MMMU, Korean,
commonsense reasoning and instruction following we cannot distinguish this build from bf16.
On GSM8K we can, and it is worse. If your workload is math or multi-step arithmetic
reasoning, measure before you switch.

⚠️ `kmmlu_direct` and `humaneval` were run in the same batch and are **excluded**: kmmlu
scored 0.00–0.12 across every subject and humaneval returned exactly 0.0000 on all three
arms, which we read as parsing/format failure rather than accuracy. `gpqa` was not run — it
is gated on the Hub.

### Speculative decoding on this checkpoint (added 2026-08-23)

We served this checkpoint with the **DFlash2 drafter** (`z-lab/Qwen3.8-27B-DFlash2`, K=7)
on a vLLM nightly build and measured two things.

**Quality is unchanged.** Holding image, checkpoint and task set fixed and toggling only the
drafter, **none of 34 metrics separated** — GSM8K, IFEval, HAERAE, KoBEST and belebele-ko
included, minimum p = 0.774 (GSM8K strict −0.83pp at p=0.774; six metrics identical to the
decimal). That is what rejection sampling should do, but it is worth measuring rather than
assuming: the losslessness is a property of the algorithm, not automatically of a build.

**⛔ The engine dies on very long prompts, and the failure is not graceful.** A length ladder
on a `max_model_len=1000000` endpoint handled 8,410 / 66,968 / 148,503 / **244,689 tokens**
cleanly (acceptance length steady at 1.42–1.47), then returned HTTP 500 at roughly 300k and
**killed the EngineCore process** — `CUDA error: cudaErrorIllegalAddress`
(`CUDAGraphsC10Utils.h:73`). One request takes the endpoint down.

If you enable a drafter here, **cap `max_model_len` at or below ~245k**. We set 245,760 and
re-verified: 240,503 tokens serve normally and anything larger returns a clean HTTP 400
instead of crashing. We have not bisected the exact boundary between 244,689 and ~300k.

**The drafter also costs 25% of the KV pool** — 1,808,112 → 1,354,786 tokens under otherwise
identical settings, so concurrency for million-token requests falls from 1.81 to 1.35. In
long-context serving the drafter's price is capacity, not latency.

**Update (2026-08-31).** A follow-up run pinned down two more things about this drafter.

Launch config, for reproducibility: `z-lab/Qwen3.8-27B-DFlash2` as the draft model, served on
vLLM 0.28.0 with
`--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7}' --mamba-cache-mode align`.

**The KV overhead is a proportion of the pool, not a fixed token count.** Switching
`--kv-cache-dtype fp8` on the target checkpoint does not shrink the drafter's relative cost —
it stays in the 34–36% range regardless of KV dtype (measured separately from the 25%
figure above; both point the same direction, that the overhead scales with pool size rather
than being absorbed by a cheaper KV representation).

**The speedup does not automatically transfer to other target checkpoints.** On the vendor's
own FP8 build, the same drafter helps across the whole concurrency range we tested (c=1–8). On
a third-party INT4 build (RedHatAI), it *loses* throughput at c=8 — 0.82x of baseline, an 18%
hit. We have not re-run that same ladder on this checkpoint, so treat the 34-metric
quality-neutral result above as specific to this build, and re-verify before assuming a
drafter's win on one target checkpoint carries over to another.

## Rebuild noise floor — why the GSM8K gap above is not attributable (added 2026-09-01)

We requantized this exact model with this exact recipe and this exact calibration set a
second time, changing nothing except letting GPTQ run again. That second run is not a
different model — it is the same instructions executed twice — so whatever daylight opens up
between the two builds is the floor under any GSM8K comparison made with this recipe, not a
property of the model.

| metric | rebuild-to-rebuild gap (floor) |
|---|---|
| GSM8K (en) strict-match | 3.56pp |
| GSM8K (en) flexible-extract | 2.27pp |
| gsm8k_ko strict-match | 2.13pp |
| gsm8k_ko flexible-extract | 1.37pp |
| RAG absent-context fabrication rate | 27.1pp (7.5σ) |

**Rule we use:** an observed gap has to clear roughly 2x its metric's floor before we call it
attributable to the treatment (here, leaving attention at bf16). Below that, a low p-value
does not rescue it — the p-value assumes the only source of variance is sampling noise on a
fixed model, and the floor shows there is a second, unmodeled source: which quantization run
you happened to get.

Applying that to the GSM8K numbers on this checkpoint:

| measurement | gap | floor | ratio | verdict |
|---|---|---|---|---|
| full set (n=1,319) strict-match, above | −4.77pp | 3.56pp | 1.34x | **not attributable** |
| n=600 subsample (quoted in the `-FP8ATTN` card's comparison table) | −5.50pp (p=0.056) | 3.56pp | 1.55x | **not attributable** |
| flexible-extract, above | −2.12pp | 2.27pp | 0.93x | already flagged as noise |

Neither GSM8K number for this checkpoint clears the bar. That does not mean "leaving attention
at bf16 doesn't hurt GSM8K" — it means we cannot tell from this data whether it does. The
sibling `ThakiCloud/Qwen3.8-27B-NVFP4-FP8ATTN`, which quantizes attention as well, shows a
larger gap (−8.33pp) that *does* clear the same floor, by 2.34x — see that card for the
reading that number supports.

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

## Does this fit your GPU? (added 2026-09-01)

A separate same-run kernel-throughput comparison on this model family (single B200) put a
plain NVFP4 arm at 3.52x the ceiling throughput of a W4A16 (Marlin) arm — 3,597.9 vs 1,022.8
output tok/s — with 3.12x better tok/J and 5.1x lower time-to-first-token (2.093s vs 10.664s).
That is the case for NVFP4 on hardware that can hold it.

The catch is exactly what this card is about: **this build's 30.14 GB does not leave much
headroom on a 32 GB-class card** once you account for KV cache and runtime overhead. On that
class of card, W4A16 is the build that actually fits; on a 96 GB-class card, an NVFP4-family
build is worth the squeeze for the throughput and tok/J win above — and at that point you
should reach for `ThakiCloud/Qwen3.8-27B-NVFP4-FP8ATTN` (22.90 GB) rather than this one, per
the banner at the top of this card.

## Usage

```bash
vllm serve ThakiCloud/Qwen3.8-27B-NVFP4-GPTQ-txt --max-model-len 32768
```

## Provenance

`quantize_meta.json` ships in this repo: GPTQ oneshot, 1,024 samples, 2367 s, and the
calibration flag that distinguishes this build from its sibling. Check the card against it.

## Scope

One model, one recipe, one engine version, one GPU generation for the kernel claim.
Evaluated on MMMU, ChartQA, GSM8K, IFEval, HAERAE, KoBEST, belebele-ko and ARC-Challenge.
Code and video behaviour are unevaluated and unclaimed; `kmmlu_direct` and `humaneval` were
run and excluded as format failures, and `gpqa` is gated on the Hub and was not run.
Long-context is covered on the speed axis only (the drafter ladder above), not on quality.
