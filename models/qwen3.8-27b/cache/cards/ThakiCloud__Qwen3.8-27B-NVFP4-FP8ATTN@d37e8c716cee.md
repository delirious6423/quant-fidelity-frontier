---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - nvfp4
  - fp8
  - mixed-precision
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

# Qwen3.8-27B-NVFP4-FP8ATTN

Mixed-precision quantization of `Qwen/Qwen3.8-27B`: **NVFP4 (4-bit) on the MLP stack,
FP8 (8-bit) on every attention projection.** **22.90 GB** on disk against the 55.59 GB
bf16 original — a 2.43x reduction.

Built with **llm-compressor**, GPTQ oneshot over 1,024 text calibration samples.

| precision | modules |
|---|---|
| NVFP4 W4A4, group_size 16 | MLP `gate_proj` / `up_proj` / `down_proj` |
| FP8 W8A8 dynamic | `self_attn.*` (16 layers) and `linear_attn.*` (48 layers) |
| bf16 (untouched) | vision tower, `lm_head`, `linear_attn` norms |

## Why this exists, and why you probably want it over our pure-NVFP4 pair

We first published `Qwen3.8-27B-NVFP4-GPTQ-mm` / `-txt`, which quantize **only the MLP**
and leave all 401 language-model attention modules in bf16. That was a worse trade than we
realized, and measuring the two side by side is what showed it:

| build | size | c=1 | c=128 | c=128 vs bf16 |
|---|---|---|---|---|
| `Qwen/Qwen3.8-27B` (bf16) | 55.59 GB | 86.5 | 2,141.4 | — |
| our pure NVFP4 (MLP only) | 30.14 GB | 126.3 | 3,186.2 | 1.488x |
| **this build** | **22.90 GB** | **138.8** | **3,513.3** | **1.641x** |
| **this build + `--kv-cache-dtype fp8`** | 22.90 GB | **138.9** | **3,586.3** | **1.675x** |

Output tokens/s, in=2048 out=256, single B200, ladder 1/8/32/128 with three repeats per
level, median. Serving config is identical across every row and was read back from the
engine rather than assumed: `max_num_seqs=256`, `max_model_len=32768`,
`gpu_memory_utilization=0.90`, `CompilationMode.VLLM_COMPILE`.

**Smaller and faster at the same time.** Using 8-bit in the right place beat using 4-bit in
more places — attention was carrying memory traffic that a blanket-MLP 4-bit pass never
touched.

## Quality — not distinguishable from bf16

MMMU validation, multiple-choice only, `max_new_tokens=16384`, temperature 0, one prompt
shared by every arm, scored by code rather than an LLM judge. Restricted to the **232 items
that produced a verdict in all six arms**, because arms lose different items to truncation
and unpaired accuracies cannot be differenced.

| build | MMMU-val MC (232 paired) | McNemar vs bf16 | discordant pairs |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` (bf16) | 0.8836 | — | — |
| our pure NVFP4 `-txt` | 0.8922 | p = 0.688 | 6 |
| this build | 0.8707 | p = 0.581 | 13 |
| this build + KV FP8 | 0.9009 | p = 0.344 | 10 |

No arm separates from bf16. Read this as **"the quantization did not break it"**, not as a
gain — 4-bit does not add knowledge, and these gaps are well inside the noise at n=232.

⚠️ **n=232 can show the absence of a large effect, not of a small one.** MMMU only;
multiple-choice only. Answer extraction failed on **0** items in every arm.

⚠️ **This build truncates more often than bf16** at a 16,384-token budget (22 items vs 13,
verdict rate 0.896 vs 0.953). The pairing above absorbs that, but we cannot explain why it
reasons longer. Treat it as an open question, not a settled property.

## Quality — math reasoning moves further than on the pure-NVFP4 pair (added 2026-08-22)

The MMMU result above stands. Looking on other axes, one moved.

**GSM8K, n = 600, greedy, vLLM v0.27.1 server + lm-eval 0.4.12, identical serving config
across all three arms:**

| build | strict-match | delta vs bf16 | p |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` (bf16) | 0.5333 | — | — |
| `…-NVFP4-GPTQ-txt` (MLP-only 4-bit) | 0.4783 | −5.50pp | 0.056 |
| **this build** | **0.4500** | **−8.33pp** | **0.004** |

**This build loses more math accuracy than the pure-NVFP4 pair it was meant to improve on**,
and the direction is consistent with what it does: it compresses the attention and SSM
projections that the pair leaves in bf16. More quantization, more loss on the axis that is
sensitive to it.

**Korean and general reasoning are unaffected** (same run, n = 600):

| task | bf16 | this build | delta | p |
|---|---|---|---|---|
| HAERAE | 0.7709 | 0.7654 | −0.55pp | 0.821 |
| KoBEST | 0.6770 | 0.6670 | −1.00pp | 0.712 |
| belebele-ko | 0.9000 | 0.8983 | −0.17pp | 0.922 |
| ARC-Challenge | 0.5700 | 0.5733 | +0.33pp | — |
| IFEval prompt-strict | 0.3050 | 0.3179 | +1.29pp | — |

## Rebuild noise floor — why the GSM8K gap above is attributable (added 2026-09-01)

We separately requantized `Qwen3.8-27B-NVFP4-GPTQ-txt` (same model, same recipe, same
calibration set, second GPTQ run only) to measure how much a GSM8K comparison can move from
the quantization process alone, independent of any real change. That gap is the floor:

| metric | rebuild-to-rebuild gap (floor) |
|---|---|
| GSM8K (en) strict-match | 3.56pp |
| GSM8K (en) flexible-extract | 2.27pp |
| gsm8k_ko strict-match | 2.13pp |
| gsm8k_ko flexible-extract | 1.37pp |
| RAG absent-context fabrication rate | 27.1pp (7.5σ) |

We only call a GSM8K gap attributable to a build's recipe once it clears roughly 2x its
metric's floor — below that, statistical significance alone does not settle it, because the
floor is a second source of variance the p-value does not model.

This build's −8.33pp is 2.34x the 3.56pp floor. It clears the bar, with room to spare but not
by a huge margin — call it attributable but not overwhelming. The `-GPTQ-txt` sibling's
−5.50pp in the table above is only 1.55x the floor and does **not** clear it; see that card's
own noise-floor section for the fuller reading. In other words: quantizing attention as well
as the MLP is the more defensible explanation for the extra math loss here, but the comparison
that shows it is closer to the noise floor than the p-values alone suggest.

## Throughput: the advantage above is regime-specific (added 2026-08-22)

The 1.641x table near the top of this card is measured at **in = 2,048, `max_model_len`
32,768, `max_num_seqs` 256, concurrency 128** — short prompts, high concurrency. It holds
in that regime.

**It reverses in the opposite regime.** Measured on a single B200 with **142k-token prompts,
`max_model_len` 1,000,000 (YARN factor 4), `max_num_seqs` 8**, free-form generation:

| concurrency | `…-GPTQ-txt` | this build |
|---|---|---|
| 1 | 105.9 | 112.5 |
| 4 | 321.7 | 277.0 |
| 8 | **477.8** | 369.9 |

Aggregate output tok/s. This build is ahead at concurrency 1 and behind by 1.29x at
concurrency 8. We have not isolated which of the three changed variables (prompt length,
context window, `max_num_seqs`) drives the reversal, so treat it as **"measure in your own
regime"** rather than as a ranking. What is settled is that the top-of-card table does not
generalize to long-context, low-concurrency serving.

⚠️ `kmmlu_direct` and `humaneval` from the same batch are **excluded** as parsing/format
failures (kmmlu 0.00–0.12 across subjects; humaneval exactly 0.0000 on all arms). `gpqa` is
gated on the Hub and was not run.

## Relationship to `RadixArk/Qwen3.8-27B-NVFP4`

That checkpoint uses the same idea through a different toolchain (`modelopt`): attention at
FP8, MLP at NVFP4, plus a **calibrated static FP8 KV cache**. Measured on our Metis
serverless path the same week, its tuned single-stream throughput is **138.9 tok/s** — the
same figure this build reaches. Same recipe, same number.

⚠️ **At c=128 it reaches 3,845 tok/s against our 3,586.** The candidate explanation is that
its KV scales are calibrated and baked into the checkpoint while ours are set dynamically at
serve time. We have not separated that axis, so we do not claim it.

## Where this fits against W4A16 (added 2026-09-01)

A separate same-run kernel-throughput comparison on this model family (single B200) put a
plain NVFP4 arm at 3.52x the ceiling throughput of a W4A16 (Marlin) arm — 3,597.9 vs 1,022.8
output tok/s — with 3.12x better tok/J and 5.1x lower time-to-first-token (2.093s vs 10.664s).
In that same comparison, an FP8-weighted arm ran at 0.86x of the NVFP4 arm's throughput while
being only 0.25 GB larger — a weak trade next to NVFP4's margin.

For hardware sizing: this build's 22.90 GB fits comfortably on a 32 GB-class card with room
for KV cache, so you do not need to fall back to W4A16 there. Its `-GPTQ-txt` sibling (30.14
GB, MLP-only NVFP4) is the one that runs out of headroom on a 32 GB card — see that card's own
hardware note.

## Long context: native 256k, and fp8 KV cache is nearly free (added 2026-08-31)

This model's `max_position_embeddings` is 262144 — 256k context is native, not an extension.
KV cost is 64 KiB per token: only 16 of the 64 layers are full attention (the other 48 are
linear-attention and contribute effectively nothing to the KV cache), so the accounting is
16 layers x 2 (K+V) x 4 heads x 256 head_dim x 2 bytes.

Turning on `--kv-cache-dtype fp8` on this hybrid architecture nearly doubles the KV pool —
901,744 to 1,759,679 tokens, 1.95x — for what measures as free: throughput was unchanged
(c=1: 66.5 to 67.6 tok/s, c=4: 72.9 to 72.6 tok/s) and quality was unchanged (absent-context
fabrication rate identical to the decimal; extraction score 0.5458 to 0.55, within rounding).
This is the basis for recommending `--kv-cache-dtype fp8` in the Usage command below rather
than treating it as an optional flag.

Ledger: `2026-08-31-metis-edge-rebuild-floor-and-ctx256.json`.

## Usage

```bash
vllm serve ThakiCloud/Qwen3.8-27B-NVFP4-FP8ATTN \
  --max-model-len 32768 --max-num-seqs 256 --kv-cache-dtype fp8
```

Blackwell (SM100+) for the native FP4 kernel path. `--kv-cache-dtype fp8` is what produces
the 1.675x row above; without it you get 1.641x.

## Scope

One model, one recipe, one engine build, one GPU generation. Evaluated on MMMU, GSM8K,
IFEval, HAERAE, KoBEST, belebele-ko and ARC-Challenge. Code and video behaviour are
unmeasured here; `kmmlu_direct` and `humaneval` were excluded as format failures and `gpqa`
is gated on the Hub. Long-context is measured on the throughput axis only (see the regime
section), not on quality. The 1.641x figures are for one traffic shape (in=2048, out=256);
prefill-heavy and decode-heavy mixes are not covered.

Ledger: `2026-08-19-qwen38-27b-recipe-6arm-b200.json`.
