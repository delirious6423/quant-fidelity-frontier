---
license: other
license_name: qwen-community-license-1.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- llama.cpp
- moe
- pruning
- reap
- qwen
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next REAP-320 "multi-domain" — GGUF

The successor to [REAP-256 "duo"](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF): **512 → 320 experts per layer, 83.8 → 64.2 GiB** (64.2 GiB = **68.96 GB** decimal — the number HF's file listing shows; every size on this card is GiB, the same convention the compared models use), calibrated on a **multi-domain corpus** instead of code-only. Built directly in response to [this community critique](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF/discussions/2) — the full head-to-head against Unsloth's UD-IQ1_M, including where this build loses, is published in that thread.

**Not an official Qwen release.** Community derivative, unaffiliated with Alibaba/Qwen.

## Why K=320 and why multi-domain

- **K=320 sits above the reliability knee.** Two independent measurements (sh0wie's REAP ladder, and our own K=256 lineage) show rare-token reliability and clean termination degrading below ~K=288-320. At K=320: zero empty-content stalls and zero truncation across all our battery passes, 9/10 on the seeded rare-name probe.
- **Calibration is the product.** Salience was measured with `llama-imatrix` over a 525K-token corpus: 30% agentic / 30% code / 15% conversation / 12.5% math / 12.5% writing. The multi-domain build beat the code-only specialist **even at code** (HumanEval 96.3% vs 95.7%), and we have since reproduced the effect at K=256, where changing *only* the corpus moved HumanEval from 89.6% to 95.1% at identical size and identical bits.

> **Correction (2026-09-01).** An earlier version of this card claimed this selection "retains 82.7% of routing mass vs 75.3% for a code-only selection at the same K." That number is withdrawn. It was measured against a corpus drawn from the same mixture that produced this selection, and the metric turns out to be circular — **each selection wins when scored against the corpus that produced it.** We then ran it as a genuine prediction on a K=256 rebuild: retained routing mass said the new selection would be *worse* (82.7% vs 78.0% for the incumbent), and every behavioural probe said the opposite, decisively. Retained routing mass is useful for understanding a prune; it does not rank two prunes. The HumanEval, rare-name and fabrication numbers below are unaffected — those are measured on the model, not on the selection.

> ⚠️ **Throughput figures in the tables below are superseded.** They have been corrected
> twice — once for a protocol error and once for a bug in the correction itself. Read
> [Corrections to this repository](#corrections-to-this-repository) at the bottom before
> citing any tok/s number from this card. Quality figures (HumanEval, rare-name,
> fabrication) are unaffected throughout.

## Measured (all on RTX 5090 32 GB, llama.cpp + expert-cache PR #27861 build)

| metric | this build | unpruned baseline |
|---|---|---|
| 6-task machine-verified battery (3 passes, effort medium) | **13.0 / 13.0 / 12.0** | 12 / 13 / 11 (UD-Q2_K_XL) |
| HumanEval-164 pass@1 (seed 42, effort low) | **96.3%** (158/164) | 95.7% (UD-IQ1_M) |
| Rare-name probe (10 seeded gens) | 9/10 | 8/10 (Q2), 5/10 (IQ1_M, truncation) |
| General-knowledge fabrication (12 questions) | **25%** | **0% (UD-IQ1_M)** |
| Decode with LRU-128 expert cache, 32 GB card | ~42-45 tok/s | ~42-46 (Q2 full, same cache) |
| Prefill, same regime | **337 tok/s (+23%)** | 273 tok/s |

**Honest limitation, stated loudly:** pruning experts costs world knowledge. On our 12-question fabrication probe this build fabricates 25% (the old K=256 duo: 42%; unpruned UD-IQ1_M: 0%). If you want a general-purpose assistant, take [Unsloth's UD-IQ1_M](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) — experts store knowledge, bits store precision. Take this build when **fit and speed in constrained VRAM** dominate: it is 5 GiB smaller than the 1-bit route, prefills ~23% faster under the same cache regime, and generates concisely.

## New: Q2 variant — same experts, fewer bits, fully GPU-resident on 32 GB

`Q2/` holds a second quant of the **same K=320 selection** (same manifest, same imatrix), pruned by binary copy from Unsloth's `UD-Q2_K_XL`: 2 shards, **57.3 GiB (61.5 GB decimal)**. That is the same disk footprint as the legacy [K=256 duo](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF) at Q3 (57.7 GiB) — which makes it a controlled bits-vs-experts comparison at constant size:

| metric (same battery, same rig) | **Q2 @ K=320** | Q3 @ K=256 (duo, same disk) | Q3 @ K=320 (main quant) |
|---|---|---|---|
| HumanEval-164 pass@1 | **95.1%** (156/164) | 89.6% | 96.3% |
| Rare-name probe | **9/10** | 7/10 | 9/10 |
| Fabrication (12 questions) | **25%** | 42% | 25% |
| Decode, `--n-cpu-moe 3` (30.3 GiB VRAM) | **86.9 tok/s** | — | doesn't fit |
| Decode, fully resident `--n-cpu-moe 0` | 98.5 tok/s (31.8 GiB — thin margin, see below) | — | doesn't fit |
| Decode, LRU-128 expert cache | 47.6 tok/s | — | ~42-45 |

At equal disk, **64 extra experts per layer beat 1 extra bit per weight on every knowledge metric**, and cost only 1.2 HumanEval points vs the Q3 build while freeing 7 GiB — enough to make the whole compute path GPU-resident on a 32 GB card. Recommended residency: `--n-cpu-moe 3` (leaves ~2 GiB VRAM margin; `--n-cpu-moe 0` measures faster but runs the card at ~98% and risks driver timeouts under load).

## Files

Main quant: `UD-Q3_K_XL`-lineage (pruned by binary copy from [Unsloth's quant](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) — surviving experts keep their original bits, no requantization). 2 shards, 64.2 GiB (68.96 GB) total. Plus the `Q2/` variant above: 2 shards, 57.3 GiB (61.5 GB).

`manifests/seleccion_mass_K320.json` ships the per-layer kept-expert manifest (`{"<layer>": [expert ids...]}`, stock ids 0-511) — the prune is reproducible from the source quant, and the manifest is directly usable by SSD-expert-streaming runtimes as a pinned set.

## Run it

Needs llama.cpp mainline ≥ 2026-08-27 (PR #27742, `qwen4exp` support):

```bash
llama-server -m Qwen3.8-Flash-Next-UD-Q3_K_XL-reap320-00001-of-00002.gguf \
  -ngl 99 --cpu-moe -c 32768 -fa on --jinja --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

On a 32 GB card, add the expert-cache from [PR #27861](https://github.com/ggml-org/llama.cpp/pull/27861) (`--moe-expert-cache 128`) for ~42-45 tok/s decode. Set reasoning effort explicitly (`"chat_template_kwargs": {"reasoning_effort": "low"}`) — the template defaults to `xhigh` and replies take minutes.

For the `Q2/` variant on a 32 GB card, skip the cache entirely and run near-resident — this is the fast path (86.9 tok/s decode, ~1000 tok/s prefill):

```bash
llama-server -m Q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-reap320-00001-of-00002.gguf \
  -ngl 99 --n-cpu-moe 3 -c 32768 -fa on --jinja --parallel 1 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

## New: run the Q2 with Unsloth's MTP head (+32% decode, costs 3.2 GiB)

Unsloth published standalone MTP heads for Qwen3.8-Flash-Next on 2026-09-01. They work on
this pruned Q2 build — the head is not affected by expert pruning — and are the single
biggest speed lever we have found for it. Measured on the same 32 GB card, same GGUF,
same flags, paired:

| `--n-cpu-moe 8`, `-c 65536`, KV `q4_0` | VRAM | decode |
|---|---|---|
| no speculation | 28104 MiB (86.2%) | 73.6 tok/s |
| **`--spec-type draft-mtp`** | 31276 MiB (95.9%) | **97.5 tok/s** |

Draft acceptance is 76–81% (mean accepted length 2.5–2.6). Quality is preserved:
HumanEval-164 94.5% with MTP vs 95.1% without — Unsloth's verification is exact, and the
one-problem gap is sampling noise.

```bash
# Needs a build with the Flash-Next MTP graph, NOT stock mainline (see "Which build" below and trap 4)
llama-server -m Q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-reap320-00001-of-00002.gguf \
  -md mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -ngl 99 --n-cpu-moe 8 -c 65536 -fa on --jinja --parallel 1 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0
```

**Which build.** Stock llama.cpp mainline has the `--spec-type draft-mtp` flag but not the
Qwen3.8-Flash-Next MTP graph, and not the loader that lets the *shared* heads borrow
`token_embd` / `output` from the main model. Any of these three works (same list as Unsloth's
`MTP/README.md`):

1. **Unsloth's fork, release tag `b10715-mix-86bd2d3` or newer** — https://github.com/unslothai/llama.cpp/releases.
   The prebuilt binaries there are **CUDA and CPU only**. For **ROCm** (and for Windows) take the
   *source tarball* of that tag and build it exactly like you build mainline (`-DGGML_HIP=ON` etc.).
   This repository's numbers come from a `b10798-mix` source build on Windows/CUDA.
2. Upstream PR https://github.com/ggml-org/llama.cpp/pull/28243 (the same code on its way to mainline; still a draft as of 2026-09-09).
3. Unsloth fork PR #144.

On a correct build the shared head prints **one error line at startup about borrowing the
embeddings**. That line is normal; it works. The `shared-Q4_K_M` head (1.78 GiB) saves another
~840 MiB of VRAM versus `shared-Q8_0` at the same acceptance (0.70 vs 0.72 measured), useful on 24 GB cards.

The head costs **~3.2 GiB of VRAM, flat** — the same at every offload level we tried
(`--n-cpu-moe` 14 / 12 / 10 / 8), so budget for it once. On a 32 GB card at 96k context,
`--n-cpu-moe 10` is the balanced point (30578 MiB, 93.8%, 92.5 tok/s); `8` is the fast one
but leaves under 1 GiB free.

> **A note on how we measured this, because we got it wrong first.** Our initial harness
> reported 369 tok/s for this configuration — a 3.6× speedup. It was wrong. It timestamped
> only stream chunks carrying `content`, while this model emits its reasoning in
> `reasoning_content`; that time landed in "time to first token" while its tokens stayed in
> the numerator. The inflation scales with how much the model reasons, so it was invisible
> on a short prompt and enormous on a long one. The +32% above is wall-clock, cross-checked
> against the server's own `predicted_per_second`. **If you benchmark a reasoning model,
> make sure your timer sees the reasoning stream** — and require two independent clocks to
> agree before you believe a number.

## New (2026-09-06): prefill — the PLE table is why your prompt speed swings 3×, and `--lazy-mode on-direct` fixes it

Flash-Next carries a 26.8 GiB per-layer-embedding table (`per_layer_token_embd.weight`, 320M rows × 90 B) that
is untouched by expert pruning and stays on the host. llama.cpp reads it through `mmap`: one page fault per
~90-byte row, on every token of the prompt. Whether those pages are in the OS cache decides your prefill
speed — and on a machine that also loads other 20–60 GB models, they usually are not. Our production log
showed 245–2652 tok/s on 11–25k-token prompts for exactly this reason, and repetitive benchmark prompts
(few distinct rows) hide it completely.

[PR #28136](https://github.com/ggml-org/llama.cpp/pull/28136) adds `--lazy-mode on-direct`: positioned reads
from a worker pool instead of demand faults. Measured on the `Q2/` build (RTX 5090, Windows 11, 95 GB RAM,
NVMe) with a genuinely cold cache (`RAMMap -Et` + fresh process per arm), a real 25k-token mixed prose+code
prompt, `cache_prompt: false`, two cycles:

| prefill tok/s | cold | warm (same prompt) | different rows, warm process |
|---|---:|---:|---:|
| `--lazy-mode on` (mmap, the default) | **167** | 681 | 442 |
| **`--lazy-mode on-direct`** | **621** | 656 | 619 |
| `--lazy-mode off` (table resident, +26.8 GiB RAM) | 590 | 673 | 573 |

`on-direct` is the one that gives the same number every time; `off` is fast while the resident table
survives but loses it whenever Windows trims the working set. With Unsloth's MTP head loaded (same
protocol): 259 → 506 cold, 404 → 604 on new rows, decode unchanged (72 tok/s). A 28k-token prompt now
prefills cold in 44 s instead of 74–152 s.

To use it today: the PR is still open, it is **POSIX-only as written** (Windows prints
`--lazy-mode on-direct is not supported on this platform` and silently falls back to mmap — you need
[this Windows port](https://gist.github.com/mjungnickel18/083da5cde978ca3b102952029d4a991a) from the PR
thread), and the prebuilt Unsloth binaries reject the value. We build Unsloth's b10798 mix source + the PR +
the Windows port, which keeps the MTP head working. Full numbers, including the Windows-native table, are
[in the PR thread](https://github.com/ggml-org/llama.cpp/pull/28136). If you cannot build:
`--lazy-mode off` (mainline and Unsloth builds accept it) gets most of the cold gain for 26.8 GiB of RAM.

## Traps (measured, not theoretical)

1. `--parallel > 1` requires `--kv-unified`.
2. Empty `content` with tight `max_tokens`: reasoning burns the budget. Detect and retry with a higher cap.
3. First requests after a cold load pay disk; throughput climbs over ~3-4 requests to its warm ceiling. The disk cost is the PLE table (see the prefill section above); `--lazy-mode on-direct` removes it.
4. **`--spec-type draft-mtp` exists in llama.cpp mainline but does not work here.** Mainline has the flag and not the MTP graph for `qwen4exp`, nor the tensor borrowing the *shared* heads need. Symptoms on mainline: with a `shared-*` head, `tensor 'token_embd.weight' not found` → `failed to load draft model` (the head omits the embeddings on purpose, it is not a corrupt download); with a fused model, `model doesn't contain MTP layers`. The non-shared head is not a workaround: it gets past the missing tensor and fails later for the same reason. Use one of the builds listed under "Which build" above (ROCm users: build the Unsloth tag from source).
5. `"reasoning_effort": "none"` returns **HTTP 500** — the chat template accepts only `xhigh` / `medium` / `low`.

## Credits

- [Qwen](https://huggingface.co/Qwen) for Qwen3.8-Flash-Next (Qwen Community License 1.0).
- [Unsloth](https://huggingface.co/unsloth) for the UD-Q3_K_XL quant this was pruned from, and for UD-IQ1_M, the honest benchmark opponent.
- [Cerebras Research](https://github.com/CerebrasResearch/reap) for the REAP method.
- [sh0wie](https://huggingface.co/sh0wie) for independently mapping the pruning-reliability knee and publishing saliency manifests.
- The critic in [discussion #2](https://huggingface.co/AnonimousA/Qwen3.8-Flash-Next-REAP-256-duo-GGUF/discussions/2), whose pushback made this model exist.

---

## Corrections to this repository

**2026-08-31 — every throughput number in this card was measured with a flawed protocol, and
the context they were measured at was not stated. Both are corrected below. One correction
favours this build; the other does not.**

### What was wrong

**1. The decode rate included prefill.** Throughput was computed as *total wall time ÷ output
tokens*. That folds time-to-first-token into what was reported as a decode rate, so every decode
figure in this card was **too low**. The corrected protocol streams the response and measures only
the interval between the first and last token, with `ignore_eos` forcing a fixed output length so
the sample size cannot vary. Protocol borrowed from
[`jpezzulli/pennyroyal-validation`](https://github.com/jpezzulli/pennyroyal-validation).

Note for anyone reproducing this: `llama-server`'s own `timings.predicted_per_second` tracks the
**old, prefill-contaminated** figure, not the post-first-token decode rate. Do not use it.

**2. The numbers were measured at 8,192-token context, and this card recommends a configuration
that collapses at the context it is served at.** The context was never stated. For the
`--n-cpu-moe 4` configuration this turns out not to matter — 94.37 tok/s at 8k versus 93.45 at
98,304, inside noise. **But for the `--n-cpu-moe 0` "fast path" this card recommends, it matters
enormously**, and not in the way a reader would be able to detect:

| `--n-cpu-moe 0` | VRAM | decode | TTFT (code) |
|---|---|---|---|
| at 8,192 ctx | 31,658 MiB | 116.14 tok/s | 1.83 s |
| at 98,304 ctx | 32,168 MiB (**98.7%**) | **10.82 tok/s** | **17.83 s** |

**It does not fail. It loads, it answers, and it runs 10.7× slower.** At 98.7% of a 32 GB card the
driver begins paging and throughput collapses silently. The only trace is one warning that says
nothing about the consequence:

```
common_fit_params: failed to fit params to free device memory: n_gpu_layers already set by user to 99, abort
```

This card already warned that `--n-cpu-moe 0` *"runs the card at ~98% and risks driver timeouts
under load"*, and recommended `--n-cpu-moe 3` instead. That warning was **directionally right and
mechanically wrong**, in a way that makes it easy to ignore:

- it said **"under load"** — this happens at **one concurrent request**, with nothing else on the card;
- it said **"driver timeouts"** — there is no timeout. Nothing errors. It just gets 10.7× slower;
- it gave no magnitude, so "risks timeouts" reads like an edge case rather than a 10× cliff.

**Do not run `--n-cpu-moe 0` at long context on a 32 GB card.** Use `--n-cpu-moe 4`: it is slower
at short context and it is the only one measured here that holds its throughput at 98,304.

### Corrected table

All figures: post-first-token decode, `ignore_eos` with 1,024 forced output tokens, temperature
1.0 / top-p 0.95 / top-k 20 / min-p 0.0, `reasoning_effort: low`, warm-up pass discarded, single
RTX 5090 (32 GB) with no other process on the GPU.

Two prompts are reported because they differ a lot on this build: a **prose** prompt written to
force continuously distinct output (which suppresses n-gram draft hits), and a **code** prompt
(where the n-gram drafter earns its keep).

| configuration | context | VRAM | decode (prose) | decode (code) | TTFT | previously claimed |
|---|---|---|---|---|---|---|
| **`--n-cpu-moe 4` (as served, recommended)** | 98,304 | 30,730 MiB | **93.45** | **126.95** | 1.11 / 3.07 s | not listed |
| `--n-cpu-moe 4` | 8,192 | 29,552 MiB | 94.37 | 126.53 | 1.15 / 3.02 s | not listed |
| `--n-cpu-moe 3` | 8,192 | 30,108 MiB | 103.66 | 106.84 | 1.64 / 2.18 s | 86.9 tok/s |
| `--n-cpu-moe 0` (resident) | 8,192 | 31,658 MiB | 116.14 | 129.12 | 0.95 / 1.83 s | 98.5 tok/s |
| ⚠️ `--n-cpu-moe 0` (resident) | 98,304 | 32,168 MiB (98.7%) | **10.82** | **12.08** | 8.92 / 17.83 s | — |
| `--cpu-moe --moe-expert-cache 128` | 8,192 | ~22 GB | 58.46 | 56.58 | 2.20 / 3.63 s | 47.6 tok/s |
| `--cpu-moe --moe-expert-cache 128` | 98,304 | ~22 GB | 61.31 | 58.21 | 2.34 / 3.60 s | — |
| **`--moe-expert-cache 192 --cache-reuse 256`** | 98,304 | ~22 GB | **70.52** | **71.68** | 2.12 / 3.80 s | — |

**Q3 main quant** (`UD-Q3_K_XL`-lineage, 64.2 GiB), same protocol. Note this build is served
**without** speculative decoding, where the Q2 rows above carry `--spec-type ngram-map-k4v`:

| configuration | context | decode (prose) | decode (code) | TTFT | previously claimed |
|---|---|---|---|---|---|
| `--cpu-moe --moe-expert-cache 128` | 8,192 | 54.09 | 53.89 | 2.40 / 4.69 s | ~42-45 tok/s |
| `--moe-expert-cache 192 --cache-reuse 256` | 98,304 | 66.59 | 66.22 | 2.74 / 3.97 s | — |

An earlier independent run of the `--n-cpu-moe 4` / 98,304 row two hours before this sweep gave
94.61 / 127.47 — 1.2% from the figure above, which is the reproducibility this protocol delivers.

### Two things the corrected numbers show that the old ones hid

**The expert cache is not the slow option — it is the option that holds.** At 8,192 tokens
`--n-cpu-moe 0` beats the cache almost 2:1 (116.14 vs 58.46). At the 98,304 this build is served
at, the cache is **6.5× faster** (70.52 vs 10.82), because it leaves ~10 GB of headroom instead of
running the card at 98.7%. The original table presents the cache as the bottom rung. At real
context it is the top one.

**`--cache-reuse` speeds up decode, not just prefill.** `--moe-expert-cache 192 --cache-reuse 256`
against `--moe-expert-cache 128` alone: **+15% prose, +23% code**, both at 98,304. It was adopted
here for time-to-first-token; the generation gain was not measured until now. If you run the
expert cache, run it with `--cache-reuse`.

### What this does and does not change

Nothing about the weights changed. No re-quantisation, no re-prune. This is a measurement
correction and a disclosure of a condition that should have been in the table from the start.

The quality claims in this card are **unaffected and unretracted**: 95.1% HumanEval, 9/10, and the
25% fabrication rate on the 12-question probe stand as published. The trade this build makes —
experts store knowledge, bits store precision — is not a throughput claim and does not move.

**If you are choosing between this and [Unsloth's UD-IQ1_M](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF),
the corrected numbers do not change that advice.** That build fabricates 0% where this one
fabricates 25%. Take this one when fit and speed in constrained VRAM dominate; take theirs when
you want general knowledge.

---

**2026-09-01 — the correction above was itself measured wrong. The throughput numbers in
it are inflated; the advice it gives about `predicted_per_second` is backwards. The
`--n-cpu-moe 0` collapse it documents is real and stands.**

### The bug in our corrected harness

The 2026-08-31 protocol streams the response and measures the interval between the first
and last token. It timestamps a chunk only when that chunk carries `content`:

```python
delta = (ch.get("delta") or {}).get("content")
if delta:
    if first_at is None: first_at = now
    last_at = now
...
ct = usage["completion_tokens"]              # ALL tokens, reasoning included
decode_tps = (ct - 1) / (last_at - first_at) # window of content only
```

This model emits its chain of thought in **`reasoning_content`**, a different field of the
delta. Those chunks are never timestamped, so the entire reasoning phase falls inside what
the harness calls "time to first token" — while its tokens remain in the numerator. The
numerator counts reasoning; the denominator excludes the time that produced it.

**The inflation scales with how much the model reasons**, which is why it was invisible to
us: our prose prompt reasons briefly and our code prompt reasons at length, and we read the
resulting gap as a code-versus-prose effect. It was not. It was the same server.

For the row this build is actually served at — `--n-cpu-moe 4`, 98,304 context, code prompt
— we published **126.95 tok/s**. Measured by wall clock: **85.0 tok/s**.

### And so: `predicted_per_second` is fine. Please disregard our warning about it.

The 2026-08-31 note told you that `llama-server`'s `timings.predicted_per_second` reports a
"prefill-contaminated" figure and to not use it. **That is wrong.** Re-measured today, that
counter agreed with wall-clock throughput to within 3% on every configuration we tried. It
was our harness that disagreed with reality, not the server.

The rule we should have followed, and now do: **a throughput number needs two independent
clocks that agree.** We report three — wall clock, post-first-token decode, and the server's
own counter — and treat any disagreement beyond ~10% as a measurement bug rather than a
finding. Had we done that in August, the harness would have failed its first run: it printed
`old method: 94.71` next to `decode: 369.09` and we read that as the old method having been
too conservative.

### What survives and what does not

**Survives.** The `--n-cpu-moe 0` collapse at long context — 10.7× is far too large to be a
metric artifact, and it reproduces in every clock. Do not run `--n-cpu-moe 0` at 98,304 on a
32 GB card. The relative ordering *within* a single prompt and a single speculation setting
is also roughly preserved, since the inflation factor is common to those rows: the expert
cache really does overtake `--n-cpu-moe 0` at long context, and `--cache-reuse` really does
help decode and not only prefill.

**Does not survive.** Every absolute tok/s figure in the 2026-08-31 table — inflated by
roughly 1.5× on the code prompt and less on prose. And any comparison **across** rows that
differ in speculation setting, including the Q3-without-spec against Q2-with-spec table,
because those rows produce different reasoning-to-content ratios and the inflation no longer
cancels.

**Unaffected, again.** No weights changed. The quality claims — 95.1% HumanEval, 9/10 on the
rare-name probe, 25% fabrication — are measured by counting correct answers, not by timing,
and do not move.

We are not republishing a full corrected table here until it has been re-run end to end with
the three-clock protocol. The honest anchor in the meantime: **as served
(`--n-cpu-moe 4`, `--spec-type ngram-map-k4v`, 98,304 context, KV `q4_0`), this build
generates 85 tok/s wall-clock on a code prompt**, and the MTP configuration described above
raises that to roughly 92-97 depending on offload.

*Two measurement corrections in two days, the second one correcting the first. The lesson we
are taking from it: a fix to a measurement does not validate itself. It has to be reconciled
against the method it replaces, and a large unexplained improvement in your own numbers is a
symptom before it is a result.*
