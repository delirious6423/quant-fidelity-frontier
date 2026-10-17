---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - ggml-org/Qwen3.8-27B-GGUF
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: text-generation
tags:
  - gguf
  - rocm
  - rocmfpx
  - rocmfp4
  - amd
  - strix-halo
  - gfx1151
  - mtp
  - speculative-decoding
  - qwen3.5
---

# Qwen3.8-27B — ROCmFP4 + MTP drafter ladder (Strix Halo)

ROCmFP4 builds of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantised from
`ggml-org/Qwen3.8-27B-GGUF`'s BF16 (sha256 `5a3eedc837bcbd13…`, verified), **plus an MTP drafter
at five precisions** so the speculative-decoding numbers below can be reproduced rather than
taken on trust.

What is here and largely not elsewhere: **draft-acceptance rates**, a **per-backend `n-max`
sweep**, a **drafter-precision ladder**, **measured perplexity for all three presets against a
`Q4_K_M` reference**, and a **ROCm-version comparison that reverses the preset ranking**.

> 🚨 **Do not use `-ctk q8_0 -ctv turbo4`.** That specific pairing **silently corrupts long-context
> output** on this model — short prompts look fine while retrieval past ~10K tokens fails. Use
> `-ctk q8_0 -ctv q8_0` (same memory saving, verified correct) or plain f16. Details in §6.

> ⚠ **Choose FP4 for footprint, not for throughput.** On the same machine with MTP enabled on both
> sides, mainline llama.cpp on Vulkan with a plain `Q4_K_M` **ties on decode** (38.94 vs 38.67 t/s)
> and **wins prefill** (330 vs 227). What FP4 buys is **1.7 GiB less resident memory**
> (15.6 vs 17.3 GiB), which is what matters when co-residing two models on one 128 GB APU.

## Files

| file | preset | size |
| --- | --- | ---: |
| `Qwen3.8-27B-ROCMFP4-STRIX.gguf` | `Q4_0_ROCMFP4_STRIX` — **best FP4 perplexity** | 13.75 GiB |
| `Qwen3.8-27B-ROCMFP4-FAST.gguf` | `Q4_0_ROCMFP4_FAST` — **smallest**, +0.025 PPL | 13.33 GiB |
| `Qwen3.8-27B-ROCMFP4-COHERENT.gguf` | `Q4_0_ROCMFP4_COHERENT` — **dominated, see §4** | 14.41 GiB |
| `mtp-Qwen3.8-27B-ROCMFP4-STRIX.gguf` | FP4 drafter | 1.85 GiB |
| `mtp-Qwen3.8-27B-ROCMFP4-FAST.gguf` | FP4 drafter (FAST preset) | 1.50 GiB |
| `mtp-Qwen3.8-27B-ROCMFP3.gguf` | FP3 drafter | 1.55 GiB |
| `mtp-Qwen3.8-27B-ROCMFP6.gguf` | FP6 drafter | 2.27 GiB |
| `mtp-Qwen3.8-27B-ROCMFP8.gguf` | FP8 drafter | 2.86 GiB |
| `mtp-Qwen3.8-27B-ROCMFP2.gguf` | FP2 drafter — **broken, see §3** | 1.48 GiB |

Requires a [ROCmFPX](https://github.com/charlie12345/ROCmFPX) build; mainline llama.cpp does not
know the `Q4_0_ROCMFP4_*` tensor types.

## Hardware / method

AMD Ryzen AI MAX+ 395, Radeon 8060S (`gfx1151`, RDNA 3.5), 124 GiB GTT, Debian sid, kernel 7.2.
Server-measured (`llama-server` + probe), temperature 0, one job at a time. Tables in §1–§3 use a
~8K-token prompt; §4 uses `llama-perplexity`; §5 uses `llama-bench`. ROCm as stated per table.

> **Read §0 before quoting any decode number from this card.**

## 0. Decode speed is acceptance-dominated, so it is task-dependent

MTP throughput tracks draft acceptance almost linearly, and acceptance depends on how predictable
the output is. Same files, same flags, same machine:

| workload | draft acceptance | decode |
| --- | ---: | ---: |
| summarize an 8K document | **0.64–0.77** | **~28 t/s** |
| short open-ended prompt ("explain lifetime elision") | **0.60–0.66** | **27–28 t/s** |

A long, predictable prompt lets the draft head land nearly every token; an open-ended one does
not. **Quote the workload alongside the number** — a bare "t/s" for this model is not meaningful,
here or in anyone else's benchmark. Everything below is the high-acceptance (~8K prompt) regime.

## 1. Draft depth (`--spec-draft-n-max`) is per-backend

| n-max | Vulkan Q4_K_M decode | acc | FP4 decode | acc |
| ---: | ---: | ---: | ---: | ---: |
| 3 | — | — | 33.42 | 100.0% |
| 4 | 35.80 | 88.1% | 35.16 | 98.7% |
| **5** | **38.94** | 91.6% | 38.67 | 98.1% |
| 6 | 38.47 | 86.5% | 38.04 | 97.5% |
| 7 | 37.84 | 82.0% | **39.26** | 94.7% |
| 8 | 28.56 | 78.1% | 32.36 | 95.2% |
| 10 | 25.47 | 59.4% | — | — |

Acceptance decays monotonically with depth; the knee is where verifying rejected drafts costs
more than the accepted ones save. FP4 holds higher acceptance, so its knee sits deeper and
flatter. **This is not DeepSeek-V4's n=2** — draft depth does not transfer between models.

The curve is broad on top: an independent re-sweep on the running server put n=4 at 39.3 and n=6
at 38.5 t/s — a 2% spread, inside run-to-run noise — while n=3 fell to 35.5. Anywhere in 4–7 is
fine; the failure mode is leaving it at llama.cpp's **default of 16**, which roughly halves
throughput.

## 2. Drafter precision is a bandwidth lever, not a quality one

Target fixed, drafter varied, Vulkan, n=5:

| drafter | size | decode | acceptance |
| --- | ---: | ---: | ---: |
| **Q4_K_M** | 1.89 GiB | **39.16** | 91.6% |
| Q6_K | 2.28 GiB | 38.28 | 92.1% |
| Q5_K_M | 2.08 GiB | 36.93 | 89.0% |
| Q8_0 | 2.95 GiB | 34.74 | 89.0% |

**Acceptance is flat (89–92%) while decode spans 13%** — so shrinking the drafter buys bandwidth
and costs nothing in draft quality. Advice to keep drafters at ≥Q8 does not hold here. The same
ordering holds on the FPX ladder in §3, and a BF16 drafter is *slower still* despite the best
acceptance of any variant tested.

**Keep the drafter as a separate file.** A single-file build with the MTP head grafted into the
model (append `blk.N.nextn.*`, `block_count+1`, `nextn_predict_layers=1`) measured **23.0 t/s**
against **25.7** for the same model with the drafter kept as a sidecar.

## 3. FP2 destroys a drafter

FPX ladder, STRIX target, ROCm 10.1, n=5:

| drafter | decode | acceptance |
| --- | ---: | ---: |
| FP4-STRIX | **37.03** | 97.4% |
| FP3 | 36.07 | **98.1%** |
| FP6 | 30.92 | 96.6% |
| FP8 | 29.76 | 96.6% |
| **FP2** | 22.07 | **64.0%** |

FP2's codebook has no exact zero. This drafter is BF16-sourced — the case usually assumed safe —
and acceptance still collapses. **Do not use FP2 for a draft model.**

## 4. Perplexity: STRIX is the best FP4 preset, COHERENT is dominated

Held-out wikitext-2, **145 chunks at ctx 2048**, same chunks for every model, measured with
`llama-perplexity` on this machine. Lower is better.

| build | size | PPL | vs STRIX |
| --- | ---: | ---: | ---: |
| mainline `Q4_K_M` (reference) | 15.41 GiB | **6.3383** ± 0.0402 | −0.033 |
| **`ROCMFP4-STRIX`** | 13.75 GiB | **6.3715** ± 0.0402 | — |
| `ROCMFP4-FAST` | 13.33 GiB | 6.3968 ± 0.0404 | +0.025 |
| `ROCMFP4-COHERENT` | 14.41 GiB | 6.5002 ± 0.0417 | +0.129 |

- **COHERENT is dominated by STRIX**: 0.66 GiB *larger* and clearly worse (+0.129, three times the
  error bar). Its one advantage is prefill on ROCm 7.2 (§5) — a backend- and version-conditional
  win that costs quality. Do not pick it for quality.
- **STRIX vs FAST is +0.025**, smaller than either error bar — but the two are measured on
  identical chunks and STRIX is lower at **every** cumulative checkpoint from chunk 1 to 145, so
  the ordering is systematic rather than noise. The magnitude is small: FAST costs ~0.4% perplexity
  and saves 0.42 GiB. Either is defensible; **STRIX if you want the best FP4 quality, FAST if you
  want the smallest file.**
- **Q4_K_M still has the lowest perplexity of all four**, at 1.5–2.1 GiB more. FP4 is not free —
  it trades ~0.5% perplexity for ~13% less memory.

The error bars above (±0.04) are roughly half those of a 40-chunk run (±0.068), which is why the
full test set is used here.

## 5. The preset ranking flips with the ROCm version — but check decode too

`llama-bench`, pp2048:

| preset | ROCm 7.2.4 | ROCm 10.1 nightly |
| --- | ---: | ---: |
| COHERENT | **205.7** | 208.6 (+1%) |
| STRIX | 151.8 | **272.0 (+79%)** |

COHERENT wins on 7.2; STRIX wins on 10.1. Any "preset X is best" claim — including ones in other
repos — is conditional on a ROCm version that usually goes unstated.

**That table is prefill only, and decode moves the other way.** Against stable 7.2.4, the 10.1
nightly measures **+22% prefill but −9% decode**. Interactive serving is normally decode-bound, so
our production stays on **7.2.4**; take the nightly only if you are prefill-bound.

`-ub` is a further caveat: `llama-bench` shows a clear preference for `-ub 256` (370.6 vs 332.0
pp2048 on Vulkan), but in ablation on the **running server** `-b`/`-ub` sizing showed no
measurable effect. Treat it as harness-specific until reconciled.

## 6. `-ctk q8_0 -ctv turbo4` corrupts long-context output

Needle-in-a-haystack at ~14.6K tokens, `Q4_0_ROCMFP4_FAST`, no draft model, everything else
identical — only the KV cache types vary:

| `-ctk` / `-ctv` | needle @14.6K |
| --- | --- |
| f16 / f16 (default) | ✅ PASS |
| **q8_0 / turbo4** | ❌ **FAIL** |
| q8_0 / q8_0 | ✅ PASS |
| f16 / turbo4 | ✅ PASS |

**Only the pairing fails.** `turbo4` alone is fine and `q8_0` alone is fine; combined, the model
stops retrieving from long context — it rambles or answers confidently wrong, while short prompts
stay perfect. Perplexity and 8K summarization do not catch it.

It also **inflates throughput**, which is how it hides: the corrupted output is repetitive, so the
draft head accepts nearly everything (acceptance 0.92–0.98 vs a normal 0.64–0.77) and decode reads
~40% high. A speed win plus a high acceptance rate is exactly the signature to distrust.

Both affected configurations log `attention rotation disabled for fp3 ROCmFPX/TurboQuant KV cache`
at startup, but so does the passing `f16 + turbo4` case, so the warning alone is not diagnostic.

## Usage

```bash
llama-server \
  -m Qwen3.8-27B-ROCMFP4-STRIX.gguf \
  -md mtp-Qwen3.8-27B-ROCMFP4-STRIX.gguf \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  -ngl 99 -ngld 99 -fa on \
  -ctk q8_0 -ctv q8_0
```

`-ctk q8_0 -ctv q8_0` saves memory against f16 at the same speed. **Do not substitute `turbo4` for
the V cache** — see §6. Our own serving runs the `FAST` preset (4.25 bpw, 13.33 GiB) with the matching
FAST-preset drafter on ROCmFPX **Vulkan** — Vulkan beats HIP on decode (25.6 vs 22.1), HIP wins
prefill (146 vs 140). Both FAST files are in this repo; swap `STRIX` for `FAST` in the command
above to reproduce it, or keep `STRIX` for the slightly better perplexity (§4).

## Related

[`kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF`](https://huggingface.co/kingjones777/Qwen3.8-27B-ROCmFP4-STRIX-MTP-GGUF)
measures the same model on the same gfx1151 / ROCm 7.2.4 and reports **30.30 t/s @8K at acceptance
0.926**, consistent with the §0 high-acceptance regime.

It also publishes perplexity — **STRIX 5.8877 vs FAST 5.9233**, 40 chunks — and **§4 here
independently reproduces that ordering**. The two runs line up closely: their 40-chunk figures sit
within ~0.02 of our own running estimate at chunk 40 (FAST 5.9379, STRIX 5.9100), which is a
useful cross-check given they quantised independently. Our §4 numbers are higher in absolute terms
only because ours runs the full 145-chunk test set, and the later chunks are harder — this is why
perplexity is comparable *within* a run and not *across* runs with different chunk counts.

Their point that the FP4 presets are speed-equivalent, so perplexity rather than speed should be
the tiebreak, holds up.

## Provenance

Every file derives from `ggml-org/Qwen3.8-27B-GGUF`, sha256-verified before quantisation:
`Qwen3.8-27B-BF16.gguf` = `5a3eedc837bcbd13…`, `mtp-Qwen3.8-27B-BF16.gguf` = `5723e551c4ee2b8c…`.
