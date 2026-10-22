---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: image-text-to-text
quantized_by: Vardaan97
tags:
- gguf
- imatrix
- qwen4exp
- moe
- splash
- apple-silicon
- dflash2
- speculative-decoding
- mixed-precision
---

# Qwen3.8-Flash-Next · Splash Quality GGUF · v4

GGUF quantizations of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for [Splash](https://github.com/incoai/splash) 1.4.0 on Apple silicon.

This repository contains the **Quality** expert body (Quality-PLE8, Quality-PLE5, Quality-PLE4). Attention, Gated-DeltaNet, shared experts, hyper-connections, embedding/output matrices and PLE projections stay Q8_0; routed experts use the exact mix below. The streamed PLE table is separate from the resident body, so stored size and resident memory are different quantities. The [family repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF) lists every variant of this release.

## Quick start

Splash 1.4.0 or newer, a 128 GB Apple-silicon Mac, macOS 26.4 or later:

```bash
brew install incoai/tap/splash
splash serve --model Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF:Quality-PLE8
```

For the other folders, use `:Quality-PLE5` or `:Quality-PLE4`. Splash resolves `:FOLDER` to the eight split parts in that folder. The first start downloads them (152.8 GB for Quality-PLE8) and the vision projector (0.91 GB) from this repository. Splash then adds the DFlash2 draft it releases for Qwen3.8-Flash-Next, [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (1.21 GB), by itself. It picks the draft from the model family it reads in the GGUF, not from the repository name, so you don't pass `--draft-model`; this repository ships no draft. Later starts reuse the downloads.

- **Vision is on by default**, through the BF16 projector at the repository root. Add `--language-only` to serve text only.
- **Memory and context are planned automatically** from the GPU working set. `--ple-memory` defaults to `auto`, which streams the PLE table from the SSD when memory can't hold it. `--max-memory` caps Splash to leave room for other applications.
- **Wired limit.** All our measurements used `iogpu.wired_limit_mb=110592` on an M5 Max 128 GB (`sudo sysctl iogpu.wired_limit_mb=110592`, resets on reboot). A Mac at the default limit is untested.

## At a glance

| | |
|---|---|
| **What** | Custom mixed-precision GGUF of Qwen3.8-Flash-Next: the **Quality** expert body (routed experts Q5_1, Q5_K and Q6_K with a few Q8_0 down projections; everything else Q8_0) with a choice of PLE table: PLE8 (Q8_0), PLE5 (Q5_1) or PLE4 (IQ4_NL). Includes the BF16 vision projector. |
| **For** | 128 GB Apple-silicon Macs, when you want the lowest measured weight error in the family (tied with Max). |
| **Download** | PLE8 153.7 GB · PLE5 137.7 GB · PLE4 128.1 GB (target + projector, decimal GB), plus the 1.21 GB DFlash2 draft from its own repository |
| **Memory actually needed** | About 98.4–101.7 GB of unified memory: the resident body is 98.4 GB, and Splash's memory plan at load was 101.7 GB plus runtime buffers. The PLE table (54.4 / 38.4 / 28.8 GB for PLE8 / PLE5 / PLE4) **stays on the SSD** and streams, so the table choice changes disk use, not memory. |
| **Peak memory, M5 Max 128 GB** | PLE8: 100.8 GiB Splash Metal peak over the agent sweep, 97.5 GiB after load (text-only, 176K context). PLE5/PLE4: release E2E table below. |
| **Decode** | PLE8: 82.3 tok/s (agent sweep, high reasoning, native MTP, output-weighted). PLE5/PLE4: release E2E table below. |
| **Runtime** | Splash 1.4.0 or newer, with the DFlash2 draft. Memory, peak and decode above were measured on the earlier Splash build with native MTP (2026-10-19); the 1.4.0 install check passed (see Benchmarks). |

**Hub GGUF panel.** The panel groups split files by the quant tag in their filenames and shows the total download size, including the PLE table, which streams from the SSD and is not held in memory. All three table variants in this repository carry the same `Q5_K_XL` tag, so the panel shows them as one combined entry. The [family repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF) lists one entry per variant. The memory actually needed is the resident body in the [memory table](#memory-and-storage).

## Which one should I pick

Measured on the earlier Splash build with native MTP (2026-10-19), not on Splash 1.4.0. The KLD figures don't depend on the runtime.

All nine variants are the same model. They trade disk, memory, speed and weight error. **On agent tasks the four tested PLE8 variants are indistinguishable:** 7–8/16 strict and 8–9/16 with the lenient FinQA re-grade, a one-task spread that is within noise for 16 single-trial tasks. PLE5 and PLE4 were not task-tested.

**Our default:** UD-Q4_K_XL-PLE8. It had the fastest decode, the lowest memory and the quickest load, and its agent-task scores are within noise of the others. We keep Quality-PLE8 as the quality reference, since its measured weight error is lowest.

| If you want | Pick | Measured basis (M5 Max 128 GB) |
|---|---|---|
| Fastest decode, smallest weights | [UD-Q4_K_XL-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF/tree/main/UD-Q4_K_XL-PLE8) | 96.1 tok/s agent-sweep decode vs 82.3–86.7 for the custom PLE8 bodies; 82.4 GiB after load vs 95.4–98.2; 82.5 GB resident body vs 96.5–99.1. KLD 0.021037 measured (original UD-Q4_K_XL with its IQ4_NL table: 0.022195). |
| Lowest measured weight error | [Quality-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE8) or [Max-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Max-GGUF/tree/main/Max-PLE8) | Neutral KLD 0.012613 and 0.012537, a statistical tie (0.6 paired SE). Both are more than 5 paired SE below Balanced-PLE8. Max's resident body is 0.75 GB larger. |
| A custom body with a little more memory room | [Balanced-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF/tree/main/Balanced-PLE8) | 96.5 GB resident body vs 98.4 (Quality) and 99.1 (Max); KLD 0.013270. |
| Less disk, same body | PLE5 (Quality, Balanced, Max) | 38.4 GB table instead of 54.4 GB (−16.0 GB). Balanced-PLE5 measured +3.44% KLD vs Balanced-PLE8; no task run. |
| Least disk | PLE4 (Quality, Balanced) | 28.8 GB table (−25.6 GB vs PLE8). Measured KLD +13.6% (Quality) and +12.6% (Balanced) vs PLE8. |

The PLE table streams from the SSD (a 2 GiB cache in these runs), so the table choice changes disk size, not the resident body; PLE5/PLE4 runtime memory is in the release E2E table. Sweep decode rates come from each model's own agent workload, so they are descriptive rather than a controlled A/B. On an older build, UD was also faster on prose (122.8 vs 114.4–115.3 tok/s) but not on the 2K greedy city test (157.7 vs 158.3–161.9).

These custom variants were designed around 128 GB unified memory and PLE streaming. A full 262,144-token base context is a model capability, not an end-to-end validation for each quant.

## Call it

Splash serves an OpenAI-compatible API on `127.0.0.1:8000`. This is the release E2E's tool-call check, which expects two parsed `get_weather` calls:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF:Quality-PLE8",
  "messages": [{"role": "user", "content": "Call get_weather exactly twice: once for Paris and once for Tokyo, both in celsius. Do not answer in prose."}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Read weather for one city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius"]}},
                   "required": ["city", "unit"], "additionalProperties": false}}}],
  "tool_choice": "auto", "parallel_tool_calls": true,
  "reasoning_effort": "high", "max_tokens": 4096
}'
```

**Optional settings.** None of these is required on 1.4.0. They reproduce the conditions of our earlier measurements.

- `--max-context 176K` (180,224 tokens) is the server limit we tested; the agent harness kept each request within 131,072 tokens. Without it, Splash plans the context itself. The native 262,144-token window is not validated for these files.
- `--ple-memory 2G` gives the 2 GiB PLE row cache our runs used; the default `auto` sizes the cache itself.
- `--disable-ane`, `--idle-release off` and `--max-cache-disk 0` (SSD cache off) match the benchmarks. With the default `--idle-release 10m`, the weights are freed after 10 idle minutes and reloaded by the next request (loading took 41.4 s in the sweep).
- `int8` KV and `fp32` state are Splash's defaults and what the measurements used.
- `--served-model-name flash-next` adds a short model name for API calls.
- **Reasoning:** the benchmark requested `"reasoning_effort": "high"` through a custom chat template with a distinct high level. The GGUFs' own template supports `low`, `medium` and `xhigh`, so Splash renders `high` as `xhigh`. `"none"` turns reasoning off; `--default-reasoning-effort` sets a server default. Sweep sampling was temperature 1.0, top_p 0.95, top_k 20, min_p 0.

**Offline.** After one online start has installed the model, add `--offline` (or set `HF_HUB_OFFLINE=1`) to start it without contacting the Hub. A 40-hex `--revision` keeps Splash on one repository commit. To pre-fetch one folder only:

```bash
hf download Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF --include "Quality-PLE8/*.gguf" "mmproj-*"
```

## Memory and storage

Measured on the earlier Splash build with native MTP (2026-10-19), not on Splash 1.4.0.

| Variant | Target GB | Download GB | PLE table, GB | Resident body GB | Peak after load, GiB | Sweep peak, GiB | Weights loaded |
|---|---:|---:|---|---:|---:|---:|---:|
| [Quality-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE8) | 152.808 | 153.715 | Q8_0, 54.400 | 98.396 | 97.488 | 100.799 | 41.4 s |
| [Quality-PLE5](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE5) | 136.808 | 137.715 | Q5_1, 38.400 | 98.396 | E2E table | not run | E2E table |
| [Quality-PLE4](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE4) | 127.207 | 128.115 | IQ4_NL, 28.800 | 98.396 | E2E table | not run | E2E table |

Text-only (no projector), 176K context, int8 KV, fp32 state, 2 GiB PLE cache and a 110,592 MB wired limit. *Download* adds the 0.908 GB projector to the target shards; the DFlash2 draft (1.21 GB) comes from its own repository, and `imatrix_unsloth.gguf_file` (0.580 GB) is provenance only, which Splash does not download. *Resident body* is the target tensor bytes without the streamed table, from the GGUF headers. *Peak after load* is Splash's Metal allocation high-water mark after loading plus a 64-token warmup; *sweep peak* is the highest over the roughly one-hour agent run. Neither is process RSS. Whole-machine memory in use peaked at 124.2–124.6 GiB during the four PLE8 sweep runs, including the sandboxed harness, and swap never grew above its starting baseline (about 1.0 GiB). *Weights loaded* is from the server log; the OS file cache was not controlled. Rows without a sweep run are covered by the release E2E table, which also loads the projector.

## Benchmarks (M5 Max 128 GB)

### Splash 1.4.0 install check

<!-- E2E-140 -->
**Passed, 2026-10-22.** This ran on stock Splash 1.4.0 (`1482f6bb`) on an M5 Max 128 GB with the wired limit at 110,592 MB.

- **How it ran:** exactly as a user runs it, `splash serve --model REPO:FOLDER`, with Splash's defaults (vision on, automatic memory and context) plus `--max-cache-disk 0`.
- **What it downloaded:** the Hugging Face cache already held these exact files, matched by SHA-256, so the large shards weren't downloaded again. On the first start, Splash fetched the 1.21 GB DFlash2 draft from its own repository by itself, and nothing else.
- **Context:** its automatic plan held the model's full 262,144-token window.

| Model id | Process start to `Ready` | Served context | Vision | Splash Metal peak | Checks passed |
|---|---:|---:|---|---:|---|
| `Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:Quality-PLE8` | 118 s | 256K | on | 96.0 GiB | 4/4 |

The checks:

- 17 × 23 answered 391.
- A code planted in an ~8K-token prompt was retrieved.
- Two `get_weather` calls (Paris and Tokyo, celsius) came back as parsed tool calls.
- A synthetic receipt image was read into a JSON schema with all 6 fields correct.

The time to `Ready` includes loading the weights; the first start also includes the draft download. Peak memory is Splash's Metal allocation high-water mark after the checks, not process RSS.

The family repository's `Quality-PLE8` folder holds the same bytes as this repository's (identical SHA-256). `Quality-PLE5` and `Quality-PLE4` use the same body with smaller tables and were not launched.

### Scored agent sweep, 2026-10-18/19

Measured on the earlier Splash build with native MTP (2026-10-19), not on Splash 1.4.0.

**Method.** Nine local Flash-Next builds ran one at a time through the same sandboxed agent harness on the M5 Max. Each faced 16 deterministically graded tasks from public benchmarks (one trial each), four repeat attempts (F1 and H3 twice more) and one separately judged finance write-up (F3). All used the same Splash build with native MTP, text-only serving, reasoning high, temperature 1.0 / top_p 0.95 / top_k 20, identical task order and seeds, a fresh server per model, a 131,072-token working context, per-task time caps set from one UD pilot, and a two-hour window per model. The four rows below are the released PLE8 files: the same shard bytes, except that the custom bodies' tensor-free part 1 differs only by a shortened importance-matrix path. No score is transferred to PLE5 or PLE4.

| Variant | Primary /16, strict | Primary /16, lenient F1 | Repeats /4, strict → lenient | Wall s | Decode tok/s, output-weighted | Splash Metal peak GiB | MTP accepted/drafted |
|---|---:|---:|---|---:|---:|---:|---|
| Quality-PLE8 | 8 | 9 | 1 → 2 | 3883.962 | 82.27 | 100.799 | 126796/529746 (23.94%) |
| Balanced-PLE8 | 8 | 8 | 1 → 2 | 4013.255 | 86.70 | 100.739 | 124461/519211 (23.97%) |
| Max-PLE8 | 7 | 8 | 2 → 2 | 3757.541 | 85.50 | 101.625 | 135636/551397 (24.60%) |
| UD-Q4_K_XL + Q8 table | 7 | 8 | 2 → 2 | 3598.469 | 96.09 | 101.254 | 145806/547295 (26.64%) |

Per-task primary outcomes: ✓ pass · ✗ wrong or invalid · ⏱ hit the time cap · ✓ˡ fails the strict final-line rule, passes the lenient re-grade.

| Variant | M1 | M2 | C1 | C2 | T1 | F1 | F2 | H1 | H2 | H3 | H4 | I1 | I2 | B1 | B2 | B3 |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Quality-PLE8 | ⏱ | ✓ | ✗ | ✓ | ⏱ | ✓ˡ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Balanced-PLE8 | ⏱ | ✓ | ✗ | ✗ | ⏱ | ✓ | ✗ | ✓ | ✓ | ⏱ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| Max-PLE8 | ✗ | ✓ | ✗ | ✗ | ⏱ | ✓ˡ | ✗ | ✓ | ✓ | ⏱ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |
| UD-Q4_K_XL + Q8 table | ⏱ | ✓ | ✗ | ✗ | ⏱ | ✓ˡ | ✗ | ✓ | ✓ | ⏱ | ✓ | ✓ | ✓ | ✗ | ✗ | ✓ |

Tasks: M1 AIME 2026 and M2 HMMT Feb 2026 (math, no tools); C1/C2 BigCodeBench-Hard and T1 Terminal-Bench (coding); F1 FinQA and F2 FinanceMath (finance); H1/H2 LongHealth at about 96K/64K input tokens, H3/H4 MedXpertQA (health); I1/I2 IFBench (instructions); B1–B3 BFCL v4 (tools).

**Noise.** n = 16 single-trial tasks. Eleven tasks had the same outcome for all nine sweep models: H1, H2, H4, I2 and B3 always passed; M1, H3, F2, B1, B2 and T1 always failed. Among these four variants, strict scores differ only on C2 (Quality passed) and on F1's answer format (Balanced passed strict); with the lenient F1 re-grade only C2 differs. A one-task difference is within noise, and no paired uncertainty analysis was completed. The whole nine-model sweep spanned 5–9/16 strict. These are results of one harness, not general capability rankings.

**Notes.** The lenient re-grade changes only F1's format rule: it accepts source-reference lines after the numeric `FINAL ANSWER:` line. It is a post-hoc sensitivity check; strict stays the primary score. F3, an adapted FAB financial-variance write-up, is judged separately and is not in /16. All four scored 3/5 and failed the combined gate: the Sonnet 5.5 judge passed all five criteria, but the deterministic signed-figure gate rejected two. UD's F3 attempt also used the whole 131,072-token working context. B1/B2 failed for every model, and the harness showed tool results to the model as render errors in those tasks, so the tools column says little about tool-use ability. T1 and most M1 attempts hit their caps. Wall includes load, warmup, solving, archival and cleanup (Balanced's includes substantial teardown) and excludes cooldowns. Decode is Σ(output × logged tok/s)/Σoutput over completed server requests with ≥500 output tokens (50/52/53/61 requests), excluding warmup and cancelled requests. Metal peak is Splash's allocation high-water mark, not process RSS or whole-machine memory. MTP counters are proposal acceptance, not speedup.

### Neutral KLD and perplexity

Same BF16 reference and neutral text, 87 × 4096-token chunks, 178089 scored tokens; base PPL 4.045181. Chunk SE is used for KLD uncertainty; PPL ratios are the measured tool summary. These are distribution comparisons, not task scores, and they don't depend on the runtime.

| Variant | Mean KLD | Chunk SE | PPL(Q)/PPL(base) | Status |
|---|---:|---:|---:|---|
| Quality-PLE8 | 0.012613 | 0.000436 | 1.000487 | measured |
| Quality-PLE5 | 0.013128 | 0.000135 (scorer SE) | 1.001136 | measured 2026-10-19; top-1 95.850% |
| Quality-PLE4 | 0.014330 | 0.000523 | 1.000863 | measured |
| Balanced-PLE8 | 0.013270 | 0.000433 | 1.001775 | measured |
| Balanced-PLE5 | 0.013727 | 0.000443 | 1.001645 | measured; top-1 95.754% |
| Balanced-PLE4 | 0.014940 | 0.000501 | 1.000898 | measured |
| Max-PLE8 | 0.012537 | 0.000422 | 1.001522 | measured |
| Max-PLE5 | 0.012848 | 0.000109 (scorer SE) | 1.000912 | measured 2026-10-19; top-1 95.863% |
| UD-Q4_K_XL + Q8 table | 0.021037 | 0.000163 (scorer SE) | 1.004707 | measured 2026-10-19; top-1 94.807% |
| Original UD-Q4_K_XL, IQ4_NL table | 0.022195 | 0.000715 | 1.004633 | measured reference; different table |

Balanced-PLE5 adds 0.000457 KLD (+3.44%) versus PLE8 while saving 16.000 GB. Quality-PLE5 (0.013128) and Max-PLE5 (0.012848) were measured on 2026-10-19 with the same scorer. Quality and Max PLE8 are effectively tied (0.6 paired SE). Full method, raw baseline logs and paired comparisons remain in [KLD evidence](kld/KLD-TABLE.md).

### 1M context

Measured on an earlier Splash YaRN build without MTP, not on Splash 1.4.0. Splash 1.4.0 turns on YaRN when `--max-context` goes past the 256K native window, up to 1,010,000 tokens; this test has not been rerun on it.

Original unsloth UD-Q4_K_XL with its stock **IQ4_NL** table, factor 4, max context 1010000, int8 KV, fp32 state, 2 GiB PLE cache: at 1000000 prompt tokens it retrieved **8/8 planted facts** and returned NOT_FOUND for the absent fact, with **614.3 s cold TTFT**, 1628 prefill tok/s, 28.5 decode tok/s and **91.9 GiB Splash peak**, no new swap. Retrieval was also 8/8 at 262K and 520K. The sum-of-amounts test failed at 520K and 1M. This does not validate arbitrary long-context reasoning, other quants, or the custom bodies at 1M. [Reproduction](https://github.com/incoai/splash/pull/469#issuecomment-6256428597).

### Release E2E smoke, 2026-10-19

Measured on the earlier Splash build with native MTP (2026-10-19), not on Splash 1.4.0.

<!-- E2E_RESULTS:BEGIN -->
| Variant | Load s | Peak Metal GiB | Warmup MTP % | Functional /5 | Prose tokens | Decode tok/s | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| Quality-PLE4 | 45.071 | 98.669 | 68.831 | 5/5 | 300 | 66.075 | PASS |
| Quality-PLE5 | 45.071 | 98.670 | 68.831 | 5/5 | 300 | 72.368 | PASS |
| Quality-PLE8 | 45.650 | 98.671 | 61.905 | 5/5 | 300 | 65.964 | PASS |
<!-- E2E_RESULTS:END -->
The E2E decode figure is a single cold 300-token sample with reasoning off, so it reads lower than the agent-sweep decode rates in the glance box. The two numbers measure different things.

Completed 2026-10-19T07:14:16Z on the exact upload-source assembly, the stock MTP head of that build and the shipped projector; int8 KV, fp32 state, 2 GiB PLE cache. Prose timing disables reasoning and EOS for exactly 300 completion tokens; the other checks request high reasoning. Load timing is process launch to ready, with the existing OS file-cache state (not a certified cold load). The six-prompt smoke battery covers text/math, executed code assertion, two tool calls, an 8K-token needle, exact instruction format and a 300-token prose decode. Projector presence/header checks do not establish image/PDF correctness. These checks do not replace the scored sweep or validate streaming, Responses API, full-context correctness or broader hardware.

## Variants

One repository per expert body, with PLE8/PLE5 folders together and PLE4 where offered; the [family repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF) lists them all. All sizes are decimal GB; target downloads exclude the projector and importance matrix. These are **mixed quantizations**, not uniform Q8/Q6/Q5/Q4 files.

| Variant | PLE tensor / GB | Target download GB | Native resident GB | Splash target GB | KV capacity, tokens | Mean KLD (chunk se) | Same top-1 % |
|---|---|---:|---:|---:|---:|---:|---:|
| [Quality-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE8) | Q8_0 / 54.400 | 152.808 | 98.396 | 101.766 | 368,640 | 0.012613 (0.000436) | 95.928 |
| [Quality-PLE5](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE5) | Q5_1 / 38.400 | 136.808 | 98.396 | 101.723‡ | 409,600‡ | 0.013128 (scorer SE 0.000135), measured | 95.850 |
| [Quality-PLE4](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE4) | IQ4_NL / 28.800 | 127.207 | 98.396 | 101.590† | 327,680† | 0.014330 (0.000523) | 95.663 |
| [Balanced-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF/tree/main/Balanced-PLE8) | Q8_0 / 54.400 | 150.946 | 96.535 | 99.564 | 434,176 | 0.013270 (0.000433) | 95.844 |
| [Balanced-PLE5](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF/tree/main/Balanced-PLE5) | Q5_1 / 38.400 | 134.946 | 96.535 | 99.521‡ | 565,248‡ | 0.013727 (0.000443), measured | 95.754 |
| [Balanced-PLE4](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF/tree/main/Balanced-PLE4) | IQ4_NL / 28.800 | 125.346 | 96.535 | 99.470† | 442,368† | 0.014940 (0.000501) | 95.609 |
| [Max-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Max-GGUF/tree/main/Max-PLE8) | Q8_0 / 54.400 | 153.555 | 99.143 | 102.526 | 327,680 | 0.012537 (0.000422) | 95.918 |
| [Max-PLE5](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Max-GGUF/tree/main/Max-PLE5) | Q5_1 / 38.400 | 137.555 | 99.143 | 102.483‡ | 360,448‡ | 0.012848 (scorer SE 0.000109), measured | 95.863 |
| [UD-Q4_K_XL-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF/tree/main/UD-Q4_K_XL-PLE8) | Q8_0 / 54.400 | 136.935 | 82.523 | 85.496‡ | 1,540,096‡ | 0.021037 (scorer SE 0.000163), measured | 94.807 |

**File labels.** Shards are labelled for the Hub: `UD-Q4_K_XL` for the UD body, and `Q5_K_XL` for the Quality, Balanced and Max bodies. `Q5_K_XL` is the nearest standard class for these mixes: expert gate/up Q5_K/Q6_K, expert down Q5_1/Q8_0, attention Q8_0. That works out to about 6.1–6.3 bits per weight over the body, excluding the streamed PLE table (computed from body size and parameter count). The exact per-tensor mix is listed on each model card. The table precision is in the folder name: PLE8 = Q8_0, PLE5 = Q5_1, PLE4 = IQ4_NL.

‡ Measured in the release E2E (2026-10-19, earlier build with native MTP) on the exact public files: Splash's memory plan at load with `--max-context 176K`, vision projector loaded, 2 GiB PLE cache. Other rows come from the earlier v2 protocol.

PLE5 uses the same body as its PLE8 sibling and a **Q5_1 table: 38.4 GB, 6.0 bpw** (32 values per 24-byte block). Balanced-PLE5 KLD and top-1 are measured on the same neutral protocol as its PLE8/PLE4 baselines; its chunk SE is reconstructed from 87 running chunk means. Quality-PLE5 (0.013128, +4.1% vs Quality-PLE8) and Max-PLE5 (0.012848, +2.5% vs Max-PLE8) were measured on 2026-10-19 with the same scorer and reference, on the exact release files. For these two rows the SE shown is the scorer's own; the other rows use the chunk-SE reconstruction. The model treats table error as independent of body error; these estimates are not task results. Native resident tensor bytes exclude the table and are unchanged.

Native resident bytes sum target tensor bytes with the PLE table excluded. Splash target bytes and KV capacity come from runtime-format status on earlier builds; Splash 1.4.0 plans memory itself, so its figures can differ. KV capacity is shared across lanes; it is **not** a tested single-request context length. PLE8 runtime status here is from the SVG stack, with int8 KV, fp32 state, a 2 GiB PLE cache and a 110,592 MB wired limit; the SVG requests used 32,768 context. † Historical status from the earlier release build. These are separate cohorts, not matched memory comparisons. [Evidence](MEASUREMENTS.md#memory-and-files).

<a id="future-variants"></a>
**Ultra256-PLE8:** a historical Q8-focused larger-memory archive is recorded, but no validated public release is offered. **Max-PLE4:** not released. **Lite:** not released (missing body shards). Quality-H6 is an experimental control, outside this release family.

## Exact Quality tensor mix

Layer numbers are zero-based. `blk.N.` prefixes are omitted below; global tensors are named literally. Each listed type applies to every tensor of that suffix on the listed layers. All offered table variants share the same expert body. **Q4_0, Q4_1 and Q4_K are absent from these custom bodies**; PLE4 uses IQ4_NL specifically. Q5_1 down-projections are distinct from Q5_K gate/up tensors. [Header/recipe evidence](MEASUREMENTS.md#memory-and-files).

| Tensor suffix or global name | Exact type → layers |
|---|---|
| `attn_gate.weight` | Q8_0: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `attn_k.weight` | Q8_0: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `attn_k_norm.weight` | F32: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `attn_output.weight` | Q8_0: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `attn_q.weight` | Q8_0: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `attn_q_norm.weight` | F32: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `attn_qkv.weight` | Q8_0: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `attn_v.weight` | Q8_0: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `ffn_down_exps.weight` | Q5_1: 0–1, 3, 5–29, 31–45; Q8_0: 2, 4, 30, 46–47 |
| `ffn_down_shexp.weight` | Q8_0: 0–47 |
| `ffn_gate_exps.weight` | Q5_K: 7–8, 10–13, 15, 17–23, 26–29, 32–34; Q6_K: 0–6, 9, 14, 16, 24–25, 30–31, 35–47 |
| `ffn_gate_inp.weight` | F32: 0–47 |
| `ffn_gate_inp_shexp.weight` | F32: 0–47 |
| `ffn_gate_shexp.weight` | Q8_0: 0–47 |
| `ffn_up_exps.weight` | Q5_K: 7–8, 10–13, 15, 17–23, 26–29, 32–34; Q6_K: 0–6, 9, 14, 16, 24–25, 30–31, 35–47 |
| `ffn_up_shexp.weight` | Q8_0: 0–47 |
| `hc_attn_down.weight` | Q8_0: 0–47 |
| `hc_attn_inject.weight` | F32: 0–47 |
| `hc_attn_norm.weight` | F32: 0–47 |
| `hc_attn_up.weight` | Q8_0: 0–47 |
| `hc_ffn_down.weight` | Q8_0: 0–47 |
| `hc_ffn_inject.weight` | F32: 0–47 |
| `hc_ffn_norm.weight` | F32: 0–47 |
| `hc_ffn_up.weight` | Q8_0: 0–47 |
| `indexer.k_norm.weight` | F32: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `indexer.k_proj.weight` | BF16: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `indexer.q_norm.weight` | F32: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `indexer.q_proj.weight` | BF16: 3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47 |
| `output.weight` | Q8_0: global |
| `output_hc_down.weight` | Q8_0: global |
| `output_hc_norm.weight` | F32: global |
| `output_hc_up.weight` | Q8_0: global |
| `per_layer_token_embd.weight` | Q8_0: global (PLE8); Q5_1: global (PLE5); IQ4_NL: global (PLE4) |
| `ple_conv1d.weight` | F32: 1 |
| `ple_key.weight` | Q8_0: 1 |
| `ple_norm_conv.weight` | F32: 1 |
| `ple_norm_key.weight` | F32: 1 |
| `ple_norm_query.weight` | F32: 1 |
| `ple_value.weight` | Q8_0: 1 |
| `ssm_a` | F32: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `ssm_alpha.weight` | F32: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `ssm_beta.weight` | F32: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `ssm_conv1d.weight` | F32: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `ssm_dt.bias` | F32: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `ssm_norm.weight` | F32: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `ssm_out.weight` | Q8_0: 0–2, 4–6, 8–10, 12–14, 16–18, 20–22, 24–26, 28–30, 32–34, 36–38, 40–42, 44–46 |
| `token_embd.weight` | Q8_0: global |

Tensor bytes by actual storage type (GGUF headers/padding excluded; bytes are derived from the recorded shapes and block sizes):

| Variant | Type | Tensors | Parameters | Tensor bytes |
|---|---|---:|---:|---:|
| Quality-PLE8 | BF16 | 24 | 19,660,800 | 39,321,600 |
| Quality-PLE8 | F32 | 557 | 78,373,760 | 313,495,040 |
| Quality-PLE8 | Q5_1 | 43 | 36,071,014,400 | 27,053,260,800 |
| Quality-PLE8 | Q5_K | 42 | 35,232,153,600 | 24,222,105,600 |
| Quality-PLE8 | Q6_K | 54 | 45,298,483,200 | 37,158,912,000 |
| Quality-PLE8 | Q8_0 | 504 | 60,244,213,760 | 64,009,477,120 |
| Quality-PLE5 | BF16 | 24 | 19,660,800 | 39,321,600 |
| Quality-PLE5 | F32 | 557 | 78,373,760 | 313,495,040 |
| Quality-PLE5 | Q5_1 | 44 | 87,271,260,160 | 65,453,445,120 |
| Quality-PLE5 | Q5_K | 42 | 35,232,153,600 | 24,222,105,600 |
| Quality-PLE5 | Q6_K | 54 | 45,298,483,200 | 37,158,912,000 |
| Quality-PLE5 | Q8_0 | 503 | 9,043,968,000 | 9,609,216,000 |
| Quality-PLE4 | BF16 | 24 | 19,660,800 | 39,321,600 |
| Quality-PLE4 | F32 | 557 | 78,373,760 | 313,495,040 |
| Quality-PLE4 | IQ4_NL | 1 | 51,200,245,760 | 28,800,138,240 |
| Quality-PLE4 | Q5_1 | 43 | 36,071,014,400 | 27,053,260,800 |
| Quality-PLE4 | Q5_K | 42 | 35,232,153,600 | 24,222,105,600 |
| Quality-PLE4 | Q6_K | 54 | 45,298,483,200 | 37,158,912,000 |
| Quality-PLE4 | Q8_0 | 503 | 9,043,968,000 | 9,609,216,000 |

PLE streaming avoids keeping the full table resident but still uses a row cache.

## Speed on earlier builds

Measured on older Splash builds with native MTP, before the 2026-10-19 sweep build, not on Splash 1.4.0. All figures are single-stream observations on one 128 GB M5 Max (40-core GPU); they are workload- and build-specific. The older STACK and the #447-based build are separate measurements.

| Earlier Splash STACK | City greedy tok/s | Prose-category tok/s | Cold TTFT | Warm / deep-edit TTFT |
|---|---:|---:|---|---|
| Quality-PLE8 | 161.9 | 114.4 | 32K: 17.85 s | 32K: 0.144 / 9.75 s |
| Balanced-PLE8 | 158.3 | 115.3 | 64K: 44.1 s | 64K: 0.16 / 11.8 s |
| Max-PLE8 | not measured in this matched protocol | not measured | not measured | not measured |
| UD-Q4_K_XL | 157.7 | 122.8 | 32K: 17.68 s | 32K: 0.123 / 9.53 s |

City is the 2,048-token greedy workload; prose here is the **prose category** from the mixed corpus, not its all-category aggregate. Exact-token cold/warm/edit protocol, thermal gating and depth differ by row; no cross-row speedup is claimed. Quality prefill was 2,466 tok/s at 4K and 1,836 at 32K. Max's older SVG median decode was 98.6 tok/s; that is not a city/prose benchmark. On Balanced-PLE8, the final older STACK measured aggregate 72-prompt throughput 105.3 → 115.1 tok/s (+9.4%) vs the stock head, with 72/72 identical greedy outputs; the post-hoc analysis included PLE-related disk-tainted launches. [Earlier-stack evidence](MEASUREMENTS.md#earlier-stack).

**#447-based build. These tests used original UD-Q4_K_XL with its IQ4_NL table, not the custom variants or the UD+Q8 assembly in this family.**

| Runtime/configuration | City 2K tok/s | 72-prompt aggregate tok/s | Cold TTFT 4K / 32K / 64K, s | Warm TTFT, s | Edit at ¾ of 64K, s |
|---|---:|---:|---|---|---:|
| #448 on #447, defaults + GDN grid 16K | 155.0 | 114.8 / 114.5 | 1.70 / 14.46 / 31.02 | 0.09–0.10 | 8.32 |
| #448, PLE prefetch off | — | 113.0 | — | — | — |
| #447 + host-step fix, placeholder draft proposing nothing | — | 33.1 | — | — | — |

The last row is the tested **no-drafting proxy**, not a trained external draft comparison and not a matched MTP-off run. Its prose launch was unbracketed. The native-MTP aggregate averages 114.7 tok/s versus 33.1; treat this as descriptive. City was cool-down gated; prose/TTFT ran warm, one launch per arm, bracketed by the older fork reference except the no-draft proxy.

| UD-Q4_K_XL, long context | Cold TTFT 128K / 170K, s | Decode at 128K / 170K, tok/s | Peak allocation |
|---|---|---|---|
| #448 on #447 | 68.0 / 95.4 | 100.2 / 93.4 | 89.1 GiB |
| Earlier fork reference, bracketing launches | 98.8 / 139.4; 100.5 / 140.7 | 101.1 / 91.9; 98.5 / 93.4 | 89.3 GiB |

Long-context tests used 131,072 and 174,080 prompt tokens, then 256 greedy output tokens at depth, `--max-context 176K`, int8 KV, fp32 state and a 2 GiB streamed table cache; one launch per arm, no cooling gate. These establish UD runtime performance, not custom-variant long-context quality or memory fit. The greedy outputs matched the no-draft proxy on 30/72 prompts; matching verify-row composition gave 72/72. The batched kernels make rows depend on their verify batch, so cross-mode bitwise identity is not promised. [Runtime evidence](MEASUREMENTS.md#new-runtime).

## How quality was measured

| Model | Neutral mean KLD | Blind SVG mean / 5 | Valid SVG | Coding / 320 |
|---|---:|---:|---:|---:|
| Quality-PLE8 | 0.012613 | 3.69 earlier; 3.67 re-score | 9/9 | 278.35 |
| Balanced-PLE8 | 0.013270 | 3.49 | 8/9 | not measured |
| Max-PLE8 | 0.012537 | 3.62 | 9/9 | not measured |
| Quality-PLE4 | 0.014330 | not measured | — | not measured |
| Balanced-PLE4 | 0.014940 | not measured | — | not measured |
| UD-Q4_K_XL (original IQ4_NL table) | 0.022195 | 3.60 | 8/9 | 278.68 |

- **KLD:** BF16 neutral logits from base commit `de4b8e4d`, held-out prose in 30 languages; 87 chunks at 4,096 context, 178,089 scored tokens. One local Metal scoring harness, microbatch 512, versus reference CUDA batch 2,048. The PLE table stayed in CPU memory. These scores measure weight error outside Splash; the absolute harness offset is unmeasured. Use chunk and paired standard errors, not the tool's independent-token error. [Method and reference datasets](kld/METHOD.md), [ranked table](kld/KLD-TABLE.md), [paired table](kld/PAIRED-RELEASE.md).
- **Publication gates:** Quality-PLE8 has its own completed KLD run. Against Balanced-PLE8 it is −0.000660 ± 0.000125 paired se (5.3 se lower). Max-PLE8 is −0.000730 ± 0.000101 (7.2 se lower), meeting the existing requirement of at least two paired se. Quality vs Max is only +0.000070 ± 0.000117 (0.6 se): no established quality separation. Paired values recover rounded running means and need not equal subtraction of the final six-decimal summaries. Quality-PLE8 vs UD is −43.2% KLD on this set; no general task-quality percentage is implied.
- **SVG:** three prompts × seeds 17/42/123, thinking on, medium effort, temperature 1, top_p 0.95, top_k 20, 24,000-token output cap; one blind Sonnet 5.5 rater, five criteria. Invalid/capped outputs score 1. The Quality re-score uses byte-identical images; it is not an independent replication. Earlier models and the later Quality anchor ran in different thermal/runtime cohorts. The differences among Quality, Balanced, Max and UD do not establish a reliable task ranking at this sample size.
- **Coding:** a 13-task, 320-point coding-agent harness, thinking off, one run per model on the earlier STACK. Quality and UD are descriptively tied. Permission rejections ended sessions early across the harness; these scores include that confound. Balanced/Max have no run of this coding suite; agent-task results are in Benchmarks.
- **End-to-end behaviour:** the trace review found no wavering or truncation in Quality's nine SVG cells; Max had four wavering traces and Balanced had two plus one truncation. These are descriptive judgments by a non-blind trace reviewer, not an additional benchmark win.

[Task evidence and protocol notes](MEASUREMENTS.md#task-quality). [UD table and runtime cohorts](UD-EVIDENCE.md).

## Reproducibility

Source: [unsloth's BF16 conversion](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/BF16), quantized with its importance matrix and a per-tensor type file. The target is an eight-part split; shard 3 contains only `per_layer_token_embd.weight`. Quality-PLE8 is assembled from the measured Quality body and the measured Balanced-PLE8 table; its recipe/type file describes that assembly, not a claimed new quantizer run.

Each offered folder includes `recipe.txt`, `tensor-types.tt`, `plan.txt`, `quantize.log`, and `sha256sums.txt`. The explicit tensor map and release size plan are reconstructed from recorded headers; the recipe is the historical rule file, and assembly/body evidence is labeled in each log. The complete descriptor inventory is in [TENSOR-INVENTORY.md](TENSOR-INVENTORY.md). The original first part contains no tensors: only its importance-matrix path is shortened to `imatrix_unsloth.gguf_file`; all weights are unchanged. See [VARIANTS.md](VARIANTS.md) for source identities and hashes. All PLE5 folders reuse the byte-identical 38,400,184,512-byte Q5_1 shard, SHA-256 `49339744a76e378904a6574ff469c5fa44c60388c411f53abda4c915d13a8061`. Its completed build/splice and independent rehash receipts agree; full-set checks read headers only. All PLE8 folders use the byte-identical table shard, SHA-256 `cde244f3da0df479d14fb328dd4ef9fcb5e048b594a4f615974730132dcf1acb`; repeated content is eligible for Hub storage deduplication, which does not remove each folder's logical download size.

## Repository layout

```text
README.md  LICENSE  VARIANTS.md  MEASUREMENTS.md  UD-EVIDENCE.md  TENSOR-INVENTORY.md
Quality-PLE8/  Qwen3.8-Flash-Next-Splash-Quality-PLE8-Q5_K_XL-00001-of-00008.gguf … 00008-of-00008.gguf
Quality-PLE5/  Qwen3.8-Flash-Next-Splash-Quality-PLE5-Q5_K_XL-00001-of-00008.gguf … 00008-of-00008.gguf
Quality-PLE4/  Qwen3.8-Flash-Next-Splash-Quality-PLE4-Q5_K_XL-00001-of-00008.gguf … 00008-of-00008.gguf
mmproj-Qwen3.8-Flash-Next-BF16.gguf
imatrix_unsloth.gguf_file
kld/  KLD-TABLE.md  METHOD.md  PAIRED-RELEASE.md  logs
```

## Limitations and known issues

- **Runtime.** The agent sweep, speed, memory and E2E results come from earlier Splash builds with native MTP. Splash 1.4.0 decodes with the DFlash2 draft and plans memory itself, so its speed, memory and output with these files can differ; they have not been measured yet.
- **Hub install path.** Installing from the Hub with stock Splash 1.4.0 was checked on one variant of this body (see [Splash 1.4.0 install check](#splash-140-install-check)). The other table variants use the same body.
- **Wired limit and hardware.** Every measurement is from one M5 Max 128 GB with `iogpu.wired_limit_mb=110592`. At the default limit Splash plans a smaller budget and may hold less context or stream more of the model from the SSD; that is untested, as are other Macs and memory sizes.
- **Reasoning effort.** With the GGUFs' own chat template, `high` renders as `xhigh`; the benchmark's distinct high level came from a custom template.
- **Answer format.** In the FinQA task, three of these four PLE8 variants (and six of the nine sweep models) put source references after a required `FINAL ANSWER:` line. If you parse the last line, say so in the prompt or post-process.
- **Context.** Long tool-heavy sessions can fill the window: UD's F3 attempt used the whole 131,072-token working context. Beyond 176K is untested with these files; the 1M result used original UD on an earlier build.
- **Reproducibility.** Batched kernels make outputs depend on batch composition and cache state, so even greedy outputs are not bitwise reproducible across those conditions.
- **Not validated** beyond the text-only agent sweep and the six-prompt release smoke: image/PDF correctness (vision is on by default, but our checks only loaded the projector), streaming and Responses API details, cancellation and cache behaviour, full 262,144-token correctness, and long-context or code-text KLD.
- **Small samples.** 16 single-trial agent tasks, a common BFCL harness defect, one blind SVG rater, and permission-rejection confounds in the earlier coding runs.
- **PLE5** has no task benchmarks (including SVG and coding). All three PLE5 variants have measured KLD: Quality 0.013128, Balanced 0.013727, Max 0.012848. UD+Q5_1 measured city +2.9%, approximately equal to UD+Q8, with prose/TTFT within noise; those UD-body observations do not establish speed for the custom PLE5 bodies.
- **Quality gaps.** Quality-PLE8 lacks matched city/prose, 4K–64K TTFT/prefill and 128K/170K fit and decode on the sweep build. Quality-PLE4 also lacks sweep-build speed, SVG and coding; its historical HTTP/113K smoke is not current validation.

## Changelog

**v3 → v4 (Splash 1.4.0).** The model files are unchanged. The MTP head is removed and this card is rewritten.

- Instructions now use the released Splash 1.4.0 (`brew install incoai/tap/splash`, then `splash serve --model REPO:FOLDER`). The source build, PR checkout and `SPLASH_MTP_*` settings are gone.
- Speculative decoding is DFlash2: Splash downloads the released draft by itself. The native MTP head (`mtp/`) is removed from the repository and the instructions; MTP-era measurements stay, labelled as such.
- Vision is on by default; memory, context and PLE streaming are planned automatically. The earlier benchmark flags are listed as optional.
- Download sizes no longer include the 2.787 GB MTP head. Added the Hub-panel note, family-repository links and a placeholder for the 1.4.0 install check.

**Earlier.** v3 renamed the custom-body shards from `-Mixed-` to `-Q5_K_XL-` so the Hub's GGUF panel classifies them (contents and SHA-256 unchanged), moved to public per-model repositories, and added the scored agent sweep, the release E2E smoke and the 1M-context note. v2 split the release into one repository per expert body, added Quality-PLE8 and Max-PLE8 with measured KLD and paired uncertainty, added blind SVG/coding results and exact tensor mixes, and put valid YAML at the top of the card.

## Credits

- **unsloth**: source BF16 GGUF conversion, importance matrix and BF16 vision projector, redistributed unchanged. Inherited `general.quantized_by`, `general.repo_url` and `general.tags` still name unsloth.
- **Qwen**: the base model and license.
- **Splash contributors**: the [runtime](https://github.com/incoai/splash) and the [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2).
- The BF16 reference logits and neutral evaluation text come from the public datasets cited in [kld/METHOD.md](kld/METHOD.md).

## Licence

These files are derivatives of Qwen3.8-Flash-Next under **Qwen Community License 1.0**. Read [LICENSE](LICENSE) for the full terms; it is identical to the base repository's license.
