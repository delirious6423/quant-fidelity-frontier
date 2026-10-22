---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: gguf
pipeline_tag: image-text-to-text
quantized_by: unsloth
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

# Qwen3.8-Flash-Next · Splash · UD-Q4_K_XL with a Q8_0 PLE table

**Our recommended default for Splash.** The body is **[unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL)**, unchanged. The only difference from unsloth's file is the per-layer embedding (PLE) table: here it is a standard Q8_0 conversion of the official BF16 table, in place of unsloth's IQ4_NL table. Nothing in the body was re-quantized; this repository is an assembly of existing files for [Splash](https://github.com/incoai/splash) 1.4.0. Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). The [family repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF) lists every variant of this release.

## Quick start

Splash 1.4.0 or newer, a 128 GB Apple-silicon Mac, macOS 26.4 or later:

```bash
brew install incoai/tap/splash
splash serve --model Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF:UD-Q4_K_XL-PLE8
```

Splash resolves `:UD-Q4_K_XL-PLE8` to the four split parts in that folder. The first start downloads them (136.9 GB) and the vision projector (0.91 GB) from this repository. Splash then adds the DFlash2 draft it releases for Qwen3.8-Flash-Next, [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (1.21 GB), by itself. It picks the draft from the model family it reads in the GGUF, not from the repository name, so you don't pass `--draft-model`; this repository ships no draft. Later starts reuse the downloads.

- **Vision is on by default**, through the BF16 projector at the repository root. Add `--language-only` to serve text only.
- **Memory and context are planned automatically** from the GPU working set. `--ple-memory` defaults to `auto`, which streams the PLE table from the SSD when memory can't hold it. `--max-memory` caps Splash to leave room for other applications.
- **Wired limit.** All our measurements used `iogpu.wired_limit_mb=110592` on an M5 Max 128 GB (`sudo sysctl iogpu.wired_limit_mb=110592`, resets on reboot). A Mac at the default limit is untested.

## At a glance

| | |
|---|---|
| **What** | unsloth's UD-Q4_K_XL body, unchanged, plus a standard Q8_0 PLE table (54.4 GB) spliced into shard 2 in place of the stock IQ4_NL table (28.8 GB). Includes unsloth's BF16 vision projector. |
| **For** | 128 GB Apple-silicon Macs. It has the fastest measured decode and the smallest resident body in the family. Its measured KLD is 0.021037, higher than the custom bodies' 0.0125–0.0133. |
| **Memory actually needed** | **About 82–89 GB of unified memory.** The resident body is 82.5 GB (decimal), and Splash's memory plan at load was 85.5 GB plus runtime buffers (82.4 GiB in use after load). The 54.4 GB PLE table **stays on the SSD** and streams, so it never needs to fit in memory. |
| **Disk / download** | 137.8 GB (target 136.9 GB + projector 0.9 GB, decimal), plus the 1.21 GB DFlash2 draft from its own repository. |
| **Peak memory, M5 Max 128 GB** | 101.3 GiB Splash Metal peak over the agent sweep (long tool-heavy sessions with KV cache); 82.4 GiB after load (text-only, 176K context). |
| **Decode** | 96.1 tok/s (agent sweep, high reasoning, native MTP, output-weighted). |
| **Runtime** | Splash 1.4.0 or newer, with the DFlash2 draft. Memory, peak and decode above were measured on the earlier Splash build with native MTP (2026-10-19); the 1.4.0 install check passed (see Benchmarks). |

**Hub GGUF panel.** The panel groups split files by the quant tag in their filenames and shows the total download size, including the PLE table, which streams from the SSD and is not held in memory. It reads `Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-00001-of-00004.gguf` as **UD-Q4_K_XL** (PLE8 is not a recognized tag), so expect one UD-Q4_K_XL entry of about 137 GB, of which 54.4 GB is the streamed table. The [family repository](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF) lists one entry per variant. The memory actually needed is the resident body in the [memory table](#memory-and-storage).

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
  "model": "Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF:UD-Q4_K_XL-PLE8",
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
- `--disable-ane`, `--idle-release off` and `--max-cache-disk 0` (SSD cache off) match the benchmarks. With the default `--idle-release 10m`, the weights are freed after 10 idle minutes and reloaded by the next request (loading took 12.9 s in the sweep).
- `int8` KV and `fp32` state are Splash's defaults and what the measurements used.
- `--served-model-name flash-next` adds a short model name for API calls.
- **Reasoning:** the benchmark requested `"reasoning_effort": "high"` through a custom chat template with a distinct high level. The GGUFs' own template supports `low`, `medium` and `xhigh`, so Splash renders `high` as `xhigh`. `"none"` turns reasoning off; `--default-reasoning-effort` sets a server default. Sweep sampling was temperature 1.0, top_p 0.95, top_k 20, min_p 0.

**Offline.** After one online start has installed the model, add `--offline` (or set `HF_HUB_OFFLINE=1`) to start it without contacting the Hub. A 40-hex `--revision` keeps Splash on one repository commit. To pre-fetch the files only:

```bash
hf download Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF --include "UD-Q4_K_XL-PLE8/*.gguf" "mmproj-*"
```

## Memory and storage

Measured on the earlier Splash build with native MTP (2026-10-19), not on Splash 1.4.0.

| Variant | Target GB | Download GB | PLE table, GB | Resident body GB | Peak after load, GiB | Sweep peak, GiB | Weights loaded |
|---|---:|---:|---|---:|---:|---:|---:|
| [UD-Q4_K_XL-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF/tree/main/UD-Q4_K_XL-PLE8) | 136.935 | 137.842 | Q8_0, 54.400 | 82.523 | 82.376 | 101.254 | 12.9 s |

Text-only (no projector), 176K context, int8 KV, fp32 state, 2 GiB PLE cache and a 110,592 MB wired limit. *Download* adds the 0.908 GB projector to the target shards; the DFlash2 draft (1.21 GB) comes from its own repository, and `imatrix_unsloth.gguf_file` (0.580 GB) is provenance only, which Splash does not download. *Resident body* is the target tensor bytes without the streamed table, from the GGUF headers. *Peak after load* is Splash's Metal allocation high-water mark after loading plus a 64-token warmup; *sweep peak* is the highest over the roughly one-hour agent run. Neither is process RSS. Whole-machine memory in use peaked at 124.2–124.6 GiB during the four PLE8 sweep runs, including the sandboxed harness, and swap never grew above its starting baseline (about 1.0 GiB). *Weights loaded* is from the server log; the OS file cache was not controlled. The release E2E table also loads the projector.

## Benchmarks (M5 Max 128 GB)

### Splash 1.4.0 install check

<!-- E2E-140 -->
**Passed, 2026-10-22.** This ran on stock Splash 1.4.0 (`1482f6bb`) on an M5 Max 128 GB with the wired limit at 110,592 MB.

- **How it ran:** exactly as a user runs it, `splash serve --model REPO:FOLDER`, with Splash's defaults (vision on, automatic memory and context) plus `--max-cache-disk 0`.
- **What it downloaded:** the Hugging Face cache already held these exact files, matched by SHA-256, so the large shards weren't downloaded again. On the first start, Splash fetched the 1.21 GB DFlash2 draft from its own repository by itself, and nothing else.
- **Context:** its automatic plan held the model's full 262,144-token window.

| Model id | Process start to `Ready` | Served context | Vision | Splash Metal peak | Checks passed |
|---|---:|---:|---|---:|---|
| `Vardaan97/Qwen3.8-Flash-Next-Splash-GGUF:UD-Q4_K_XL-PLE8` | 183 s (incl. draft download) | 256K | on | 80.9 GiB | 4/4 |

The checks:

- 17 × 23 answered 391.
- A code planted in an ~8K-token prompt was retrieved.
- Two `get_weather` calls (Paris and Tokyo, celsius) came back as parsed tool calls.
- A synthetic receipt image was read into a JSON schema with all 6 fields correct.

The time to `Ready` includes loading the weights; the first start also includes the draft download. Peak memory is Splash's Metal allocation high-water mark after the checks, not process RSS.

The family repository's `UD-Q4_K_XL-PLE8` folder holds the same bytes as this repository's (identical SHA-256).

### DFlash2 speed before 1.4.0, 2026-10-20

A DFlash2 speed run of UD-Q4_K_XL with a Q8_0 table, on the upstream main branch just before 1.4.0 (text only, 176K context), is reported on the [UD-Q5_K_XL card](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q5_K_XL-Splash-GGUF#speed-with-dflash2-2026-10-20): 134.0 tok/s for one stream, 174.7 tok/s in total with 4 concurrent streams. It is not a 1.4.0 result.

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

Original unsloth UD-Q4_K_XL with its stock **IQ4_NL** table, factor 4, max context 1010000, int8 KV, fp32 state, 2 GiB PLE cache: at 1000000 prompt tokens it retrieved **8/8 planted facts** and returned NOT_FOUND for the absent fact, with **614.3 s cold TTFT**, 1628 prefill tok/s, 28.5 decode tok/s and **91.9 GiB Splash peak**, no new swap. Retrieval was also 8/8 at 262K and 520K. The sum-of-amounts test failed at 520K and 1M. This does not validate arbitrary long-context reasoning, other quants, or this release's Q8-table UD assembly at 1M. [Reproduction](https://github.com/incoai/splash/pull/469#issuecomment-6256428597).

### Release E2E smoke, 2026-10-19

Measured on the earlier Splash build with native MTP (2026-10-19), not on Splash 1.4.0.

<!-- E2E_RESULTS:BEGIN -->
| Variant | Load s | Peak Metal GiB | Warmup MTP % | Functional /5 | Prose tokens | Decode tok/s | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| UD-Q4_K_XL-PLE8 | 17.038 | 83.559 | 68.831 | 5/5 | 300 | 79.716 | PASS |
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

## Exact UD tensor mix

This retains unsloth’s per-layer mixture. Only `per_layer_token_embd.weight` changes from IQ4_NL to Q8_0. In particular, this is not a uniform Q4_K model. The [exact-name map](UD-Q4_K_XL-PLE8/tensor-types.tt) and [inventory](TENSOR-INVENTORY.md) cover all 1,224 tensors.

| Tensor suffix or global name | Exact type → zero-based layers |
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
| `ffn_gate_exps.weight` | Q4_K: 0–1, 3–47; Q5_K: 2 |
| `ffn_gate_inp.weight` | F32: 0–47 |
| `ffn_gate_inp_shexp.weight` | F32: 0–47 |
| `ffn_gate_shexp.weight` | Q8_0: 0–47 |
| `ffn_up_exps.weight` | Q4_K: 0–1, 3–47; Q5_K: 2 |
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
| `per_layer_token_embd.weight` | Q8_0: global |
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

| Type | Tensors | Parameters | Tensor bytes |
|---|---:|---:|---:|
| BF16 | 24 | 19,660,800 | 39,321,600 |
| F32 | 557 | 78,373,760 | 313,495,040 |
| Q4_K | 94 | 78,852,915,200 | 44,354,764,800 |
| Q5_1 | 43 | 36,071,014,400 | 27,053,260,800 |
| Q5_K | 2 | 1,677,721,600 | 1,153,433,600 |
| Q8_0 | 504 | 60,244,213,760 | 64,009,477,120 |

## UD table and runtime evidence

Measured on older Splash builds with native MTP (the earlier STACK and the #447-based tests), before the 2026-10-19 sweep build, not on Splash 1.4.0. All figures are single-stream observations on a 128 GB M5 Max (40-core GPU), int8 KV, fp32 state, a 2 GiB streamed PLE row cache and GDN checkpoint grid 16,384. Decimal target download is 136.935 GB, excluding the 0.908 GB projector and 0.580 GB importance matrix. Native body tensor bytes excluding PLE are 82,523,491,840; this is not full runtime RAM. Q8-table phase-3 peak allocation was 89.28 GiB; runtime repacking, KV and caches are additional to native tensor bytes.

| Q8-table assembly on earlier STACK | City tok/s | MTP acceptance | Cold 4K / 32K / 64K TTFT, s | 128K / 170K cold TTFT, s | Decode at 128K / 170K, tok/s |
|---|---:|---:|---|---|---|
| UD cells | 162.7 | 5.360 accepted/cycle | 32K: 17.43 | — | — |
| pr447 phase 1 / 1b / 3, Q8 arm | 162.727 | city acceptance rate 0.786 | 1.68 / 17.63 / 41.33 | 101.56 / 137.73 | 95.7 / 104.2 |

The UD-cells paired city change versus original IQ4_NL was +3.04% throughput and +4.83% accepted/cycle (5.113 → 5.360). That table experiment had incomplete/tainted prose; it does not establish a general prose speedup. Phase 1b Q8 prose aggregate was 112.092 tok/s, but both reference brackets were disk-tainted; Q8 cold TTFT cells were also disk-tainted. Phase 3 Q8 was clean, one launch; reference brackets included disk/swap taints. Q8 output identity differs as expected when table precision changes. Table-location results were city −0.10% and cold 32K TTFT +1.62% on NVMe versus internal for Q8; these are marginal single-protocol observations, not a storage guarantee.

**The #448 measurements below used the same unsloth UD body with its original IQ4_NL table, not this Q8-table assembly.** No #448 Q8-table run is claimed.

| Runtime on original UD | City tok/s | 72-prompt aggregate tok/s | Cold 4K / 32K / 64K TTFT, s | Warm / 64K deep-edit TTFT, s |
|---|---:|---:|---|---|
| #448 on #447, defaults + grid | 155.0 | 114.8 / 114.5 | 1.70 / 14.46 / 31.02 | 0.09–0.10 / 8.32 |
| #448, PLE prefetch off | — | 113.0 | — | — |
| Fixed #447, zero-proposal placeholder | — | 33.1 | — | — |

The last row is an unbracketed no-drafting proxy, not a genuine matched same-tree MTP-off experiment. Native MTP averages 114.7 tok/s in the two prose launches; one was disk-tainted. City was cooling-gated; prose/TTFT were warm, with a disk-tainted 4K cell. Do not treat the rows as a controlled general speedup.

| Original UD, phase 3 | Cold 128K / 170K TTFT, s | Decode at depth, tok/s | Peak allocation |
|---|---|---|---|
| #448 on #447 | 67.99 / 95.37 | 100.2 / 93.4 | 89.06 GiB |
| Earlier STACK reference brackets | 98.76 / 139.44; 100.47 / 140.71 | 101.1 / 91.9; 98.5 / 93.4 | 89.27 GiB |

Depth prompts were 131,072 and 174,080 tokens followed by 256 greedy output tokens; requested context was 176K. One launch per arm, no cooling gate. The #448 128K cell and references carried disk taints; the final reference also had swap-ins. The ledger answer lay outside the generated prompts, so this is a timing probe, not a long-context retrieval/quality pass. #448 outputs matched the no-draft proxy on 30/72 prompts, or 72/72 when verify-row composition matched; the batched kernels depend on verify-row composition.

## Quality evidence and estimate

**Q8-table KLD: 0.021037 ± 0.000163 (scorer SE), same top-1 94.807%, PPL ratio 1.004707** (measured 2026-10-19). Same scorer and reference as every other row. To fit the 128 GB Mac, the measurement used a 5-part re-split of the exact release files in which the PLE table sits in its own part. All 1,224 tensors were verified byte-identical, so the weights are unchanged. That is 5.2 % lower than the original IQ4_NL-table baseline (0.022195), and about 1.6× the custom PLE8 bodies' KLD. See [method](kld/METHOD.md).

Original UD scored **coding 278.68/320** (13 tasks, one run, thinking off, permission-rejection confounds) and **blind SVG 3.60/5**, 8/9 valid (3 prompts × 3 seeds, thinking on, one blind rater, invalid/capped outputs score 1). These used the original IQ4_NL table and earlier STACK. They are context for unsloth’s body, not reruns of this Q8-table assembly. Small sample differences do not establish a task ranking.

## Assembly and checksums

Parts 1, 3 and 4 are byte-identical to unsloth’s pinned release. Part 2 keeps every body tensor byte-identical (296 tensors) and swaps only `per_layer_token_embd.weight`: unsloth's IQ4_NL table (4-bit, 28.80 GB) becomes a standard Q8_0 table (54.40 GB). Historical splice verification matched source-body and donor-table payload hashes. Part-2 full SHA-256: `7d8b4597b27356c8645fd41078979b382b1cd1233cd388fda0326f5b28038b43`. No payload is rehashed merely to prepare this card. [Exact four filenames, sizes and SHA-256](VARIANTS.md); [assembly recipe](UD-Q4_K_XL-PLE8/recipe.txt). The BF16 projector and importance matrix remain unchanged.

## Repository layout

```text
README.md  LICENSE  VARIANTS.md  MEASUREMENTS.md  UD-EVIDENCE.md  TENSOR-INVENTORY.md
UD-Q4_K_XL-PLE8/  Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-00001-of-00004.gguf … 00004-of-00004.gguf
mmproj-Qwen3.8-Flash-Next-BF16.gguf
imatrix_unsloth.gguf_file
kld/  KLD-TABLE.md  METHOD.md  PAIRED-RELEASE.md  logs
```

## Limitations and known issues

- **Runtime.** The agent sweep, speed, memory and E2E results come from earlier Splash builds with native MTP. Splash 1.4.0 decodes with the DFlash2 draft and plans memory itself, so its speed, memory and output with these files can differ; they have not been measured yet.
- **Hub install path.** Installing from the Hub with stock Splash 1.4.0 passed (see [Splash 1.4.0 install check](#splash-140-install-check)).
- **Wired limit and hardware.** Every measurement is from one M5 Max 128 GB with `iogpu.wired_limit_mb=110592`. At the default limit Splash plans a smaller budget and may hold less context or stream more of the model from the SSD; that is untested, as are other Macs and memory sizes.
- **Reasoning effort.** With the GGUFs' own chat template, `high` renders as `xhigh`; the benchmark's distinct high level came from a custom template.
- **Answer format.** In the FinQA task, three of these four PLE8 variants (and six of the nine sweep models) put source references after a required `FINAL ANSWER:` line. If you parse the last line, say so in the prompt or post-process.
- **Context.** Long tool-heavy sessions can fill the window: UD's F3 attempt used the whole 131,072-token working context. Beyond 176K is untested with these files; the 1M result used original UD on an earlier build.
- **Reproducibility.** Batched kernels make outputs depend on batch composition and cache state, so even greedy outputs are not bitwise reproducible across those conditions.
- **Not validated** beyond the text-only agent sweep and the six-prompt release smoke: image/PDF correctness (vision is on by default, but our checks only loaded the projector), streaming and Responses API details, cancellation and cache behaviour, full 262,144-token correctness, and long-context or code-text KLD.
- **Small samples.** 16 single-trial agent tasks, a common BFCL harness defect, one blind SVG rater, and permission-rejection confounds in the earlier coding runs.
- **UD+Q8 gaps.** This Q8-table assembly has no SVG/coding rerun; its results on the sweep build are the agent sweep, the release E2E and its KLD. Original-table #448 results do not validate it. Q8 prose comparisons and TTFT controls on the earlier builds carry the taints described above, and the 128K/170K timing is not a retrieval-quality pass.

## Changelog

**v3 → v4 (Splash 1.4.0).** The model files are unchanged. The MTP head is removed and this card is rewritten.

- Instructions now use the released Splash 1.4.0 (`brew install incoai/tap/splash`, then `splash serve --model REPO:FOLDER`). The source build, PR checkout and `SPLASH_MTP_*` settings are gone.
- Speculative decoding is DFlash2: Splash downloads the released draft by itself. The native MTP head (`mtp/`) is removed from the repository and the instructions; MTP-era measurements stay, labelled as such.
- Vision is on by default; memory, context and PLE streaming are planned automatically. The earlier benchmark flags are listed as optional.
- Download sizes no longer include the 2.787 GB MTP head. Added the Hub-panel note, family-repository links, a pointer to the pre-1.4.0 DFlash2 speed run and a placeholder for the 1.4.0 install check.

**Earlier.** v3 moved to a public repository with byte-identical model files and added the scored agent sweep, the release E2E smoke and the 1M-context note.

## Credits and licence

**unsloth** created the UD-Q4_K_XL body quantization and supplied the BF16 GGUF conversion, importance matrix and BF16 projector; the PLE table is a standard Q8_0 conversion of the official BF16 table. Inherited unsloth provenance metadata stays intact. **Qwen** supplied the base model; these derivatives remain under **Qwen Community License 1.0**. [LICENSE](LICENSE) is the same as the base; this assembly grants no broader rights. The **Splash contributors** made the [runtime](https://github.com/incoai/splash) and the [DFlash2 draft](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2). The neutral BF16 reference logits and evaluation text come from the public datasets cited in [kld/METHOD.md](kld/METHOD.md).
