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
- mtp
- mixed-precision
---

# Qwen3.8-Flash-Next · Splash · UD-Q4_K_XL with a Q8_0 PLE table

**Our recommended default for Splash.** The body is **[unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-Q4_K_XL)**, unchanged. The only difference from unsloth's file is the per-layer embedding (PLE) table: here it is a standard Q8_0 conversion of the official BF16 table, in place of unsloth's IQ4_NL table. Nothing in the body was re-quantized; this repository is an assembly of existing files for [Splash](https://github.com/incoai/splash). Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

## At a glance

| | |
|---|---|
| **What** | unsloth's UD-Q4_K_XL body, unchanged, plus a standard Q8_0 PLE table (54.4 GB) spliced into shard 2 in place of the stock IQ4_NL table (28.8 GB). Includes the stock MTP head and BF16 vision projector. |
| **For** | 128 GB Apple-silicon Macs. It has the fastest measured decode and the smallest resident body in the family. Its measured KLD is 0.021037, higher than the custom bodies' 0.0125–0.0133. |
| **Memory actually needed** | **About 82–89 GB of unified memory.** The resident body is 82.5 GB (decimal), and Splash's memory plan at load is 85.5 GB plus runtime buffers (82.4 GiB in use after load). The 54.4 GB PLE table **stays on the SSD** and streams through a 2 GiB cache (`--ple-memory 2G`), so it never needs to fit in memory. |
| **Disk / download** | 140.6 GB on disk (target 136.9 GB + MTP head 2.8 GB + projector 0.9 GB, decimal). The Hub's GGUF panel shows this file size (about 137 GB for the target), **not** the memory needed. |
| **Peak memory, M5 Max 128 GB** | 101.3 GiB Splash Metal peak over the agent sweep (long tool-heavy sessions with KV cache); 82.4 GiB after load (text-only, 176K context). |
| **Decode** | 96.1 tok/s (agent sweep, high reasoning, native MTP, output-weighted). |
| **Splash build** | Build from source at PR [#448](https://github.com/incoai/splash/pull/448) commit `fa842cb5`: Splash 1.3.2 plus the upstream Flash-Next stack (#449–#468) plus native MTP. **The released Splash 1.3.2 package cannot load Qwen3.8-Flash-Next.** Recipe below. |

## Which one should I pick

All nine files are the same model on the same Splash build. They trade disk, memory, speed and weight error. **On agent tasks the four tested PLE8 variants are indistinguishable:** 7–8/16 strict and 8–9/16 with the lenient FinQA re-grade, a one-task spread that is within noise for 16 single-trial tasks. PLE5 and PLE4 were not task-tested.

**Our default:** UD-Q4_K_XL-PLE8. It has the fastest decode, the lowest memory and the quickest load, and its agent-task scores are within noise of the others. We keep Quality-PLE8 as the quality reference, since its measured weight error is lowest.

| If you want | Pick | Measured basis (M5 Max 128 GB) |
|---|---|---|
| Fastest decode, smallest weights | [UD-Q4_K_XL-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF/tree/main/UD-Q4_K_XL-PLE8) | 96.1 tok/s agent-sweep decode vs 82.3–86.7 for the custom PLE8 bodies; 82.4 GiB after load vs 95.4–98.2; 82.5 GB resident body vs 96.5–99.1. KLD 0.021037 measured (original UD-Q4_K_XL with its IQ4_NL table: 0.022195). |
| Lowest measured weight error | [Quality-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF/tree/main/Quality-PLE8) or [Max-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Max-GGUF/tree/main/Max-PLE8) | Neutral KLD 0.012613 and 0.012537, a statistical tie (0.6 paired SE). Both are more than 5 paired SE below Balanced-PLE8. Max's resident body is 0.75 GB larger. |
| A custom body with a little more memory room | [Balanced-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Balanced-GGUF/tree/main/Balanced-PLE8) | 96.5 GB resident body vs 98.4 (Quality) and 99.1 (Max); KLD 0.013270. |
| Less disk, same body | PLE5 (Quality, Balanced, Max) | 38.4 GB table instead of 54.4 GB (−16.0 GB). Balanced-PLE5 measured +3.44% KLD vs Balanced-PLE8; no task run. |
| Least disk | PLE4 (Quality, Balanced) | 28.8 GB table (−25.6 GB vs PLE8). Measured KLD +13.6% (Quality) and +12.6% (Balanced) vs PLE8. |

The PLE table is streamed from the SSD through a 2 GiB cache, so the table choice changes disk size, not the resident body; PLE5/PLE4 runtime memory is in the release E2E table. Sweep decode rates come from each model's own agent workload, so they are descriptive rather than a controlled A/B. On the earlier build, UD was also faster on prose (122.8 vs 114.4–115.3 tok/s) but not on the 2K greedy city test (157.7 vs 158.3–161.9).

These custom variants were designed around 128 GB unified memory and PLE streaming. A full 262,144-token base context is a model capability, not an end-to-end validation for each quant.

## Run it with Splash

**1. Build Splash.** The released Splash 1.3.2 package has no Flash-Next support. Build PR [#448](https://github.com/incoai/splash/pull/448) at `fa842cb5`: Splash 1.3.2 plus the upstream Flash-Next stack [#449–#468](https://github.com/incoai/splash/pull/468) plus our native-MTP commits. The agent sweep and the release E2E used a build of this commit (`0beba3b2` adds two test-only commits; its production binary is byte-identical). Building needs macOS 26.4 or later, Xcode 26 or newer, Python 3.12–3.14 and a Metal 4 compiler.

```bash
git clone https://github.com/incoai/splash.git
cd splash
# The measured build. Its commits are kept on this branch since #448 was rebased onto #469.
git fetch https://github.com/Vardaan97/splash.git flashnext-mtp-on-468-f84a
git checkout --detach fa842cb514ca6e181a352f5541d4550d22a8ba86
make -j4
```

**Newer build (optional).** PR #448 now sits on the upstream YaRN stack (#469). Its current head adds `--yarn-factor` long context and later fixes. It passes the full `make check`, and with UD-Q4_K_XL-PLE8 and the stock MTP head it decoded at 166 and 159 tok/s on 262K- and 520K-token prompts (`--yarn-factor 4`). The agent sweep and the release E2E above were **not** re-run on it. To use it, replace the two `git` lines with `git fetch origin pull/448/head && git checkout --detach FETCH_HEAD`.

**2. Serve.** These are the benchmark settings. The first start downloads 140.6 GB into the Hugging Face cache and installs the model.

```bash
sudo sysctl iogpu.wired_limit_mb=110592   # resets on reboot
SPLASH_MTP_ADAPTIVE=1 SPLASH_MTP_DRAFTS=7 \
SPLASH_MTP_DRAFT_VOCAB=65536 SPLASH_MTP_DRAFT_VOCAB_WARMSTART=1 \
SPLASH_MTP_DRAFT_VOCAB_GATE=0.06 SPLASH_GDN_CHECKPOINT_TOKENS=16384 \
SPLASH_PLE_PREFETCH=1 \
./splash serve --model Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF:UD-Q4_K_XL-PLE8 \
  --ple-memory 2G --kv-format int8 --state-format fp32 --max-context 176K \
  --disable-ane --idle-release off --max-cache-disk 0 \
  --served-model-name flash-next
```

**3. Call it** (OpenAI-compatible, with tools). This is the release E2E's tool-call check, which expects two parsed `get_weather` calls:

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "flash-next",
  "messages": [{"role": "user", "content": "Call get_weather exactly twice: once for Paris and once for Tokyo, both in celsius. Do not answer in prose."}],
  "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Read weather for one city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius"]}},
                   "required": ["city", "unit"], "additionalProperties": false}}}],
  "tool_choice": "auto", "parallel_tool_calls": true,
  "reasoning_effort": "high", "max_tokens": 4096
}'
```

**Settings**

- `--ple-memory 2G` is required on 128 GB: the PLE table (54.4, 38.4 or 28.8 GB) stays on the SSD behind a 2 GiB row cache.
- `--max-context 176K` (180,224 tokens) is the tested server limit; the agent harness kept each request within 131,072 tokens. The native 262,144-token window is not validated for these files. 1M context needs a different build (see Benchmarks).
- `int8` KV and `fp32` state are Splash's defaults. `--disable-ane`, `--idle-release off` and `--max-cache-disk 0` (SSD prefix cache off) reproduce the benchmark; Splash's defaults for those three were not benchmarked. With Splash's default `--idle-release 10m`, the weights are freed after 10 idle minutes and reloaded by the next request (loading took 12.9 s in the sweep).
- The seven `SPLASH_*` switches are the benchmark's MTP and PLE settings: adaptive depth up to 7 drafts, a 65,536-token draft vocabulary with warm start, gate 0.06, a GDN checkpoint every 16,384 tokens, and PLE prefetch.
- **Native MTP:** leave `--draft-model` unset. The installer takes `mtp/mtp-*-shared-Q8_0.gguf` from the repository.
- **Text only:** add `--language-only` to skip the vision projector. The agent sweep ran text-only.
- **Reasoning:** the benchmark requested `"reasoning_effort": "high"` through a custom chat template with a distinct high level. The GGUFs' own template supports `low`, `medium` and `xhigh`, so Splash renders `high` as `xhigh`. `"none"` turns reasoning off; `--default-reasoning-effort` sets a server default. Sweep sampling was temperature 1.0, top_p 0.95, top_k 20, min_p 0.

**Offline.** After one online start has installed the model, add `--offline` (or set `HF_HUB_OFFLINE=1`) to start it without contacting the Hub. The sweep and E2E servers ran with `HF_HUB_OFFLINE=1` from local files. A 40-hex `--revision` keeps Splash on one repository commit. To pre-fetch one folder only:

```bash
hf download Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF --include "UD-Q4_K_XL-PLE8/*.gguf" "mtp/*" "mmproj-*"
```

The v2 recipe (#448 on #447 at `6db6e5ee`, with `--ple-ssd-cache`) is superseded: #448 now sits on the #449–#468 stack, where the flag is `--ple-memory`.

## Memory and storage

| Variant | Target GB | Download GB | PLE table, GB | Resident body GB | Peak after load, GiB | Sweep peak, GiB | Weights loaded |
|---|---:|---:|---|---:|---:|---:|---:|
| [UD-Q4_K_XL-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF/tree/main/UD-Q4_K_XL-PLE8) | 136.935 | 140.629 | Q8_0, 54.400 | 82.523 | 82.376 | 101.254 | 12.9 s |

Measured on the M5 Max 128 GB with the build and settings above, text-only (no projector), 176K context, int8 KV, fp32 state, 2 GiB PLE cache and a 110,592 MB wired limit. *Download* adds the 2.787 GB MTP head and 0.908 GB projector to the target shards; `imatrix_unsloth.gguf_file` (0.580 GB) is provenance only and Splash does not download it. *Resident body* is the target tensor bytes without the streamed table, from the GGUF headers. *Peak after load* is Splash’s Metal allocation high-water mark after loading plus a 64-token warmup; *sweep peak* is the highest over the roughly one-hour agent run. Neither is process RSS. Whole-machine memory in use peaked at 124.2–124.6 GiB during the four PLE8 sweep runs, including the sandboxed harness, and swap never grew above its starting baseline (about 1.0 GiB). *Weights loaded* is from the server log; the OS file cache was not controlled. The release E2E table also loads the projector.

## Benchmarks (M5 Max 128 GB)

### Scored agent sweep, 2026-10-18/19

**Method.** Nine local Flash-Next builds ran one at a time through the same sandboxed agent harness on the M5 Max. Each faced 16 deterministically graded tasks from public benchmarks (one trial each), four repeat attempts (F1 and H3 twice more) and one separately judged finance write-up (F3). All used the same Splash build (`0beba3b2`, production-identical to `fa842cb5`), native MTP, text-only serving, reasoning high, temperature 1.0 / top_p 0.95 / top_k 20, identical task order and seeds, a fresh server per model, a 131,072-token working context, per-task time caps set from one UD pilot, and a two-hour window per model. The four rows below are the released PLE8 files: the same shard bytes, except that the custom bodies' tensor-free part 1 differs only by a shortened importance-matrix path. No score is transferred to PLE5 or PLE4.

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

Same BF16 reference and neutral text, 87 × 4096-token chunks, 178089 scored tokens; base PPL 4.045181. Chunk SE is used for KLD uncertainty; PPL ratios are the measured tool summary. These are distribution comparisons, not task scores.

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

Balanced-PLE5 adds 0.000457 KLD (+3.44%) versus PLE8 while saving 16.000 GB. Quality-PLE5 (0.013128) and Max-PLE5 (0.012848) were measured on 2026-10-19 with the same scorer. The earlier precision report's pending Q5_1 status is superseded by the completed Balanced receipt/log. Quality and Max PLE8 are effectively tied (0.6 paired SE). Full method, raw baseline logs and paired comparisons remain in [KLD evidence](kld/KLD-TABLE.md).

### 1M context: separate #469 YaRN build required

**Requires the upstream Splash #469 YaRN build with the fractional-tail fix (`68267851`, equivalent fix `56ab22c0`), not the bench #468 build or the run recipe above.** Original unsloth UD-Q4_K_XL with its stock **IQ4_NL** table, factor 4, max context 1010000, int8 KV, fp32 state, 2 GiB PLE cache, no native MTP: at 1000000 prompt tokens it retrieved **8/8 planted facts** and returned NOT_FOUND for the absent fact, with **614.3 s cold TTFT**, 1628 prefill tok/s, 28.5 decode tok/s and **91.9 GiB Splash peak**, no new swap. Retrieval was also 8/8 at 262K and 520K. The sum-of-amounts test failed at 520K and 1M. This does not validate arbitrary long-context reasoning, other quants, or this release's Q8-table UD assembly at 1M. [Fractional-tail fix and reproduction](https://github.com/incoai/splash/pull/469#issuecomment-6256428597).

### E2E_RESULTS

<!-- E2E_RESULTS:BEGIN -->
| Variant | Load s | Peak Metal GiB | Warmup MTP % | Functional /5 | Prose tokens | Decode tok/s | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| UD-Q4_K_XL-PLE8 | 17.038 | 83.559 | 68.831 | 5/5 | 300 | 79.716 | PASS |
<!-- E2E_RESULTS:END -->
The E2E decode figure is a single cold 300-token sample with reasoning off, so it reads lower than the agent-sweep decode rates in the glance box. The two numbers measure different things.

Completed 2026-10-19T07:14:16Z on the exact upload-source assembly, stock MTP head and shipped projector; Splash bench #468 (`0beba3b2`), int8 KV, fp32 state, 2 GiB PLE cache, native MTP. Prose timing disables reasoning and EOS for exactly 300 completion tokens; the other checks request high reasoning. Load timing is process launch to ready, with the existing OS file-cache state (not a certified cold load). The six-prompt smoke battery covers text/math, executed code assertion, two tool calls, an 8K-token needle, exact instruction format and a 300-token prose decode. Projector presence/header checks do not establish image/PDF correctness. These checks do not replace the scored sweep or validate streaming, Responses API, full-context correctness or broader hardware.

## Variants

One repository per expert body, with PLE8/PLE5 folders together and PLE4 where offered. All sizes are decimal GB; target downloads exclude the separate MTP head, projector and importance matrix. These are **mixed quantizations**, not uniform Q8/Q6/Q5/Q4 files.

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

‡ Measured in the release E2E (2026-10-19) on the exact public files: Splash's memory plan at load with `--max-context 176K`, vision projector loaded, 2 GiB PLE cache. Other rows come from the earlier v2 protocol.

PLE5 uses the same body as its PLE8 sibling and a **Q5_1 table: 38.4 GB, 6.0 bpw** (32 values per 24-byte block). Balanced-PLE5 KLD and top-1 are measured on the same neutral protocol as its PLE8/PLE4 baselines; its chunk SE is reconstructed from 87 running chunk means. Quality-PLE5 (0.013128, +4.1% vs Quality-PLE8) and Max-PLE5 (0.012848, +2.5% vs Max-PLE8) were measured on 2026-10-19 with the same scorer and reference, on the exact release files. For these two rows the SE shown is the scorer's own; the other rows use the chunk-SE reconstruction. The model treats table error as independent of body error; these estimates are not task results. PLE5 runtime memory and KV capacity have not been measured; native resident tensor bytes exclude the table and are unchanged.

Native resident bytes sum target tensor bytes with the PLE table excluded. Splash target bytes come from runtime-format status and vary with build and settings. KV capacity is shared across lanes; it is **not** a tested single-request context length. PLE8 runtime status here is from the SVG stack, with int8 KV, fp32 state, a 2 GiB PLE cache and a 110,592 MB wired limit; the SVG requests used 32,768 context. † Historical status from the earlier release build. These are separate cohorts, not matched memory comparisons. Family links are release destinations; new destinations become usable only after release. [Evidence](MEASUREMENTS.md#memory-and-files).

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

These rows come from earlier builds (the earlier STACK and the #447-based tests, including #448 on #447 at `6db6e5ee`), not the `fa842cb5` build recommended above. All figures are single-stream observations on a 128 GB M5 Max (40-core GPU), int8 KV, fp32 state, a 2 GiB streamed PLE row cache and GDN checkpoint grid 16,384. Decimal target download is 136.935 GB, excluding the 2.787 GB stock MTP head, 0.908 GB projector and 0.580 GB importance matrix. Native body tensor bytes excluding PLE are 82,523,491,840; this is not full runtime RAM. Q8-table phase-3 peak allocation was 89.28 GiB; runtime repacking, KV and caches are additional to native tensor bytes.

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

Original UD scored **coding 278.68/320** (13 tasks, one run, thinking off, permission-rejection confounds) and **blind SVG 3.60/5**, 8/9 valid (3 prompts × 3 seeds, thinking on, one blind rater, invalid/capped outputs score 1). These used the original IQ4_NL table and earlier STACK. They are context for unsloth’s body, not reruns of this Q8-table assembly or #448. Small sample differences do not establish a task ranking.

## Assembly and checksums

Parts 1, 3 and 4 are byte-identical to unsloth’s pinned release. Part 2 keeps every body tensor byte-identical (296 tensors) and swaps only `per_layer_token_embd.weight`: unsloth's IQ4_NL table (4-bit, 28.80 GB) becomes a standard Q8_0 table (54.40 GB). Historical splice verification matched source-body and donor-table payload hashes. Part-2 full SHA-256: `7d8b4597b27356c8645fd41078979b382b1cd1233cd388fda0326f5b28038b43`. No payload is rehashed merely to prepare this card. [Exact four filenames, sizes and SHA-256](VARIANTS.md); [assembly recipe](UD-Q4_K_XL-PLE8/recipe.txt). The stock MTP head, BF16 projector and importance matrix remain unchanged.

## Hub panel

The public filename parser reads `Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-00001-of-00004.gguf` as **UD-Q4_K_XL**. PLE8 is not a recognized classifier. Expect a UD-Q4_K_XL target group (136.935 GB for all four shards if grouped) and a separate Q8_0 MTP head; the panel will not explain the replaced table or streamed RAM. This is a parser-based expectation; server rendering remains unverified until publication.

## Limitations and known issues

- **Runtime.** The released Splash 1.3.2 package cannot load Qwen3.8-Flash-Next; use the `fa842cb5` build above. Other Splash revisions may change flags, speed or output.
- **Hub install path.** The release E2E loads exactly these files from a local assembly. Installing them from the Hub with `splash serve --model OWNER/REPO:VARIANT` was not exercised before publication.
- **Reasoning effort.** With the GGUFs' own chat template, `high` renders as `xhigh`; the benchmark's distinct high level came from a custom template.
- **Answer format.** In the FinQA task, three of these four PLE8 variants (and six of the nine sweep models) put source references after a required `FINAL ANSWER:` line. If you parse the last line, say so in the prompt or post-process.
- **Context.** Long tool-heavy sessions can fill the window: UD's F3 attempt used the whole 131,072-token working context. Beyond 176K is untested with these files; 1M needs the separate #469 build and was tested only with original UD.
- **Reproducibility.** Batched kernels make outputs depend on batch composition and cache state, so even greedy outputs are not bitwise reproducible across those conditions.
- **Coverage.** One 128 GB M5 Max with a raised wired limit. Other Macs and memory sizes are untested.
- **Not validated** beyond the text-only agent sweep and the six-prompt release smoke: image/PDF correctness (the projector is only loaded), streaming and Responses API details, cancellation and cache behaviour, full 262,144-token correctness, and long-context or code-text KLD. No matched MTP-off ablation exists on the recommended build.
- **Small samples.** 16 single-trial agent tasks, a common BFCL harness defect, one blind SVG rater, and permission-rejection confounds in the earlier coding runs.
- **UD+Q8 gaps.** This Q8-table assembly has no measured KLD and no SVG/coding rerun; its only results on the recommended build are the agent sweep and the release E2E. Original-table #448 results do not validate it. Q8 prose comparisons and TTFT controls on the earlier builds carry the taints described above, and the 128K/170K timing is not a retrieval-quality pass.

## Changelog: v2 → v3

- Public repository under **Vardaan97**. The model files are byte-identical to v2; only text changed.
- New at-a-glance summary, family-wide pick guide, copy-paste run recipe, memory and storage table, and known issues.
- The run recipe moves to PR #448 at `fa842cb5` on the upstream #449–#468 stack (`--ple-memory`), the build used for the agent sweep and the release E2E. The v2 pin `6db6e5ee` on #447 (`--ple-ssd-cache`) is superseded.
- Added the scored nine-model agent sweep (strict and lenient, per task), the separate F3 judge result, the 1M-context note (#469 build, original UD only) and the release E2E smoke table.

## Credits and licence

**unsloth** created the UD-Q4_K_XL body quantization; the PLE table is a standard Q8_0 conversion of the official BF16 table. **Vardaan97**: assembly and measurements. **unsloth** also supplied the conversion, importance matrix, stock shared Q8_0 MTP head and BF16 projector. **Qwen** supplied the base model; these derivatives remain under **Qwen Community License 1.0**. [LICENSE](LICENSE) is the same as the base; this assembly grants no broader rights. Inherited unsloth provenance metadata stays intact. **AtomicChat** supplied the neutral BF16 reference logits/evaluation corpus. Splash credits: janhilgard for native MTP; jianc99 for #447; the maintainers and contributors to #410/#423/#429/#448.
