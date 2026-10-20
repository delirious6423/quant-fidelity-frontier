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
- mtp
- mixed-precision
---

# Qwen3.8-Flash-Next · Splash GGUF · v3 family index

**Quantized by Vardaan97.** The release family is organized into one repository per expert body. Use the linked cards for exact tensor mixes, measured results and Splash #448 usage. This combined repository retains its existing files and download links; it is the family pointer for new releases. All four per-model repositories are public.

## Start here

1. **Pick a variant.** For most people: **[UD-Q4_K_XL-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF)**. It's the fastest, uses the least memory and loads quickest, and its agent-task scores are within noise of the others. For the lowest measured weight error, pick **[Quality-PLE8](https://huggingface.co/Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF)**. All variants need a 128 GB Apple-silicon Mac.
2. **Build Splash from source.** The released Splash 1.3.2 package can't load Qwen3.8-Flash-Next yet. The recipe is in [Run it with Splash](#run-it-with-splash): clone, fetch our measured build, `make`.
3. **Serve with `--model Vardaan97/REPO:FOLDER`**, using the [settings below](#run-it-with-splash). Leave `--draft-model` unset to use the bundled MTP head.

**Files in this repository.** It still holds three variants from the earlier releases: `Balanced-PLE8/`, `Balanced-PLE4/` and `Quality-PLE4/`, plus `mtp/`, the BF16 vision projector and the importance matrix. Every GGUF shard is byte-identical (same SHA-256) to the matching shard in the per-model repositories; the files in both places carry the same `Q5_K_XL` label (the repositories differ only in the variant prefix). Existing downloads remain valid. New downloads should use the per-model repositories, whose cards carry each variant's tensor mix, measurements and tests.

## At a glance

| | |
|---|---|
| **What** | Index of nine GGUF builds of Qwen3.8-Flash-Next for Splash, in four public Vardaan97 repositories (linked in the tables below). This repository keeps its earlier files. |
| **For** | 128 GB Apple-silicon Macs. Every measurement here is from one M5 Max 128 GB. |
| **Download** | 129.0–157.2 GB per variant (target + MTP head + projector, decimal GB); table below. |
| **Peak memory, M5 Max 128 GB** | PLE8 variants: 100.7–101.6 GiB Splash Metal peak over the agent sweep, 82.4–98.2 GiB after load (text-only, 176K context). PLE5/PLE4: release E2E table below. |
| **Decode** | PLE8 variants: 82.3–96.1 tok/s (agent sweep, high reasoning, native MTP, output-weighted); UD-Q4_K_XL-PLE8 is fastest. |
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

**2. Serve.** These are the benchmark settings. The first start downloads 129–157 GB (depending on the variant) into the Hugging Face cache and installs the model. For another variant, replace the `--model` value with that variant’s `Vardaan97/REPO:FOLDER`, such as `Vardaan97/Qwen3.8-Flash-Next-UD-Q4_K_XL-PLE8-GGUF:UD-Q4_K_XL-PLE8`.

```bash
sudo sysctl iogpu.wired_limit_mb=110592   # resets on reboot
SPLASH_MTP_ADAPTIVE=1 SPLASH_MTP_DRAFTS=7 \
SPLASH_MTP_DRAFT_VOCAB=65536 SPLASH_MTP_DRAFT_VOCAB_WARMSTART=1 \
SPLASH_MTP_DRAFT_VOCAB_GATE=0.06 SPLASH_GDN_CHECKPOINT_TOKENS=16384 \
SPLASH_PLE_PREFETCH=1 \
./splash serve --model Vardaan97/Qwen3.8-Flash-Next-Splash-Quality-GGUF:Quality-PLE8 \
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

**Settings in short:** `--ple-memory 2G` is required on 128 GB; `--max-context 176K` is the tested limit; `--disable-ane --idle-release off --max-cache-disk 0` reproduce the benchmark; leave `--draft-model` unset for native MTP; add `--language-only` for text only. With the GGUFs' own template, `high` renders as `xhigh`. After one online install, `--offline` (or `HF_HUB_OFFLINE=1`) starts without contacting the Hub. The v2 recipe (#448 on #447 at `6db6e5ee`, `--ple-ssd-cache`) is superseded.

## Memory and storage

| Variant | Download GB | Peak after load, GiB | Sweep peak, GiB | Weights loaded |
|---|---:|---:|---:|---:|
| Quality-PLE8 | 156.502 | 97.488 | 100.799 | 41.4 s |
| Quality-PLE5 | 140.502 | E2E table | not run | E2E table |
| Quality-PLE4 | 130.902 | E2E table | not run | E2E table |
| Balanced-PLE8 | 154.640 | 95.437 | 100.739 | 36.5 s |
| Balanced-PLE5 | 138.640 | E2E table | not run | E2E table |
| Balanced-PLE4 | 129.040 | E2E table | not run | E2E table |
| Max-PLE8 | 157.249 | 98.196 | 101.625 | 36.2 s |
| Max-PLE5 | 141.249 | E2E table | not run | E2E table |
| UD-Q4_K_XL-PLE8 | 140.629 | 82.376 | 101.254 | 12.9 s |

M5 Max 128 GB, build and settings above, text-only, 176K context, 2 GiB PLE cache, 110,592 MB wired limit. *Download* = target shards + 2.787 GB MTP head + 0.908 GB projector (target and resident sizes are in Variants). Peaks are Splash Metal allocation high-water marks after load plus a 64-token warmup, and over the roughly one-hour agent sweep; not process RSS. Whole-machine memory peaked at 124.2–124.6 GiB during the PLE8 sweeps, with no swap growth over the ~1.0 GiB baseline. Load times are from server logs with an uncontrolled OS file cache. Unswept rows are covered by the release E2E table.

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

**Notes.** The lenient re-grade changes only F1's format rule (source-reference lines after `FINAL ANSWER:`); it is post-hoc, and strict stays primary. F3, an adapted FAB write-up judged separately and not in /16: all four scored 3/5, because the deterministic signed-figure gate rejected two criteria that the Sonnet 5.5 judge passed; UD's attempt also used the whole 131,072-token working context. B1/B2 failed for every model, with tool results shown to the model as harness render errors. Decode uses completed requests with ≥500 output tokens (50/52/53/61); Metal peak is Splash's allocation high-water mark, not RSS; MTP counters are proposal acceptance, not speedup.

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
| Balanced-PLE4 | 41.082 | 96.618 | 75.714 | 5/5 | 300 | 63.153 | PASS |
| Balanced-PLE5 | 41.071 | 96.619 | 68.831 | 5/5 | 300 | 65.962 | PASS |
| Balanced-PLE8 | 41.074 | 96.621 | 75.714 | 5/5 | 300 | 66.475 | PASS |
| Max-PLE5 | 41.065 | 99.378 | 68.831 | 5/5 | 300 | 69.008 | PASS |
| Max-PLE8 | 41.067 | 99.379 | 68.831 | 5/5 | 300 | 69.231 | PASS |
| Quality-PLE4 | 45.071 | 98.669 | 68.831 | 5/5 | 300 | 66.075 | PASS |
| Quality-PLE5 | 45.071 | 98.670 | 68.831 | 5/5 | 300 | 72.368 | PASS |
| Quality-PLE8 | 45.650 | 98.671 | 61.905 | 5/5 | 300 | 65.964 | PASS |
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

PLE5 keeps its PLE8 sibling's body with a **Q5_1 table (38.4 GB, 6.0 bpw)**. KLD and top-1 are measured for all three PLE5 variants (Quality 0.013128 / 95.850%, Balanced 0.013727 / 95.754%, Max 0.012848 / 95.863%). None of them has agent-task results. Native resident GB excludes the streamed table; Splash target GB and KV capacity come from runtime status on earlier builds (PLE8 from the SVG stack, † from the earlier release build), are not matched memory comparisons, and KV capacity is not a tested single-request context length. Details are on each model card and in [MEASUREMENTS.md](MEASUREMENTS.md#memory-and-files).

<a id="future-variants"></a>
**Ultra256-PLE8:** a historical Q8-focused larger-memory archive is recorded, but no validated public release is offered. **Max-PLE4:** not released. **Lite:** not released (missing body shards). Quality-H6 is an experimental control, outside this release family.

## Limitations and known issues

- **Runtime.** The released Splash 1.3.2 package cannot load Qwen3.8-Flash-Next; use the `fa842cb5` build above. Installing from the Hub with `--model OWNER/REPO:VARIANT` was not exercised before publication; the release E2E loads the same files from a local assembly.
- **Behaviour.** With the GGUFs' own template, `high` renders as `xhigh`. In FinQA, three of the four PLE8 variants put source references after a required `FINAL ANSWER:` line. Long tool-heavy sessions can fill the 131,072-token working context (UD's F3 attempt did). Batched kernels make even greedy outputs depend on batch composition and cache state.
- **Coverage.** One 128 GB M5 Max with a raised wired limit; 16 single-trial agent tasks with a common BFCL harness defect. Not validated beyond the text-only sweep and the six-prompt smoke: image/PDF correctness, streaming and Responses API details, cancellation and cache behaviour, beyond-176K context, long-context or code-text KLD. Each model card lists its own gaps.
- **PLE5** has no task benchmarks (including SVG and coding). All three PLE5 variants now have measured KLD: Quality 0.013128, Balanced 0.013727, Max 0.012848. UD+Q5_1 measured city +2.9%, approximately equal to UD+Q8, with prose/TTFT within noise; those UD-body observations do not establish speed for the custom PLE5 bodies.
- **Family gaps.** The custom bodies lack matched speed, TTFT and 128K/170K validation on the recommended build. Max also lacks matched earlier-stack speed/prefill and 13-task coding, Balanced lacks that earlier coding suite, and the PLE4 siblings lack task reruns. UD+Q8 KLD is measured (0.021037); its original-table runtime results are identified in the UD card.
- **Hub panel.** The GGUF panel lists the body label (`Q5_K_XL` or `UD-Q4_K_XL`), the MTP head (`Q8_0`) and the BF16 projector. Sizes there are per file group; the streamed PLE table and resident memory are in the tables above.

## Changelog: v2 → v3

- Custom-body shard files renamed from `-Mixed-` to `-Q5_K_XL-`, so the Hub's GGUF panel classifies them. Contents and SHA-256 are unchanged, and `--model REPO:FOLDER` works as before.
- The four per-model repositories are public under **Vardaan97**. Their model files are byte-identical to v2. This repository keeps its existing files.
- New at-a-glance summary, pick guide, copy-paste run recipe, family memory and storage table, and known issues.
- The run recipe moves to PR #448 at `fa842cb5` on the upstream #449–#468 stack (`--ple-memory`), the build used for the agent sweep and the release E2E. The v2 pin `6db6e5ee` on #447 (`--ple-ssd-cache`) is superseded.
- Added the scored nine-model agent sweep (strict and lenient, per task), the separate F3 judge result, the 1M-context note (#469 build, original UD only) and the release E2E smoke table.

## Changelog: v1 → v2

Quality-PLE8 now has its own completed neutral KLD run. Max-PLE8 passes the prior two-paired-se gate against Balanced-PLE8. Both are eligible; neither has been uploaded as part of preparing this card. See their destination cards for the publication decision and remaining measurement gaps. The recommended runtime is [Splash #448](https://github.com/incoai/splash/pull/448), stacked on [#447](https://github.com/incoai/splash/pull/447): native MTP from `mtp/` without `--draft-model`, streamed PLE, draft vocabulary and an opt-in GDN checkpoint grid.

Historical numerical results remain in [MEASUREMENTS.md](MEASUREMENTS.md) and [kld/](kld/). New runtime speed measurements used UD-Q4_K_XL and must not be attributed to the custom family. Full custom-variant long-context and new-runtime validation remain pending.

## Credits

- **Vardaan97**: quantization recipes and measurements; Splash [#410](https://github.com/incoai/splash/pull/410) (streamed PLE table), [#423](https://github.com/incoai/splash/pull/423) (hyper-normalization kernels), [#429](https://github.com/incoai/splash/pull/429) (draft vocabulary), and [#448](https://github.com/incoai/splash/pull/448) (native MTP on the new runtime).
- **Qwen**: the base model and license.
- **unsloth**: source BF16 GGUF conversion, importance matrix, stock shared Q8_0 MTP head and BF16 projector, redistributed unchanged. Inherited `general.quantized_by`, `general.repo_url` and `general.tags` still name unsloth.
- **AtomicChat**: [BF16 reference logits](https://huggingface.co/datasets/AtomicChat/Qwen3.8-Flash-Next-Uncensored-GGUF-metrics) and [evaluation text](https://huggingface.co/datasets/AtomicChat/calib-corpora).
- **janhilgard**: Flash-Next and native MTP support in [Splash #253](https://github.com/incoai/splash/pull/253).
- **jianc99**: the new runtime and block-sparse attention in [Splash #447](https://github.com/incoai/splash/pull/447).
- The Splash maintainers.

## Licence

These files are derivatives of Qwen3.8-Flash-Next under **Qwen Community License 1.0**. Read [LICENSE](LICENSE) for the full terms; it is identical to the base repository's license.
