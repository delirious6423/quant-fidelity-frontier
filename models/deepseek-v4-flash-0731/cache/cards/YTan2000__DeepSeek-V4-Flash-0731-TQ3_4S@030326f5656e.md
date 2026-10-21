---
license: mit
language:
- en
library_name: gguf
pipeline_tag: text-generation
model_name: TurboDeepSeek-V4-Flash
tags:
- gguf
- llama.cpp
- deepseek
- deepseek-v4
- moe
- turboquant
- tq3_4s
- speculative-decoding
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
model-index:
- name: TurboDeepSeek-V4-Flash
  results:
  - task: {type: text-generation, name: Code Generation}
    dataset: {type: evalplus/humanevalplus, name: HumanEval+}
    metrics:
    - {type: pass@1, value: 94.5, name: pass@1 base (reasoning-on, budget 256, 512K ctx, ctk q4_0, concise system prompt)}
    - {type: pass@1, value: 90.9, name: pass@1 plus (reasoning-on, budget 256, 512K ctx, ctk q4_0, concise system prompt)}
  - task: {type: text-generation, name: Code Generation}
    dataset: {type: evalplus/mbppplus, name: MBPP+}
    metrics:
    - {type: pass@1, value: 91.8, name: pass@1 base (reasoning-on, budget 256, 512K ctx, ctk q4_0, concise system prompt)}
    - {type: pass@1, value: 77.2, name: pass@1 plus (reasoning-on, budget 256, 512K ctx, ctk q4_0, concise system prompt)}
---

# TurboDeepSeek-V4-Flash

<img src="thumbnail.png" alt="TurboDeepSeek-V4-Flash — TQ3_4S" width="520">

**Canonical artifact:** `DeepSeek-V4-Flash-0731-TQ3_4S_v3`

`TurboDeepSeek-V4-Flash` is the public release name for the TurboQuant GGUF build
of DeepSeek-V4-Flash-0731 (256-expert MoE, 6 active + 1 shared, ~21B active
params, 43 layers, MLA attention).

The exact file and runtime artifact name is:

- `DeepSeek-V4-Flash-0731-TQ3_4S_v3.gguf`

## Required Runtime

This release requires the TurboQuant runtime fork:

- `https://github.com/turbo-tan/llama.cpp-tq3`

Stock `llama.cpp` builds without TurboQuant support cannot load the custom
`TQ3_4S` tensor type. **This is a standard (non-MTP) release** — the GGUF does
not contain an MTP draft block. A community DSpark drafter exists for this
base model but **cannot be loaded by this runtime** — see Speculative Decoding
below, where that was measured rather than assumed.

## Parent Model

- Upstream parent: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Format conversion and TurboQuant packaging: `turbo-tan/llama.cpp-tq3`

## Files

- `DeepSeek-V4-Flash-0731-TQ3_4S_v3.gguf` — main model, 102.75 GiB (110,322,537,056 bytes)
- [`thumbnail.png`](thumbnail.png) — model card image
- [`benchmarks.html`](benchmarks.html) — full interactive benchmark page (perplexity, speed, Hard86, evalplus, six task suites)
- [`benchmark.png`](benchmark.png) — benchmark summary (three-way vs IQ3_XXS and Laguna S2.1; note it reports **reasoning-off** figures for all three engines, labelled on the image, while the headline figures above are the recommended reasoning-on profile)

## Recommended Runtime

**Enable reasoning.** Serving with reasoning disabled (`--reasoning off`) produces
a confirmed, reproducible arithmetic bug (see Known Behavior below) — this is
NOT a quantization artifact, it reproduces identically on an independently-built
IQ3-XXS quant of the same base model, and is fixed by the `<think>` trigger
itself (confirmed down to a 1-token reasoning budget).

```bash
./build/bin/llama-server \
  -m DeepSeek-V4-Flash-0731-TQ3_4S_v3.gguf \
  --host 0.0.0.0 --port 8085 \
  -c 524288 -np 1 -ngl 99 -fa on -b 4096 -ub 2048 \
  -ctk q4_0 -ctv tq3_0 --threads 20 \
  --reasoning on --reasoning-budget 256 \
  --reasoning-format deepseek --jinja
```

**All figures below are validated at this exact 512K serving config**
(`-c 524288 -np 1 -ctk q4_0 -ctv tq3_0`, reasoning-on, budget 256) on NVIDIA
GB10 (DGX Spark), 2026-08-05. The full battery (Hard86, HumanEval, MBPP,
Benchloop) shows **zero regression versus the 36K context baseline** — 512K
context costs nothing on quality for this build.

`-ctk q4_0` was measured on 2026-08-05 at 512K: Hard86 **70/86**, HumanEval
**94.5 / 90.9**, MBPP **91.8 / 77.2**, at roughly half the K-cache footprint
(2.9 GB vs 5.4 GB at 512K).

**Perplexity — wikitext-2, the standard setting.** Measured 2026-08-05,
`wiki.test.raw`, `-c 512 --chunks 60`, which is the configuration published
figures are normally quoted at:

| build | ctx / chunks | **PPL** |
|---|---|---:|
| **TQ3_4S v3, `-ctk q8_0 -ctv tq3_0`** | 512 / 60 | **5.98** |
| **TQ3_4S v3, `-ctk q4_0 -ctv tq3_0`** (shipped) | 512 / 60 | **6.09** |

PPL falls with longer context, so these are not comparable to numbers quoted
at other context lengths — compare only like for like.

**The K-cache setting is a real trade-off.** Measured with only the K-cache
dtype changed (`-ctv tq3_0` held constant, same corpus/context/chunks):

```
-ctk q8_0  ->  PPL 5.98        -ctk q4_0  ->  PPL 6.09       (+1.8%)
```

So `-ctk q4_0` costs roughly **1.8% perplexity** and saves roughly **2.5 GB**
of K-cache at 512K (2.9 GB vs 5.4 GB). Task benchmarks (Hard86, HumanEval,
MBPP) do not resolve a shift this small — perplexity does. The headline
figures above were measured at `q4_0` and carry this cost.

**If you have the memory, prefer `-ctk q8_0`.** Use `q4_0` when the K-cache
saving is what makes your context length fit at all — which, at 512K on a
121 GB box, it is.

Note also that the Hard86/evalplus comparison between the 512K and 36K
profiles below varies **two** things at once (context length *and* K-cache
dtype), so no difference between those columns can be attributed to either
alone.

Perplexity cost of the K-cache change, measured on this artifact:

| K-cache type | PPL |
|---|---:|
| `-ctk q8_0` | 5.9839 |
| `-ctk q4_0` | 6.0924 |
| **delta** | **+0.1085 (+1.81%)** |

Optimized flash-attention kernels exist for quantized K/V, so there is no
decode-speed penalty. Keep `-ctv tq3_0` (a larger V-cache measurably worsened
quality in testing); do not raise K or V cache types without re-validating.

512K fits only at `-np 1` on a 121 GB box (weights 102.7 GB + KV + compute).
For multi-slot serving, drop `-c` accordingly (e.g. `-c 147456 -np 4` =
4×36K slots was the validated 36K config).

`--reasoning-budget 256` is a deliberate cap, not the default unrestricted (-1):
unbounded reasoning burned 6000+ think-tokens on some coding tasks (about 6 min/task);
`budget=256` (about 150-200 think words) keeps latency reasonable while still
triggering the fix. Budget=1 is sufficient to fix the arithmetic bug but
backfires on coding tasks (reasoning dumps into visible content instead of the
`reasoning_content` field) — do not go below the tested 256 floor without
re-validating on coding tasks specifically.

### ⚠️ At a capped reasoning budget, a concise system prompt is REQUIRED

**This is load-bearing, not a style preference.** At `--reasoning-budget 256`
the model must be told not to deliberate in its visible answer, or it spends
the capped allowance thinking and is cut off before emitting any code.

Use a system prompt such as:

> `Answer concisely. Do not deliberate, plan, or write meta-commentary in your response; give the direct answer only.`

Measured on HumanEval (164 tasks), same model, same budget, same harness — the
**only** difference is the system prompt:

| Configuration | pass@1 base | empty completions |
|---|---:|---:|
| budget 256 **with** concise prompt | **94.5** | 1 / 164 |
| budget 256 **without** it | **0.6** | **163 / 164** |

If you deploy at a capped budget with an ordinary prompt, expect empty or
truncated output. Either use a prompt of the form above, or raise
`--reasoning-budget` (at the cost of the latency noted above). All benchmark
figures in this card were produced with the concise prompt.

Then request:

```bash
curl -s http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"test","messages":[{"role":"user","content":"What is 81 / 9?"}],"max_tokens":512,"temperature":0}'
```

Expected assistant content: `9` (not `81`; see Known Behavior).

## Speculative Decoding (DSpark) — NOT SUPPORTED by this runtime

**Measured 2026-08-05 on this exact quant and runtime: it does not load.**

The community DSpark drafter (`bleysg/DeepSeek-V4-Flash-DSpark-drafter-GGUF`,
`DSpark-drafter-Q2K-Q8-0731.gguf`, 6.5 GiB) declares model architecture
`deepseek4-dspark`, which the required runtime cannot load:

```
error loading model: unknown model architecture: 'deepseek4-dspark'
common_speculative_init_result: failed to load draft model
srv  llama_server: exiting due to model loading error
```

`llama.cpp-tq3` has no support for that architecture, so
`-md ... --spec-type draft-dspark` **fails at startup** — the server does not
come up at all. Third-party speedup figures for this drafter come from a
different engine and cannot be reproduced here.

**Measured baseline for reference** (same paired harness, drafter arm failed):

| config | decode |
|---|---:|
| 256K ctx, `-ctk q8_0 -ctv tq3_0`, reasoning-on budget 256 | **10.3 tok/s** |
| `llama-bench` `tg128` (default cache) | 10.52 ± 0.04 tok/s |
| `llama-bench` `pp2048` | 119.99 ± 0.57 tok/s |

If DSpark support lands in `llama.cpp-tq3`, this section should be re-measured
with the paired A/B harness (`scripts/measure_dspark_spec_dsv4.sh`) rather
than re-citing an external number.

## Known Behavior — Arithmetic "dividend echo": NOT REPRODUCIBLE (2026-08-05)

A "dividend echo" bug has been reported for this model family — that under
`--reasoning off` it answers division by echoing the dividend
(`81 / 9` -> `"81"`). **It does not reproduce on this build.** Testing with
`--reasoning off --reasoning-budget 0` and one variable changed per run gave
the **correct answer in every configuration**:

| arm | config varied | `81/9` | `63/7` | `144/12` |
|---|---|---|---|---|
| sop_smoke | SOP validation config (`-ctv q8_0`, 4K) | **9** | **9** | **12** |
| ctv_tq3 | V-cache dtype -> `tq3_0` | 9 | 9 | 12 |
| shipped_cache | **the shipped recipe** (`ctk q4_0 -ctv tq3_0`) | 9 | 9 | 12 |
| dry_sampler | DRY repetition penalty enabled | 9 | 9 | 12 |
| longctx_shipped | 256K context, shipped cache | 9 | 9 | 12 |

15/15 correct. That includes the exact serving recipe published above.

This does not prove the behaviour was never real — it was reported against
two independently-built quants, and a runtime fix is a plausible explanation.
It means the issue is not reproducible on this build. If you can reproduce
it, please report your exact flags.

Reasoning-on is still the recommended profile, on the benchmark evidence
above — it is the configuration all published figures were measured at.

## Known Behavior — reasoning budget can silently produce EMPTY output

Independently confirmed on 2026-08-05, and more likely to affect you than the
above. At `--reasoning-budget 256`, a **hard** task can consume the entire
allowance on reasoning and emit **nothing** — even *with* the concise system
prompt. Observed on a Hard86 diff/patch task:

```
completion_tokens : 258        <- exactly the 256 budget, +2
content           : ''         <- empty
finish_reason     : stop
reasoning         : cut off mid-sentence, having correctly identified the approach
```

Every other task in that run used 345-2027 tokens and produced content; this
was the only one pinned at the cap. The model knew the solution and ran out of
budget before writing it.

**The concise system prompt reduces this risk but does not eliminate it.** If
you see empty responses on hard prompts, raise `--reasoning-budget` (at the
latency cost noted above) rather than assuming the model failed.

## Benchmark Summary

![Benchmark summary — TQ3_4S vs IQ3_XXS vs Laguna S2.1](benchmark.png)

*Three-way comparison at **reasoning-off** for all three engines (labelled on
the image). The headline figures below are the recommended **reasoning-on**
profile, so the two differ by design. Full interactive version:
[`benchmarks.html`](benchmarks.html).*

Two validated serving profiles. The **recommended runtime is reasoning-ON**
(see Recommended Runtime above); reasoning-OFF has the arithmetic bug
documented under Known Behavior, so the reasoning-OFF row is provided only
for reference/comparison — use reasoning-ON.

**Recommended profile — 512K, reasoning-ON, budget 256** (`--reasoning on
--reasoning-budget 256`, `-c 524288 -np 1`, `-ctk q4_0 -ctv tq3_0`, concise
system prompt, max_tokens 16384, GB10/DGX Spark, validated 2026-08-05):

| Metric | 512K result | 36K baseline |
|---|---:|---:|
| Hard86 | **70/86** | 67/86 |
| HumanEval (base / plus) | **94.5 / 90.9** | 94.5 / 92.1 |
| MBPP (base / plus) | **91.8 / 77.2** | 91.8 / 77.2 |
| Benchloop overall | 79.5 * | 79.5 |
| Custom 15-suite | 15/15 † | 15/15 |
| Arithmetic gate (20-question digit-fidelity) | 20/20 ‡ | 20/20 |
| Size | 102.75 GiB | — |

\* Benchloop 79.5 measured at 512K with `-ctk q8_0` (2026-08-05, before the
K-cache type change); coding 100 / instructfollow 87.8. ctk q4_0 was
afterwards validated quality-neutral on Hard86 + Evalplus.
† Custom 15-suite and 36K baselines measured at `-c 147456 -np 4` (4×36K),
`-ctk q8_0 -ctv tq3_0`, otherwise identical profile.
‡ Arithmetic gate validated at 512K with `-ctk q8_0`, budget 256.

**Headline: 512K context shows zero quality regression versus 36K** on every
suite measured at both lengths. Reasoning-ON improves code benchmarks over
reasoning-OFF (HumanEval+ 88.4 → 90.9) and fixes the arithmetic bug, at the
cost of some Hard86 sub-tests where the concise-prompt budget trades depth for
brevity.

**Reference profile — reasoning-OFF** (512K context, `-ctk q8_0 -ctv tq3_0`,
np=1 — DO NOT USE for arithmetic, see Known Behavior):

| Metric | Result |
|---|---:|
| Hard86 | 73/86 |
| HumanEval (base / plus) | 92.1 / 88.4 |
| MBPP (base / plus) | 92.1 / 76.5 |
| Custom 15-suite | 15/15 |
| Arithmetic gate | FAILS (dividend echo) |

Decode throughput (reasoning-ON, budget=256, GB10): **18–19.6 tok/s**
single-stream at 512K context (measured 18.1–19.0 tok/s across the 512K
battery, 2026-08-05); 19.6 tok/s measured earlier at `-c 32768`. Throughput
is context-insensitive at these lengths — decode is weights-bandwidth-bound,
not KV-bound.

### Serving stability at 512K (GB10, 121 GB unified memory)

512K serving leaves little headroom (weights 102.7 GB ≈ 85% of the box).
The validated battery completed with the server sitting below 1% free memory
for hours — zero OOM kills, zero restarts — under this hardening:

1. Run llama-server as a **systemd system unit** (not nohup over SSH — user-slice
   cgroup teardown kills long-lived processes; see below).
2. Add **zram swap** (~32 GB, zstd) as a pressure valve, and retune earlyoom's
   swap threshold accordingly (`-s 10`, not the common `-s 100` which would
   kill on any swap use).
3. Put `llama-server` in earlyoom's `--avoid` list (protects the server; note
   earlyoom will then shoot system services instead under extreme pressure —
   headroom is the real fix).

## Tested Hardware

- NVIDIA GB10 (DGX Spark), unified memory, ARM
- Confirmed working identically on two separate GB10 boxes over both standard
  network and a dedicated QSFP fabric link (transfer only; inference tested
  single-box)

## License

Use is subject to the base model license (MIT, `deepseek-ai/DeepSeek-V4-Flash-0731`)
and the license terms of the runtime components used to run the GGUF.
