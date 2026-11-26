---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
inference: false
tags:
- htr
- halo-trellis
- quantized
- strix-halo
- gfx1151
- custom-runtime
---

![Qwen3.8 Flash Next — Halo Trellis](media/hero.png)

# Qwen3.8-Flash-Next-HTR

**HTR release v0.2.7 — 2.32× faster cold boot, +7.3% to +21.5% faster decode, and unified AMD Strix Halo runtime (Qwen3.8 + ERNIE-5.5).** [Runtime release](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.7) · [Release history](docs/RELEASES.md).

HawgAuto's Halo Trellis quantization of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), built for **AMD Strix Halo / Radeon8060S** with the HTR-aware vLLM runtime. This is a serving update: **all checkpoint weights, tokenizer/configuration files and quantization are unchanged**.

## Recommended model — Htr5Core

**[Htr5Core](Htr5Core/) is the recommended inference model.** The quick start downloads this complete variant. **[Bf16Core](Bf16Core/)** is a complete research reference restoring the BF16 core; its GPU serving, performance and maximum context remain separately unqualified. [Variant details](docs/VARIANTS.md). Download one variant, not the whole repository; the root checkpoint remains a compatibility copy of Htr5Core.

## What's new in v0.2.7

v0.2.7 adds **2.32× faster cold empty-workspace startup**, **+7.3% to +21.5% faster decode** across BF16 and INT8 KV on AMD Strix Halo (`gfx1151`), and unifies the container with [ERNIE-5.5-Flash-Preview-HTR](https://huggingface.co/hstolte/ERNIE-5.5-Flash-Preview-HTR) (strictly auto-gated so Qwen3.8 paths remain untouched). **All Qwen3.8 weights, tokenizer/config files, and sampling defaults are unchanged — just pull the new container image.**

- **2.32× faster cold empty-workspace boot:** `388.2 s → 167.3 s` (**−221 s**), warm BF16 boot **1.24× faster** (`114.7 s → 92.6 s`), and KV8 config-change boot **1.52× faster** (`309.8 s → 204.3 s`).
- **Faster decode on both BF16 & INT8 KV (`int8_per_token_head`, MTP-3):**
  - KV8 greedy `c=1`: **`45.16 → 48.45 tok/s` (+7.3%)**
  - KV8 13K warm decode: **`44.55 → 54.15 tok/s` (+21.5%)** · 13K cold prefill **`14.30 s → 12.08 s` (1.18× faster, `1,077.8 tok/s`)**
  - BF16 `c=4` stochastic: **`63.15 / 95.48 → 69.96 / 107.49 tok/s` (+10.8% / +12.6%)**
  - Mid-inference Triton JIT warnings: **`0`**.

### Verified zero-overlay E2E results (`v0.2.7` vs `v0.2.6`, INT8 KV, MTP-3)

| Workload | `v0.2.6` KV8 | **`v0.2.7` KV8 Zero-Overlay** | **Delta** |
|---|---:|---:|---:|
| Config-change boot | `309.8 s` | **`204.3 s`** | **1.52× faster (−105.5 s)** |
| Greedy `c=1` decode (`128 tok`) | `45.16 tok/s` | **`48.45 tok/s`** | **+7.3%** |
| `13K` cold prefill / decode | `14.30 s (910.5)` / `54.50` | **`12.08 s (1,077.8)` / `60.11`** | **1.18× / +10.3%** |
| `13K` warm prefill / decode | `11.44 s (1,137.7)` / `44.55` | **`11.69 s (1,113.6)` / `54.15`** | **−2.1% / +21.5%** |
| `c=4 × 4K` (`wall` / `sum_dec`) | `21.77 s` / `61.65 tok/s` | **`21.46 s` / `62.82 tok/s`** | **1.01× / +1.9%** |
| `c=8` (`wall` / `agg` / `sum_dec`) | `15.51 s` / `66.01` / `90.39` | **`15.23 s` / `67.25` / `91.20`** | **1.02× / +1.9% / +0.9%** |

- **Full evidence:** [v0.2.7 release notes](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.7) · [V027-NATIVE-MANIFEST.json](https://github.com/HawgAuto/vllm-htr/blob/htr-v0.2.7/htr_runtime/V027-NATIVE-MANIFEST.json).

## Previous release: v0.2.6

v0.2.6 moves the remaining serving hot-path kernels off Triton JIT onto **20 new zero-spill ahead-of-time (AOT) HIP kernels** compiled specifically for AMD Strix Halo (`gfx1151`), and upgrades **3 decode extensions** with RDNA 3.5 ISA optimizations. **All model weights, tokenizer/config files, and sampling defaults are unchanged — just pull the new container image.**

- **2× faster long-context prefill:** 13K-token warm prefill drops from **23.58 s → 11.44 s** (**551.9 → 1,137.7 tok/s**), and cold 13K prefill reaches **14.30 s** (**910.5 tok/s**, **1.65×** faster).
- **Zero Triton JIT pauses during serving:** Every hot-path prefill and decode op (QSA attention, GDN, `causal_conv1d`, W4A16 MoE, HyperConnection, dense prefill, PLE transpose, Gumbel sampler) now runs from prebuilt `.hsaco`/`.so` binaries (**`0`** mid-inference Triton JIT warnings).
- **Full-INT8 WMMA attention prefill (with instant BF16 fallback):** Uses RDNA 3.5 `v_wmma_i32_16x16x16_iu8` (`qsa_tau_v5_kv8_i8`) for **1.66–1.70× faster** QSA prefill (**−865 ms** on 13K warm prompts, **−1.83 s** on `c=4×4K`, **−1.45 s** on `c=8`) while maintaining high numerical fidelity (`cos = 0.999994`, `0 NaNs`, exact 200/200-char greedy & 12K match, `63.45%` MTP-3 spec acceptance). Pass `-e HTR_QSA_TAU_V5_I8=0` to switch back to BF16 WMMA at any time.
- **Faster multi-request concurrency & strided-cache fix:** `c=8` wall time improves **19.03 s → 15.51 s** (**1.23× faster**, `66.01 tok/s` wall aggregate) and `c=4 × 4K` long-context concurrency improves **23.82 s → 21.77 s** (**+7.7% sum decode** to `61.65 tok/s`), including a bit-exact fix for strided-1D `causal_conv1d` cache indexing.
- **RDNA 3.5 (`gfx1151`) ISA decode upgrades:** Unlocks hidden dual-issue `v_dual_dot2acc_f32_bf16` (4 BF16 FMAs/lane/cycle), single-cycle `v_perm_b32` BF16 packing, and bank-staggered VOPD pairing across `htr_bf16_gemv.so`, `htr_moe_k6_shx.so`, and `htr_kernels.so`.

### Verified zero-overlay E2E results (`v0.2.6` vs `v0.2.5`)

Exclusive AMD Strix Halo (`gfx1151`), Htr5Core, MTP3, `max-model-len=32768`, `--kv-cache-dtype int8_per_token_head` (`-e HTR_KV8_ENABLE=1`):

| Workload | `v0.2.5` KV8 | **`v0.2.6` Zero-Overlay Gate** | **Change vs v0.2.5** |
|---|---:|---:|---:|
| `13K` warm prefill (`13,018 tok`) | `551.9 tok/s` (`23.58 s`) | **`1,137.7 tok/s`** (`11.44 s`) | **2.06× faster** |
| `13K` cold prefill (`13,017 tok`) | `551.9 tok/s` (`23.58 s`) | **`910.5 tok/s`** (`14.30 s`) | **1.65× faster** |
| Greedy `c=1` decode (`128 tok`) | `45.05 tok/s` | **`45.16 tok/s`** | **+0.2%** |
| `c=4 × 4K` (`16,702` ctx) — wall / sum decode | `23.82 s` / `57.25 tok/s` | **`21.77 s`** / **`61.65 tok/s`** | **1.09× wall / +7.7% dec** |
| `c=8` high-concurrency — wall / agg (`8×128`) | `19.03 s` / `53.80 tok/s` | **`15.51 s`** / **`66.01 tok/s`** | **1.23× wall** |
| Mid-inference Triton JIT warnings | multiple | **`0`** | **eliminated** |

- **Full evidence:** [v0.2.6 release notes](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.6) · [Qualification JSON](evidence/v0.2.6-release-qualification.json).

## Previous release: v0.2.5

- **Full clean-slate `gfx1151` Wave32 stack (18 new native binaries + 38 runtime updates, default-on):** K6 routed-MoE `rows=1..16` with zero-realloc shared-expert folding (`T<=4`), HPF4 MoE prefill, nontemporal `global_load_b128` K4 `lm_head`, dense WMMA, fused GDN MTP spec decode, `hiphc_v2` fused HyperConnection, native QSA decode union for **BF16 / `int8_per_token_head` / `int4_per_token_head` KV**, `ple_gather_v25` + native PLE short-conv/gate-norm, MTP DDN v2/WMMA + BF16 GEMV, and native Q38 filter + Gumbel + sparse rejection sampling.
- **BF16 KV served E2E A/B vs `v0.2.4` (`max-num-seqs=4`, `32K` ctx, MTP3, `4 GiB` KV):**
  - Greedy `c=1`: `42.21 → 44.60 tok/s` (**`+5.7%` tok/s / `+7.6%` step speed**, `55.7 → 51.8 ms/step`)
  - `4K` decode (`5,812` ctx): `54.59 → 59.56 tok/s` (**`+9.1%`**); `4K` prefill: `977.7 → 983.9 tok/s`
  - `c=4` concurrent (`t=0.7`, `4×256`): `71.33 → 100.50 tok/s` sum decode (**`+40.9%`**), `51.44 → 69.99 tok/s` wall agg (**`+36.1%`**, `19.91s → 14.63s`)
  - `c=4` greedy (`t=0.0`, `4×256`): `80.37 → 109.90 tok/s` sum decode (**`+36.7%`**, `15.59s → 11.68s`)
  - MTP3 acceptance: `69.96% → 73.80%` (`3.10 → 3.21 tok/step`)
- **INT8 KV long-context & high-concurrency A/B vs `v0.2.4` (`int8_per_token_head` + `V5-Q4`, `max-num-seqs=8`, `4 GiB` KV):**
  - Greedy `c=1`: `43.18 → 45.05 tok/s` (**`+4.3%`**)
  - `13K` prefill (`13,017 tok`): `409.6 → 551.9 tok/s` (**`+34.7%`**, `31.78s → 23.58s`)
  - `13K` decode (`13,017` ctx): `43.39 → 63.92 tok/s` (**`+47.3%`**)
  - `c=4 × 4K` (`16,702` ctx): `33.99s → 23.82s` wall (**`+42.7%`**, `15.06 → 21.49 tok/s` wall agg)
  - `c=8` concurrent (`8×128`): `103.96 → 132.18 tok/s` sum decode (**`+27.1%`**), `44.76 → 53.80 tok/s` wall agg (**`+20.2%`**)
- **Full evidence:** [v0.2.5 release notes](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.5) · [Qualification JSON](evidence/v0.2.5-release-qualification.json).

## Previous release: v0.2.4

- Default-on **K6 routed-MoE decode at 2–4 rows**, plus **dense WMMA at eligible 9–15-row projections**. Existing shape fallbacks and parity guards remain.
- **Single-stream improvement in the selected-stack screen:** steady aggregate output 48.05 → 55.35 tok/s (+15.19%); mixed-prompt output/full wall 34.08 → 36.41 tok/s (+6.82%); mixed-prompt per-request decode median 48.10 → 53.83 tok/s (+11.91%). MTP3, maximum four sequences, 40,960 configured context, two mirrored visits per arm.
- **No established multi-stream or prefill win.** Steady 2/3/4-stream changes were −5.60% / +2.44% / +4.14%; the mixed-prompt changes were +1.77% / +4.46% / +0.96%. These small screens are not statistical noise bounds or universal speed promises.
- **Task screens:** HumanEval+ 159 → 162 / 164; ToolEval 125 / 138 both; Hermes20 1850 → 1950 / 2000. 262K configured context, MTP3, INT8 KV, thinking on, temperature1.0/top-p0.95/top-k20 and 2GiB optional PLE cache. These selected-stack results were not rerun on the packaged image.
- **PLE startup repair:** async row limit0 now disables that path instead of crashing; its self-check wait is bounded. Empty-workspace cold-start repair is included.
- **Unchanged weights, tokenizer/configuration and checkpoint sampling defaults.** No forced seed 42, no persistent unpacked-weight shadow, no mandatory compiler-cache download. PLE hot rows remain off by default.

Known limitation: HumanEval/157 looped on the selected stack with evaluation seed 42, reproduced twice; seeds 1–3 and 101–140 did not loop. Across 44 distinct tested seeds, candidate had one loop-triggering seed and predecessor none. This is not a population loop-rate estimate or a claim that reasoning loops are fixed. [Exact tables, limits and qualification](https://github.com/HawgAuto/vllm-htr/blob/htr-v0.2.4/htr_runtime/releases/v0.2.4/README.md). Historical throughput targets remain unmet.

## At a glance — Htr5Core

- **Parameters:** **125B with 6B activated, plus 51B n-gram embeddings and 4B MTP. Vision included.**
- **Platform:** Linux, AMD Strix Halo/gfx1151, 128 GB unified system memory and fast local SSD storage. Leave memory for model loading, runtime, KV, compilation and the OS; do not co-load another large shared-memory GPU workload.
- **Storage:** 173.759 GB complete checkpoint. Its 102.4 GB PLE tables stay on SSD; the optional hot-row cache consumes only its selected CPU-RAM budget.
- **Current qualification:** see the [versioned v0.2.6 release evidence](evidence/v0.2.6-release-qualification.json) for exact zero-overlay release-gate results and INT8 WMMA parity checks. Earlier capacity, accuracy and timing results keep their original image scope.
- **Context and concurrency history:** prior BF16 1M and quantized-KV 2M/3M/4M recipes, and earlier concurrency results, remain available below on their original pinned images. They are not new v0.2.6 capacity or performance claims.
- **Quick start:** warm the native configuration once, keep `/workspace`, then restart with the same settings to reuse compiled programs. The historical v0.2.2 native fresh/seeded pair reached readiness in 446.810 / 143.003 s; one pair is not a universal startup guarantee. New shapes may compile.

[Complete v0.2.2 configuration/timing tables](https://github.com/HawgAuto/vllm-htr/blob/main/htr_runtime/releases/v0.2.2/QUALIFICATION.md) · [Machine-readable qualification](evidence/v0.2.2-release-qualification.json).

## Required runtime

HTR is a custom format, not GGUF. Use [vLLM HTR v0.2.7](https://github.com/HawgAuto/vllm-htr/releases/tag/htr-v0.2.7), not stock Transformers/vLLM or llama.cpp. Executable source: [`ebec966358ecd0f71b8efe14ab70ef602890bf5f`](https://github.com/HawgAuto/vllm-htr/tree/ebec966358ecd0f71b8efe14ab70ef602890bf5f). Public image: `ghcr.io/hawgauto/vllm-htr@sha256:ac7d31a9fcf22da19c412f3d2db2192d163a2e0a06d9a40de7f4697542186e5b` (alias `v0.2.7`). No GitHub login, host compiler or separate launcher download is required. The release's short MTP0/MTP3 folding checks and documented-command tool round trip are recorded separately from historical speed/quality screens.

The native path inherits checkpoint sampling defaults unless overridden: temperature 0.7, top-p 0.8, top-k 20 for Htr5Core. The older maintainer-selected 1.0 / 0.95 profile remains an optional native override, not a forced default. Thinking/template settings are unchanged. [Sampling and full flag reference](docs/RUNNING.md#sampling-for-existing-clients).

## Quick start and local cache reuse

### 1. Download Htr5Core and pull the runtime

```sh
VARIANT=Htr5Core
hf download hstolte/Qwen3.8-Flash-Next-HTR \
  --revision v0.2.6 --include "$VARIANT/*" --include "meta.yaml" \
  --local-dir ./Qwen3.8-Flash-Next-HTR

docker pull ghcr.io/hawgauto/vllm-htr@sha256:ac7d31a9fcf22da19c412f3d2db2192d163a2e0a06d9a40de7f4697542186e5b
```

Already have a verified Htr5Core checkpoint? Keep it; no weight redownload is needed. The original root checkpoint is Htr5Core too. Model and runtime versions are coordinated serving releases, not new quantizations.

### 2. Start the native runtime

Use absolute paths. The workspace must be writable by the image's UID 1000. Replace `renderD128` with the AMD render node if needed. Run this in its own shell, or save it as a script:

```bash
#!/usr/bin/env bash
set -euo pipefail
IMAGE=ghcr.io/hawgauto/vllm-htr@sha256:ac7d31a9fcf22da19c412f3d2db2192d163a2e0a06d9a40de7f4697542186e5b
MODEL=${MODEL:-"$PWD/Qwen3.8-Flash-Next-HTR/Htr5Core"}
WORKSPACE=${WORKSPACE:-"$PWD/htr-cache/Htr5Core-v027"}
mkdir -p "$WORKSPACE"
exec docker run --rm --init --pull=never --name vllm-htr \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add "$(stat -c %g /dev/kfd)" \
  --group-add "$(stat -c %g /dev/dri/renderD128)" \
  --pids-limit 8192 --shm-size 16g --ulimit core=0 \
  --security-opt no-new-privileges:true \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$MODEL,dst=/model,readonly" \
  --mount "type=bind,src=$WORKSPACE,dst=/workspace" \
  "$IMAGE" /model --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 --max-num-seqs 1 \
  --max-num-batched-tokens 2048 --kv-cache-memory-bytes 2147483648
```

This launches BF16 KV, MTP3, one sequence and a2GiB KV pool. Retain the workspace: first start compiles, subsequent starts reuse matching programs. This is compilation caching, not an expanded weight cache. There is no new `--build-seed` wrapper in the native path. Do not pass old launcher-only `--cpu-check`, `--mtp`, `--context` or `--kv-cache-gib` flags to `vllm serve`.

To turn MTP off, append `--speculative-config null`. Select KV with `--kv-cache-dtype`; use `--max-model-len`, `--kv-cache-memory-bytes`, `--max-num-seqs` and `--max-num-batched-tokens` for native budgets. You can save the configuration in YAML with `--config`. For optional16MiB PLE hot-row caching, add `-e HTR_PLE_ROW_CACHE_BYTES=16777216` before the image reference. Keep different profiles/variants in their own workspace; new request shapes can compile. [Complete guide and optional source build](docs/RUNNING.md).

### 3. Connect

After readiness, use `http://127.0.0.1:8000/v1` with model ID `qwen38-htr` (or discover it from `/v1/models`):

```sh
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-htr","messages":[{"role":"user","content":"Explain how speculative decoding works."}],"max_tokens":512}'
```

Thinking consumes output tokens. Supply OpenAI-compatible `tools` normally. The bind is loopback-only; add authentication and a reverse proxy before external exposure.

## Preserved v0.2.0 performance and concurrency measurements

The following rows remain tied to their original cached-process, greedy, fixed-length v0.2.0 qualification stack. They are not new v0.2.2 timings or sampled-chat promises. The old launcher admitted MTP0 1–32 and MTP3 1–4 streams; the current native configuration remains user-controlled, with new release qualification scoped above.

| Measured performance on Radeon 8060S | MTP0 | MTP3 |
|---|---:|---:|
| **Decode at 2K input** | **33.78 tok/s** | **43.47 tok/s** |
| **Decode at 8K input** | **33.81 tok/s** | **55.28 tok/s** |
| **Prefill at 2K input** | **1,020.06 tok/s** | **1,234.60 tok/s** |
| **Prefill at 8K input** | **1,227.13 tok/s** | **1,147.94 tok/s** |
| **Concurrent streams supported by the launcher** | **1–32** | **1–4** |
| **4-stream decode: all four streams active** | **68.90 tok/s total** | **52.42 tok/s total** |
| Mean per stream during that decode window | 17.22 tok/s | 13.10 tok/s |
| 4-stream decode: whole wave, including overlapping prefill stalls | 40.18 tok/s total | 26.40 tok/s total |
| 4-stream output / full request-wave wall time | 22.28 tok/s total | 18.80 tok/s total |

**How to read these speeds:** single-request rows use 2,048 / 8,192 input tokens and 128 output tokens. The four-stream rows use a mixed 512 / 2,048 / 8,192 / 8,192-token workload. “All four streams active” measures only simultaneous decoding; the whole-wave and full-wall rows show the effect of prompt processing and scheduling. These are cached-process, greedy fixed-length measurements from the v0.2.0 qualification stack, before its final self-check scratch-lifetime cleanup—not sampled-chat promises. [Exact settings, fresh/cached results and observed ranges](docs/SERVING-PERFORMANCE.md).

## Preserved v0.2.1 context and KV-accuracy profiles

Use the [pinned v0.2.1 guide](docs/RUNNING-v0.2.1.md) for the old profile-launcher commands; do not apply those aliases to the new native entrypoint. The tables below preserve their original v0.2.1 image/measurement scope, including the retained v0.2.0 BF16 baseline.

### Choose your context profile

| KV cache | MTP | Max context | YaRN | KV pool |
|---|---:|---:|---:|---:|
| BF16 KV ([retained v0.2.0 profile](docs/RUNNING-v0.2.0.md)) | 0 / 3 | **1,000,000** | ×4 | 27 GiB |
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 0 | **2,000,000** | ×8 | 27.75 GiB |
| Scaled 8-bit per-token/head KV (`int8_per_token_head`) | 3 | **2,000,000** | ×8 | 27.75 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 0 | **4,000,000** | ×16 | 26.875 GiB |
| Rotated 4-bit per-token/head KV (`int4_per_token_head`) | 3 | **3,000,000** | ×16 | 22.375 GiB |

The four quantized-KV profiles use **one sequence** and **2,048-token prefill chunks**. They passed allocation, API readiness and generation; these are capacity-qualified profiles, not filled-window benchmarks. [Copyable launch commands and memory settings](docs/QUANTIZED-KV.md).

### KV accuracy retention versus BF16

| KV cache | MTP | BF16 prediction agreement | Prediction disagreement | Tie-aware agreement |
|---|---:|---:|---:|---:|
| Scaled 8-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.80% |
| Rotated 4-bit | 0 | **98.18%** (486/495) | **1.82%** | 99.39% |
| Scaled 8-bit | 3 | **98.79%** (489/495) | **1.21%** | 99.80% |
| Rotated 4-bit | 3 | **98.38%** (487/495) | **1.62%** | 99.60% |

Measured on **495 teacher-forced prompt positions** against the same Htr5Core model with BF16 KV. Strict agreement compares complete top-choice sets; tie-aware agreement permits a shared tied top token. Disagreement is the measured prediction change, not a broad task-accuracy loss. All six functional fixtures passed in every compared cell. [Exact counts, logprob deltas and evidence](evidence/v0.2.1-release-qualification.json).

This is a serving update: **Htr5Core weights, tokenizer and weight quantization are unchanged**. Vision weights/processor assets remain included.

## Quantization recipe

The recommended Htr5Core model uses a mix of precisions rather than treating every component the same.

| Component | Htr5Core precision |
|---|---|
| Selected dense language-model weights / shared experts | HTR5 (5-bit) |
| Routed experts | HTR4 (4-bit) |
| Output head | HTR4 (4-bit) |
| Hyperconnections | GPTQ8 (8-bit integer) |
| PLE, vision tower and merger | Original BF16 |
| Norms, supporting tensors and integer controls | Retained component formats |
| Speculative draft | Included mixed-precision MTP draft |

## Method and lineage

HTR combines calibration-guided error correction with a trellis-based weight format designed for Strix Halo. The goal is to reduce weight storage while keeping the model close to its reference behavior.

Its lineage includes [exlr8](https://github.com/kindlingai/exlr8), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [QTIP](https://arxiv.org/abs/2406.11235). See the [methodology](https://github.com/HawgAuto/HTR/blob/main/docs/METHODOLOGY.md) and [full acknowledgments](https://github.com/HawgAuto/HTR/blob/main/docs/LINEAGE.md).

![The HTR quantization pipeline](media/method-map.png)

<a id="size-memory-and-1m-context"></a>

## Size, memory and context profiles

The **Htr5Core** checkpoint contains **37 weight files**: 23 target safetensors shards, 13 indexed HTR expert shards and one draft safetensors shard. An additional 26 files supply the configuration, tokenizer, processors, manifests and license. Shard consolidation is lossless, not requantization.

| Download component | Files | GB | GiB |
|---|---:|---:|---:|
| Target checkpoint (including PLE and vision) | 23 | 109.027 | 101.539 |
| Routed experts | 13 | 60.705 | 56.536 |
| MTP draft | 1 | 3.978 | 3.704 |
| **Complete checkpoint + supporting files** | **63** | **173.759** | **161.826** |

Exact checkpoint total: **173,759,296,105 bytes**; card, artwork and reports are additional. The largest shard is 4.997 GB. [File sizes and SHA-256](evidence/checkpoint-manifest.json) · [Sizing data](evidence/package-sizing.json)

**PLE is already included in these totals.** Its n-gram embedding tables account for 102.400 GB (95.368 GiB) of the target payload. The runtime reads requested PLE rows from SSD instead of placing the entire table on the GPU. Keep the model on a fast local SSD; download size is not resident GPU memory.

### Resident memory and context profiles — Htr5Core

The intended host has **128 GB shared system memory**. On the historical v0.2.0/default MTP3 profile, the engine logged **65.68 GiB model-load allocation**, **8 GiB BF16 KV**, and **0.19 GiB graph memory**. Those are separate allocations, not a total-RAM or co-loading allowance: the runtime, compilation, activations and filesystem cache also need headroom. Keep the large n-gram tables on a fast local SSD and avoid another large shared-memory GPU workload.

**Historical v0.2.1 quantized-KV profiles:** choose 2M, 3M or 4M from the [v0.2.1 profile table](#choose-your-context-profile). The cache budget and MTP mode must match.

**Retained BF16 1M context:** use `--context 1000000 --yarn-factor 4 --kv-cache-gib 27`. MTP0/MTP3, with either one or four slots, passed the public builder and cold replay on the v0.2.0 final image. The measured profiles reached approximately **99.2–104.1 GiB peak GPU shared-memory usage (GTT)** across build/replay and used swap during startup/compilation. The documented commands do not impose a Docker memory cap.

With four slots, the shared 27 GiB KV pool has an equal-share estimate of **281,684 tokens per slot for MTP0** or **255,030 for MTP3**; it is not four independent million-token histories. Warmups reached **32,768 input tokens**. Full-million prefill and long-position retrieval have not been measured. [Per-profile memory, capacity and startup results](docs/RUNNING-v0.2.0.md#measured-bf16-kv--yarn-4-profiles).

## Quantization quality

This matched **4-bit dense-layer comparison on WikiText** illustrates the choice of HTR. Only the selected dense weights change; both methods use the same reference model, whose experts are already quantized.

| Dense-layer format | PPL ↓ | KL ↓ | Top-1 agreement ↑ |
|---|---:|---:|---:|
| BF16 dense-layer reference | 4.295 | 0.0000 | 100.00% |
| HTR 4-bit | **4.308** | **0.0284** | **93.77%** |
| GPTQ 4-bit | 4.344 | 0.0520 | 91.68% |

**PPL** measures next-token prediction on the text; lower is better. **KL** measures divergence from the reference over its top 128 tokens plus a remaining-mass bucket; lower is better. It is not full-vocabulary KL. **Top-1 agreement** is how often the models choose the same next token—not task accuracy. Bold values mark the better quantized result.

HTR stays closer to the reference than GPTQ in this comparison. This illustrative HTR4 study is not a final-composite measurement of either shipped variant. These are component-level results, not a final score for the complete mixed-precision release. The [full quality report](docs/RECIPE-COMPARISONS.md) covers the other precision and calibration choices, including their tradeoffs; [source metrics](evidence/recipe-quality.json) retain the detailed measurements.

## Release verification — Htr5Core

The v0.2.2 integration matrix covers ordinary/MTP3 generation, all three Octave formats, retained INT8/INT4 KV, bounded PLE off/on and native fresh/seeded startup. Five matched pairs have exact tokens/text/finish and teacher-probe logprobs; legacy probes did not request logprobs. The EOS-enabled prose probes hit their128-token cap, so they prove bounded nonempty generation rather than natural termination. Tool-result turns stopped normally. [Qualification evidence](evidence/v0.2.2-release-qualification.json) · [Public-command delivery receipt](evidence/v0.2.2-delivery.json).

Single-sequence/configured32768/2GiB-KV execution does not establish new filled-window, concurrency, broad model-accuracy or image/video scores. Earlier qualification and benchmark evidence is retained with its original version/image/sampler labels; the same checkpoint does not make those measurements new-runtime results. FP8 KV and unrelated deferred research are not claimed.

[Historical v0.2.0 qualification](evidence/v0.2.0-release-qualification.json) · [Historical serving measurements](evidence/v0.2.0-serving-summary.json) · [Release history](docs/RELEASES.md) · [Vision and MTP inventory](evidence/vision-and-mtp.json).

## Download statistics

Full-repository downloads include [`meta.yaml`](meta.yaml) automatically. For selective downloads, include it alongside the checkpoint files so [Hugging Face's normal download counter](https://huggingface.co/docs/hub/models-download-stats) can record the request. No extra client telemetry is added; Hub counts are not a count of complete downloads or unique users.

## License and attribution

Source model: Qwen/Qwen3.8-Flash-Next, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Quantization/tooling: HawgAuto HTR; serving infrastructure: the separately pinned HawgAuto vLLM/HaloLoom stack.

The weights and derivative are governed by **Qwen Community License 1.0**, preserved verbatim in [LICENSE](LICENSE). This is not Apache-2.0 or the HTR code repository's MIT license. The license contains conditions for commercial Model-as-a-Service and AI Work Assistant use; review the complete text before deployment. Model authorship and capabilities belong to Qwen; HTR does not change those license obligations.
