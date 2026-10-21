---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
language:
  - en
tags:
  - gguf
  - llama.cpp
  - rocm
  - rocmfpx
  - rocmfp4
  - imatrix
  - mtp
  - speculative-decoding
  - strix-halo
  - gfx1151
  - 4-bit
  - conversational
---

# Qwen3.8-27B — Q4_0_ROCMFP4_STRIX (ROCmFPX)

A **4.39 BPW** quantization of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using the
**ROCmFP4** data type family and an **unsloth importance matrix**, built for AMD Strix Halo
(Ryzen AI MAX 300-series / Radeon 8000S, `gfx1151`).

The point of this file is that it gets a 27B model into **13.98 GiB** while keeping the
**built-in MTP (NextN) speculative-decoding head** — so it runs at roughly **1.8× the decode speed**
of the same file with speculation off, with no separate draft model.

| | |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) |
| BF16 source used | [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) |
| Imatrix | unsloth's `imatrix_unsloth.gguf` — 496 entries over 1251 chunks, `unsloth_calibration_dataset` |
| Quant type | `Q4_0_ROCMFP4_STRIX` (`general.file_type = 105`) |
| Size / bits-per-weight | **13.98 GiB** / **4.39 BPW** (from 52.1 GiB / 16.00 BPW) |
| Tensors | 866 · **0 quantization fallbacks** |
| Context | 262,144 tokens (native) |
| Architecture | `qwen35` — hybrid SSM + attention, 65 blocks, 27.3B params |
| MTP head | **included** (`qwen35.nextn_predict_layers = 1`) |

> ⚠️ **This is not a normal GGUF.** It uses data types that stock `llama.cpp` cannot read at all,
> and it needs a specific fork to load. See
> [Requirements](#requirements-you-need-a-specific-llama-cpp-fork). It will **not** work in
> LM Studio, Ollama, or `llama.cpp` built from upstream `master`.

---

## Table of contents

- [What is ROCmFP4 / ROCmFPX?](#what-is-rocmfp4--rocmfpx)
- [How this model was made](#how-this-model-was-made)
- [Requirements: you need a specific llama.cpp fork](#requirements-you-need-a-specific-llama-cpp-fork)
  - [Prebuilt Windows binaries (no build required)](#prebuilt-windows-binaries-no-build-required)
- [Build and run](#build-and-run)
- [Recommended settings (validated on Strix Halo)](#recommended-settings-validated-on-strix-halo)
- [How the settings were chosen (full sweep results)](#how-the-settings-were-chosen-full-sweep-results)
- [KV cache options](#kv-cache-options)
- [Prompt cache sizing](#prompt-cache-sizing)
- [Known limitations](#known-limitations)
- [Reproducing this exact file](#reproducing-this-exact-file)
- [Credits](#credits)

---

## What is ROCmFP4 / ROCmFPX?

`ROCmFP4` is a 4-bit quantization family with custom GPU kernels written for AMD hardware
(`ROCMFP4`, `ROCMFP4_FAST`, and a `STRIX` mix preset), added by the
[charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) fork of `llama.cpp`. The same fork
also adds other custom types (ROCmFPX 2/3/6/8-bit, TurboQuant).

Two things matter to a user:

1. **They are not in upstream `llama.cpp`.** Upstream has no `GGML_TYPE_Q4_0_ROCMFP4`; loading this
   file with a stock build fails. A GGUF carrying these types is only readable by this fork (or a
   build that has merged these kernels).
2. **`_STRIX` is a recipe, not a single type.** It is a per-tensor assignment tuned for Strix Halo's
   memory bandwidth: hot weights get the faster variant, attention K/V get the higher-quality
   dual-scale variant, and embeddings/norms are kept in higher precision. See
   [the type map](#the-resulting-type-map) below.

---

## How this model was made

1. **Downloaded the BF16 GGUF** of Qwen3.8-27B from unsloth (2 shards, 50.9 GiB total) plus
   unsloth's importance matrix.
2. **Re-quantized with the ROCmFP4 `_STRIX` recipe**, driven by that imatrix.
3. **Left the source split intact** — no `--keep-split` — so the two shards merged into a single
   output file, which is what keeps the MTP head in the same file as the model.

The imatrix is genuinely consumed, not decoration: both ROCmFP4 quantizers are imatrix-aware
(`rocmfp4_quantize_q4_0_weighted` / `..._fast_weighted`) and use it for imatrix-weighted scale
search.

The exact command:

```bash
llama-quantize \
  --imatrix imatrix_unsloth.gguf \
  Qwen3.8-27B-BF16-00001-of-00002.gguf \
  Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  Q4_0_ROCMFP4_STRIX
```

`llama-quantize` reports the source as 52115.19 MiB @ 16.00 BPW and the output as
14307.95 MiB @ **4.39 BPW**, with **zero** shape fallbacks.

### The resulting type map

| Type | Tensors | Applied to |
|---|---:|---|
| `Q4_0_ROCMFP4_FAST` | 422 | hot weights — `attn_q`, `attn_qkv`, `attn_output`, `ffn_*`, `output.weight` |
| `Q4_0_ROCMFP4` | 82 | `attn_k`, `attn_v` — the dual-scale quality variant |
| `Q6_K` | 1 | `token_embd.weight` |
| `Q8_0` | 1 | `blk.64.nextn.eh_proj.weight` — the MTP projection, auto-protected |
| `F32` | 360 | norms, SSM 1-D vectors, `ssm_conv1d` (shape oddity) |

### The MTP head is preserved

Qwen3.8-27B ships a NextN/MTP head as `blk.64.nextn.*`. It survives quantization and is present in
this file — **you do not need a separate draft model**:

```
qwen35.nextn_predict_layers = 1
Q8_0  blk.64.nextn.eh_proj.weight
F32   blk.64.nextn.enorm.weight / hnorm.weight / shared_head_norm.weight
```

When served correctly, the log confirms it is drafting from the model's own head:

```
load_model: creating MTP draft context against the target model '...ROCMFP4_STRIX.gguf'
common_speculative_state_draft_mtp: - n_max=4, n_min=0, p_min=0.00, n_embd=5120
load_model: speculative decoding context initialized
```

---

## Requirements: you need a specific llama.cpp fork

**This file will not load in upstream `llama.cpp`.** Use the fork it was made with:

- Repo: **https://github.com/charlie12345/ROCmFPX**
- Validated commit: **`c49ebdbd5c9f01ec242369f9e7f7967855f80cba`** (branch `main`)

Because forks move, **pin a commit**. `Q4_0_ROCMFP4` support is a prerequisite; check that
`llama-quantize --help` lists `Q4_0_ROCMFP4_STRIX` (type 105) before blaming the file.

### Prebuilt Windows binaries (no build required)

A prebuilt **Windows x64** build for `gfx1151` is attached to this repo, so you can skip compiling:

**[`llama.cpp-ROCmFPX-windows-gfx1151.zip`](./llama.cpp-ROCmFPX-windows-gfx1151.zip)** — 58 MB

It contains `llama-server`, `llama-cli`, `llama-quantize`, `llama-imatrix`, `llama-bench`,
`llama-perplexity` and `test-backend-ops`, plus every matching DLL (`ggml-hip`, `ggml-vulkan`,
`ggml-cpu`, `ggml-base`, `llama`, `llama-common`, `mtmd`). Built from commit `c49ebdbd…` with the
flags below under MSVC 14.44. Extract it anywhere and run the executables from that folder — the
DLLs must stay next to the `.exe` files.

The archive ships **only this project's binaries**. Install the runtime parts yourself:

| Backend | What you must install | Why |
|---|---|---|
| **Vulkan** — `-dev Vulkan0` | A current AMD Adrenalin driver | `ggml-vulkan.dll` needs only `vulkan-1.dll`, which the graphics driver already provides. Nothing else to do. |
| **ROCm** — `-dev ROCm0` | The **AMD ROCm HIP SDK, 7.x** | `ggml-hip.dll` imports `libhipblas.dll`, which ships *only* with the ROCm SDK — an `amdhip64_7.dll` sitting in `C:\Windows\System32` is not sufficient on its own. Then add `<ROCm install>\bin` to `PATH`. |

Also install the **Visual C++ 2015–2022 x64 Redistributable** (`MSVCP140.dll`, `VCRUNTIME140.dll`),
which almost every Windows machine already has.

The `7` in `amdhip64_7.dll` means the ROCm major version must be **7.x**, not 6.x. Device code for
`gfx1151` is baked into `ggml-hip.dll`, so the ROCm backend will not run on a different AMD GPU —
use `-dev Vulkan0` there, or build from source.

`SHA256SUMS` inside the archive lists a checksum for every file (`Get-FileHash` on Windows,
`sha256sum -c` elsewhere). Because the zip contains `.exe` files fetched from the internet, Windows
may mark them as blocked; if they refuse to launch, right-click the **zip** → *Properties* → tick
**Unblock** → *OK*, and then extract.

### Build notes

You need an **HIP-capable build** (`-DGGML_HIP=ON`). A Vulkan-only build is much slower on prefill.

```bash
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_HIP_FORCE_MMQ=ON \
  -DGGML_HIP_ROCWMMA_FATTN=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1151 \
  -DGGML_VULKAN=ON \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build -j
```

`-DGGML_HIP_FORCE_MMQ=ON` is **required** by the ROCmFP4 kernels — do not drop it.
`gfx1151` is Strix Halo; substitute your target.

Two gotchas found while building this on Windows, in case they save someone an afternoon:

- **Pin MSVC to 14.44.** On MSVC 14.51+ a `<cmath>` change makes ROCm clang redeclare
  `isgreater`/`isless`/… as `__device__`, and every `ggml-cuda/` translation unit fails. Note that
  `vswhere -latest` may return an install whose only toolset is the broken one.
- **Pass absolute paths to the HIP SDK's `clang.exe`/`clang++.exe`** so a different clang earlier on
  `PATH` cannot be picked up.

---

## Build and run

Runtime DLLs/SOs come from the HIP SDK (ROCm 7.1 here), so put them on the library path first. On
Windows you can skip this section entirely by using the
[prebuilt zip](#prebuilt-windows-binaries-no-build-required).

**Tuned command (the one validated below):**

```bash
llama-server \
  -m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  -dev ROCm0 -ngl 999 -fa on --jinja \
  -c 262144 -b 512 -ub 512 -t 16 \
  -ctk q8_0 -ctv q8_0 --cache-ram 10240 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.0 \
  --host 127.0.0.1 --port 8090
```

The chat UI is embedded in `llama-server` — open `http://127.0.0.1:8090`.

**For a one-shot CLI check** (note `-st`, not `-no-cnv` — this fork's `llama-cli` rejects
`-no-cnv` / `--no-conversation`):

```bash
llama-cli -m Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  -dev ROCm0 -ngl 999 -fa on --jinja -st \
  -p "Write a Python LRU cache with O(1) get/put." -n 256 \
  --spec-type draft-mtp --spec-draft-n-max 4 --spec-draft-p-min 0.0
```

---

## Recommended settings (validated on Strix Halo)

**Test machine:** AMD Ryzen AI MAX+ 395, Radeon 8060S (`gfx1151`), **128 GB** unified LPDDR5X,
Windows 11, ROCm 7.1 HIP SDK.

| Measured on the recommended config | |
|---|---|
| Memory (RSS, whole server, 262k ctx) | **~43 GB** |
| Decode — code generation | **~23–25 tok/s** |
| Decode — reasoning / thinking | **~28 tok/s** |
| Decode — prose | **~21 tok/s** |
| Prefill | **~240 tok/s** near-empty, ~150–190 tok/s at 64k depth |
| MTP speedup | **~1.8×** vs the same file with speculation off (9.9 → 17.9 tok/s, prose at 16k ctx) |

All figures are greedy decoding on this machine. Prefill is dominated by context depth, decode by
memory bandwidth; both scale with your hardware.

### Why each setting

| Setting | Value | Reason (measured) |
|---|---|---|
| `-dev` | `ROCm0` | ROCm prefill is **25–40 % faster** than Vulkan. Vulkan is faster at *raw* decode (+33 %), but MTP shrinks that to 0–5 %, and prefill dominates agent workloads. |
| `-c` | `262144` | The model's native maximum. Only **16 of 65 blocks** are full attention (`full_attention_interval = 4`), so a huge context is cheap. |
| `-b` / `-ub` | `512` / `512` | `-ub 2048` (a common default) costs **9–15 % prefill** and gains **nothing** for decode. |
| `-ctk` / `-ctv` | `q8_0` | ~100 % of `f16` decode speed on both backends, halves KV memory, and is the closest quantized KV gets to lossless. |
| `--cache-ram` | `10240` | 10 GiB holds a **whole 262k conversation** (it needs 9,888 MiB). |
| `--spec-draft-n-max` | `4` | 2–4 are equivalent; **6 costs 15–25 %**. |
| `--spec-draft-p-min` | `0.0` | **~12 % faster than the 0.75 default.** Draft early-stop only — cannot affect output quality. |

Two of these are the opposite of what you'd guess, so they are worth repeating: the **default
`p-min` of 0.75 costs ~12 %**, and **`n-max 6` costs 15–25 %**. Both are the "more conservative"
direction, and both are slower.

---

## How the settings were chosen (full sweep results)

Everything below was swept on the test machine with greedy sampling. If you only read one thing,
read the two "wrong defaults" above.

### Draft depth — `--spec-draft-n-max` (prose)

| device | n-max 2 | n-max 3 | n-max 4 | n-max 6 |
|---|---:|---:|---:|---:|
| Vulkan0 | 23.00 | 22.75 | 21.38 | 17.65 |
| ROCm0 | 20.58 | 20.96 | 20.65 | 17.65 |

Drafts beyond position ~3 are almost never accepted (acceptance by position:
`0.747, 0.467, 0.267, 0.120, 0.053, 0.000`), yet the target still pays a batched verification over
all 6 candidates plus extra recurrent-state rollback. On agent-style prompts, 3 vs 4 is a wash that
depends on output type — `n-max 3` favours code emission, `n-max 4` favours thinking-heavy output:

| | n-max 3 code | n-max 4 code | n-max 3 think | n-max 4 think |
|---|---:|---:|---:|---:|
| Vulkan0 | **26.35** | 23.84 | 30.20 | **31.23** |
| ROCm0 | 24.13 | 23.98 | 27.44 | **29.61** |

Since a reasoning model emits far more thinking tokens than answer tokens, **4** is the better
default.

### Draft early-stop — `--spec-draft-p-min`

| p-min | code tok/s | think tok/s | vs 0.0 | acceptance | mean acc. length |
|---|---:|---:|---:|---:|---:|
| **0.0** | **23.19** | **28.70** | **100 %** | 0.892 | 4.55 |
| 0.25 | 23.09 | 28.55 | 100 % | 0.892 | 4.55 |
| 0.5 | 22.09 | 26.23 | 95 % | 0.882 | 4.40 |
| 0.75 *(default)* | 21.06 | 25.50 | 88 % | 0.964 | 4.39 |
| 0.9 | 18.21 | 23.06 | 78 % | 0.963 | 4.19 |

Monotonic — raising `p-min` always costs speed. Note the misleading acceptance column: `p-min 0.9`
has the *highest* acceptance fraction **and is 22 % slower**, because it stops drafting early and so
converts fewer tokens per verification step. **Judge speculative decoding by tokens per step, not by
acceptance rate.** `p-min` is a draft early-stop only — every accepted token is still verified
against the target model, so it cannot affect output quality.

### Backend — `-dev ROCm0` vs `-dev Vulkan0`

Prefill (ROCm wins):

| test | ROCm0 | Vulkan0 |
|---|---:|---:|
| pp512, no MTP | 283.9 | 223.0 |
| pp4096, no MTP | 270.4 | 216.7 |
| prefill 1534 tok @128k ctx | 241.5 | 188.9 |
| prefill 1534 tok @256k ctx | 246.4 | 177.4 |

Decode (Vulkan wins, until MTP is enabled):

| setting | ROCm0 | Vulkan0 |
|---|---:|---:|
| no MTP | 9.80 | **13.08** (+33 %) |
| MTP, n-max 4, code @128k | 23.98 | 23.84 (tie) |
| MTP, n-max 4, think @128k | 29.61 | **31.23** (+5 %) |

So Vulkan's large raw-decode advantage essentially disappears once MTP is on. Total-time model for
a turn at 64k context (1200 tokens prefilled, 900 generated), where `1/K` = a full 64k re-prefill
every K turns:

| config | no-miss | 1/50 | 1/30 | 1/25 | 1/15 | 1/5 |
|---|---:|---:|---:|---:|---:|---:|
| Vulkan0 + turbo4 KV | **38.3** | **45.5** | 50.3 | 52.7 | 62.4 | 110.4 |
| ROCm0 + turbo4 KV | 40.8 | 46.2 | **49.7** | **51.5** | **58.7** | **94.4** |

Crossover is roughly **one full re-prefill per 25–30 turns**. ROCm0 is the default because its
downside is small (+6 % if you never miss) and its upside is large (−15 % when you do). Choose
Vulkan0 only if your prefix is append-only and your turns are decode-bound.

### Batch size — `-b` / `-ub`

No-MTP `llama-bench` (pp512 / pp4096 / tg128):

| device | b/ub | pp512 | pp4096 | tg128 |
|---|---|---:|---:|---:|
| ROCm0 | 512/512 | 283.9 | **270.4** | 9.70 |
| ROCm0 | 2048/2048 | 281.5 | 229.0 | 9.80 |
| Vulkan0 | 512/512 | 223.0 | **216.7** | 13.08 |
| Vulkan0 | 2048/2048 | 222.8 | 197.7 | 13.12 |

Use `-ub 512`.

---

## KV cache options

The fork accepts these KV cache types (`-ctk` / `-ctv`):
`f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`, plus fork-only
`q4_0_rocmfp4, q4_0_rocmfp4_fast, q3_0_rocmfpx, q6_0_rocmfpx, q8_0_rocmfpx`, and TurboQuant
`turbo3, turbo4`. Flash attention is enabled for **all** quantized KV types in this build.

Measured at 128k context (RSS = whole server working set; decode relative to `f16`):

| device | KV type | RSS GB | code tok/s | think tok/s | prefill tok/s | decode vs f16 |
|---|---|---:|---:|---:|---:|---:|
| Vulkan0 | **turbo4** | 33.55 | **26.14** | **31.12** | 177.5 | **106 %** |
| Vulkan0 | turbo3 | 33.05 | 25.03 | 30.08 | 175.3 | 102 % |
| Vulkan0 | q4_0_rocmfp4_fast | 33.43 | 24.62 | 28.20 | 159.5 | 100 % |
| Vulkan0 | f16 | 39.26 | 24.56 | 31.31 | 188.9 | 100 % |
| Vulkan0 | q8_0 *(standard)* | 35.54 | 24.20 | 30.04 | 191.5 | 99 % |
| Vulkan0 | q4_0_rocmfp4 | 33.56 | 22.56 | 27.12 | 165.5 | 92 % |
| Vulkan0 | q8_0_rocmfpx | 35.78 | 10.57 | 13.32 | 80.7 | **43 %** |
| Vulkan0 | q6_0_rocmfpx | 35.65 | 8.28 | 10.38 | 51.1 | **34 %** |
| ROCm0 | f16 | 39.45 | 24.09 | 29.48 | 241.5 | 100 % |
| ROCm0 | q8_0 *(standard)* | 35.74 | 24.01 | 27.52 | 229.4 | 100 % |
| ROCm0 | turbo3 | 33.19 | 23.83 | 25.19 | 241.5 | 99 % |
| ROCm0 | turbo4 | 33.69 | 23.43 | 27.07 | 238.8 | 97 % |
| ROCm0 | q8_0_rocmfpx | 35.98 | 15.82 | 19.09 | 102.8 | 66 % |
| ROCm0 | q4_0_rocmfp4 | 33.86 | 15.31 | 17.57 | 68.2 | 64 % |

**Read this before picking a KV type:**

- The fork's **ROCmFPX fp cache types (`q8_0_rocmfpx`, `q6_0_rocmfpx`) are 2–3× slower** — 34–43 %
  of `f16` decode on Vulkan. Despite the name matching the model's family, they are the wrong choice
  for the cache. Only the **ROCmFP4 pair** has a fast path on Vulkan.
- **`turbo4` (TurboQuant) is the best fork KV type** — the fastest decode measured (106 % of `f16`)
  while saving 5.7 GB, and near-free on ROCm.
- **`q8_0` is the safe default** and is what the recommended config uses: ~100 % of `f16` speed,
  best Vulkan prefill, saves 3.7 GB. 8-bit KV is as close to lossless as quantized KV gets.
- KV quantization generally costs a little decode speed (dequant work) — it buys memory, not speed,
  except `turbo4` on Vulkan.
- **The MTP draft KV is always `f16`** regardless of `-ctk`/`-ctv` (the draft context logs
  `cache_k=f16, cache_v=f16`), so it adds a fixed ~4 KiB/token that you cannot quantize this way.
- The fork applies a Walsh–Hadamard rotation to the KV cache for accuracy, which is *disabled* for
  fp3/TurboQuant types (they carry their own), logged as
  `attention rotation disabled for fp3 ROCmFPX/TurboQuant KV cache`.

> **Naming trap:** TurboQuant's CLI names are `turbo3` / `turbo4`, **not** `turbo3_0` / `turbo4_0`.
> The latter fails with `Unsupported cache type` even though `GGML_TYPE_TURBO4_0` is in the allow-list.

---

## Prompt cache sizing

`llama-server` can keep conversation prefixes resident in RAM (`--cache-ram`) so a returning agent
skips re-prefill. Measured state size (`q8_0` KV), from the
`prompt_save: ... total state size` log line:

| prompt tokens | total state | of which MTP draft |
|---:|---:|---:|
| 1,005 | 186.960 MiB | 3.945 |
| 7,959 | 445.284 MiB | 31.242 |
| 31,900 | 1,334.633 MiB | 125.218 |

Perfectly linear: `state(N) = 149.6 MiB + N × 0.037148 MiB` (**38.04 KiB/token**). That checks out
against the architecture exactly — main KV is 16 layers × 4 KV heads × 256 dim × 2 = 32,768
elem/token × 1.0625 B = 34.0 KiB, plus 4.0 KiB f16 draft.

| KV type | state per token | full 262k context | 10 GiB covers |
|---|---:|---:|---:|
| `q8_0` *(recommended)* | 38.04 KiB | 9.66 GiB | 271,629 tokens |
| `turbo4` | ~22.4 KiB | ~5.8 GiB | ~460,000 tokens |
| `f16` | 68.04 KiB | 17.2 GiB | ~155,000 tokens |

So **`--cache-ram 10240` fits a whole 262k conversation** with ~5 % headroom. The ~150 MiB fixed
term is the hybrid model's recurrent state *per cached conversation*, so each additional cached 262k
conversation needs another ~9.7 GiB. The stock default is 8192 MiB, which is not enough for a full
262k context.

---

## Known limitations

- **Requires the fork.** Will not load in upstream `llama.cpp`, LM Studio, or Ollama. See
  [Requirements](#requirements-you-need-a-specific-llama-cpp-fork).
- **MTP output is not bit-identical to non-speculative greedy decoding.** In a controlled A/B
  (same server, same prompt, greedy, only MTP toggled) output diverged — an equally coherent but
  differently-phrased continuation, at character 503 of ~800. This persisted even with the fork's
  `--spec-mtp-strict-qwen` (which requires `-np 1` and claims exact greedy output). Likely cause is
  batched verification shifting logits enough to flip a near-tie; this is a hybrid SSM/attention
  model, which is the hardest case for state rollback. **Treat MTP as a speed feature, not a
  reproducible-output feature.** If you need reproducible greedy output, disable speculation.
- **KV-cache quantization quality was not measured.** The speed and memory figures above say nothing
  about accuracy. `q8_0` was chosen as the default precisely because it is near-lossless and costs
  nothing measurable; `turbo4` is faster but is 4-bit. This has not been validated with perplexity
  or task evals.
- **Quantization quality of the model itself was not independently evaluated** (no perplexity or
  benchmark run). The type map follows the `_STRIX` recipe and calibration uses unsloth's own
  imatrix, but treat quality as unverified until you test your own workload.
- **One unreproduced GPU fault was observed:** a single `FLASH_ATTN_EXT failed` /
  `ROCm error: unspecified launch failure` on a 64k prefill with `--cache-ram 65536` and three
  prompts already cached. `f16`, `q8_0`, and `turbo4` subsequently each prefilled 63,813 tokens
  cleanly, so it looks transient rather than type-specific. Related gotcha: **a crashed HIP context
  leaves the GPU wedged, and the next launch dies ~0.2 s in at `hipMemGetInfo`.** That second
  failure is not a new bug — kill the process, wait ~30 s, relaunch.
- **Model metadata says `general.quantized_by = "Unsloth"`**, inherited from unsloth's BF16 source
  file. The ROCmFP4 quantization was performed locally, so correct that field
  (`--override-kv general.quantized_by=str:<you>`) if you redistribute.

---

## Reproducing this exact file

1. Get the fork at the pinned commit and build with `-DGGML_HIP=ON -DGGML_HIP_FORCE_MMQ=ON`
   (see [Build notes](#build-notes)).
2. Download `Qwen3.8-27B-BF16-00001-of-00002.gguf`, `...-00002-of-00002.gguf` and
   `imatrix_unsloth.gguf` from [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
3. Quantize — pass **only shard 1** as input and **omit `--keep-split`**, so the shards merge into
   one file that retains the MTP head:

```bash
llama-quantize \
  --imatrix imatrix_unsloth.gguf \
  Qwen3.8-27B-BF16-00001-of-00002.gguf \
  Qwen3.8-27B-Q4_0_ROCMFP4_STRIX.gguf \
  Q4_0_ROCMFP4_STRIX 32
```

Use `--dry-run` first: it prints the projected size/BPW and the full per-tensor type map without
writing anything. Expect `14307.95 MiB @ 4.39 BPW` and 0 fallbacks.

4. Verify the output carries the MTP head — `blk.64.nextn.*` tensors (4 of them) and
   `qwen35.nextn_predict_layers = 1`.

---

## Credits

- **Base model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0)
- **BF16 GGUF + importance matrix:** [unsloth](https://huggingface.co/unsloth) — the imatrix does
  real work here (imatrix-weighted scale search), so this quantization would be worse without it.
- **ROCmFP4 / ROCmFPX types and kernels:**
  [charlie12345/ROCmFPX](https://github.com/charlie12345/ROCmFPX) — the fork that makes these types
  exist, and the only way to run this file.
- **Upstream:** [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

All benchmark numbers in this card were measured on the single Strix Halo machine described above,
with greedy decoding. They will differ on other hardware; the *relative* comparisons (ROCm vs
Vulkan, `n-max` 4 vs 6, `p-min` 0.0 vs 0.75, `-ub` 512 vs 2048) are the parts worth carrying over.