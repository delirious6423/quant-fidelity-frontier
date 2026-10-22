---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - llama.cpp
  - ROCm
  - Strix Halo
  - gfx1151
  - moe
  - multimodal
  - quantization
  - qwen
  - qwen3
  - qwen4exp
  - rocmfpx
  - vision
library_name: gguf
---
# Qwen3.8-Flash-Next-ROCmFPX

**ROCmFPX quantized GGUF of Qwen3.8-Flash-Next (7 tiers + vision mmproj)** — locally requantized from the official [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

> **License**: this quantization follows the base model's [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). It is **not** Apache-2.0.

## ⚡ Quick Start

Requires [strix.cpp](https://github.com/ArtomYuan/strix.cpp) (this fork supports ROCmFPX; **upstream llama.cpp cannot load this batch** — see the Format Note at the end).

```bash
# Single machine (small tier, e.g. Q4_FAST 87.8 GiB)
strix-server -m Qwen3.8-Flash-Next_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080

# Vision: append --mmproj mmproj-Qwen3.8-Flash-Next_BF16.gguf
# Large tiers (>=113 GiB): single-machine mmap paging, or two-machine RPC pooling — see Usage
```

OpenAI-compatible API: `http://127.0.0.1:8080/v1`.

## 🎯 Which tier? (3-second decision)

| Use case | Recommended tier |
|:--|:--|
| Smallest size / single-machine full-resident | **Q4_0_ROCMFP4_FAST** (87.8 GiB) |
| Daily driver / long context | **Q4_0_ROCMFP4_FAST_COHERENT / _STRIX** (113.3 GiB) |
| Balanced quality / speed | **Q6_0_ROCMFPX** (135.7 GiB) |
| Agent / instruction-following | **Q6_0_ROCMFPX_AGENT** (158.0 GiB) |
| Quality first / maximum quality | **Q8_0_ROCMFPX / _AGENT** (170.2 / 173.6 GiB) |

> Note: tiers above `~113 GiB` cannot be fully resident on a single 128 GB machine — use mmap paging (slower) or two-machine RPC pooling (see Usage).

## Model Overview

- **Base model**: Qwen3.8 Flash Next (official)
- **Architecture**: `qwen4exp` — MoE, 512 experts (10 active per token), 48 transformer blocks, hidden 2560, approx. 176.9B params total
- **Modalities**: text + vision (mmproj included: CLIP + `qwen3vl_merger` vision tower)
- **Context**: up to 256K (`qwen4exp.context_length = 262144`)

## Files

| File | Size |
|:--|--:|
| Qwen3.8-Flash-Next_Q4_0_ROCMFP4_FAST.gguf | 87.79 GiB (94,268,794,528 B) |
| Qwen3.8-Flash-Next_Q4_0_ROCMFP4_FAST_COHERENT.gguf | 113.30 GiB (121,652,681,888 B) |
| Qwen3.8-Flash-Next_Q4_0_ROCMFP4_STRIX.gguf | 113.33 GiB (121,683,156,128 B) |
| Qwen3.8-Flash-Next_Q6_0_ROCMFPX.gguf | 135.68 GiB (145,688,675,488 B) |
| Qwen3.8-Flash-Next_Q6_0_ROCMFPX_AGENT.gguf | 158.01 GiB (169,667,624,608 B) |
| Qwen3.8-Flash-Next_Q8_0_ROCMFPX.gguf | 170.15 GiB (182,698,116,768 B) |
| Qwen3.8-Flash-Next_Q8_0_ROCMFPX_AGENT.gguf | 173.58 GiB (186,375,492,768 B) |
| mmproj-Qwen3.8-Flash-Next_BF16.gguf | 0.85 GiB (907,542,784 B) |

SHA-256 checksums for all files: see `sha256.txt`.

## Quantization Tiers

| Tier | file_type | BPW | Size | Hardening (from quantization log) | Suggested use |
|:--|:--|--:|--:|:--|:--|
| Q4_0_ROCMFP4_FAST | 103 | 4.26 | 87.79 GiB | none (bare baseline) | smallest footprint / single-machine full-resident |
| Q4_0_ROCMFP4_FAST_COHERENT | 104 | 5.50 | 113.30 GiB | token embedding → Q6_K; per-layer embedding → Q8_0 | daily driver / long context |
| Q4_0_ROCMFP4_STRIX | 105 | 5.50 | 113.33 GiB | + attention K/V → dual-scale FP4 | Q4 size, steadier quality |
| Q6_0_ROCMFPX | 110 | 6.59 | 135.68 GiB | uniform `q6_0_rocmfpx` | balanced quality / speed |
| Q6_0_ROCMFPX_AGENT | 114 | 7.67 | 158.01 GiB | vocab/embed + selective attention → Q8_0_ROCMFPX | Agent / instruction-following |
| Q8_0_ROCMFPX | 111 | 8.26 | 170.15 GiB | uniform `q8_0_rocmfpx` | quality first |
| Q8_0_ROCMFPX_AGENT | 115 | 8.43 | 173.58 GiB | vocab/embed + selective attention → standard Q8_0 | maximum quality |

## Benchmarks (llama-bench)

### Main table — two-machine RPC pooling (2× Strix Halo, 256 GB pool)

**How it was measured**: strix.cpp's RPC build (`build-rpc`, sha256 `66dd122b…`) — two Strix Halo machines (128 GB unified memory each) pooled into a 256 GB virtual machine over a 40G link. A = client, B = `ggml-rpc-server`.

**Caliber**: r5 drop-first (drop the first shot, mean of the last 4 shots — removes warm-up effects).

| Tier | config | pp256 (t/s) | pp2048 (t/s) | tg256 (t/s) |
|:--|:--|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 3/1 · none | 525.89 | 565.77 | 27.99 |
| Q4_0_ROCMFP4_FAST_COHERENT | 3/1 · none | 529.56 | 552.62 | 27.84 |
| Q4_0_ROCMFP4_STRIX | 3/1 · none | 528.25 | 562.97 | 28.35 |
| Q6_0_ROCMFPX | 6/1 · dio | 311.99 | 361.93 | 22.84 |
| Q6_0_ROCMFPX_AGENT | 8/1 · dio | 363.27 | 419.87 | 21.72 |
| Q8_0_ROCMFPX | 3/1 · none | 458.33 | 530.87 | 20.22 |
| Q8_0_ROCMFPX_AGENT | 3/1 · none | 462.60 | 522.75 | 20.09 |

- "config" = layer split / load mode. `3/1`: A keeps 25% of layers, B holds 75%; `none`/`dio` = the `-lm` value (rationale in the Tech Notes at the end).
- r3 reference (3-shot drop-first): FAST 527.84 / 566.07 / 28.23; FC 526.26 / 555.85 / 28.06; STRIX 528.61 / 566.00 / 28.32; Q6 317.00 / 362.36 / 22.74; Q6A 364.82 / 419.17 / 21.93; Q8 464.03 / 533.08 / 20.30; Q8A 450.40 / 526.71 / 20.03.
- Sample jitter (tg256 stddev/mean) < 1% across all valid passes.

### Reference only — single-machine caliber (do not compare with the table above)

| Tier | machine / caliber | pp256 (t/s) | pp2048 (t/s) | tg256 (t/s) | Note |
|:--|:--|--:|--:|--:|:--|
| Q4_0_ROCMFP4_FAST | B, old auto run (not mmap) | 229.65 | 331.13 | 16.25 | high variance, reference only |
| Q4_0_ROCMFP4_FAST_COHERENT | B, single-machine mmap paging | 18.37 | 32.52 | 10.47 | mixed service window |
| Q4_0_ROCMFP4_STRIX | B, single-machine mmap paging | 104.47 | 155.27 | 11.58 | clean exclusive (cleanest) |

> These two groups were run differently (RPC pooling vs single-machine mmap paging) and on different engine builds — **archived side by side only**; do not compare the numbers across them. Single-machine paging is clearly slower than pooling (pp off by several×, tg by ~2–3×).

## Usage

### ① Single machine (recommended for small tiers)

```bash
strix-server \
  -m Qwen3.8-Flash-Next_Q4_0_ROCMFP4_FAST.gguf \
  -ngl 99 -fa on --port 8080
```

- Full-resident (`-ngl 99`): the whole model in unified memory. The 87.8 GiB tier fits one 128 GB machine; >=113 GiB needs paging or pooling.
- Paging mode (slow but runs large tiers): `-lm mmap -ngl 16`.
- Vision: `--mmproj <file>` (no separate switch).
- Context: up to 256K.

### ② Two-machine RPC pooling (all 7 tiers; the benchmarked method)

**Steps**: start the RPC server on machine B → run with `--rpc` on machine A (client); the model file must live on the client (A).

```bash
# ---- Machine B (RPC server) ----
systemd-run --user --no-block -p LimitMEMLOCK=infinity /bin/bash -c \
  'cd ~/source/strix.cpp/build/bin && \
   LD_LIBRARY_PATH=$PWD exec ./ggml-rpc-server --host 10.0.10.11 --port 50052'

# ---- Machine A (client) ----
systemd-run --user --scope -p MemoryMax=102400M -p MemorySwapMax=0 -- bash -c '
  ~/source/strix.cpp/build-rpc/bin/llama-bench \
    -m Qwen3.8-Flash-Next_Q4_0_ROCMFP4_STRIX.gguf \
    --rpc 10.0.10.11:50052 -ngl 99 --split-mode layer --tensor-split 3/1 \
    -p 256,2048 -n 256 -r 5 -o json -lm none'
```

**Key points** (rationale in the Tech Notes):

- Large-embedding tiers (FAST / FC / STRIX / Q8 / Q8A): `--tensor-split 3/1` + `-lm none`.
- Q6 / Q6A: `--tensor-split 6/1` (Q6A: `8/1`) + `-lm dio`.
- Restart `ggml-rpc-server` before switching models (clears GTT); the model file must be local on the client.

## Tech Notes (advanced)

<details>
<summary>Expand: load modes & RPC notes</summary>

- **Load mode on RPC runs**: use `-lm none` for tiers with large embeddings — `-lm dio` builds an equal-size intermediate buffer that can OOM the client; `-lm mmap` has a known deadlock with RPC; `-lm none` reads straight into the destination.
- **Keep the client "light on layers"**: the `qwen4exp` embedding tensors (token_embd + per-layer + output) can only be pinned on the client's local CPU — keep the client-side layer share small.
- **`--split-mode layer` is required**: `tensor` split isn't implemented for `qwen4exp`; `row` split isn't supported over RPC.
- **RPC hygiene**: restart `ggml-rpc-server` before switching models (clears GTT); run it with `LimitMEMLOCK=infinity` to escape the cgroup memlock ceiling.

</details>

## Format Note (ROCmFPX)

**ROCmFPX is an extension format of strix.cpp** (this batch uses GGML file types 103/104/105/110/111/114/115). **Upstream llama.cpp cannot load them** — use [strix.cpp](https://github.com/ArtomYuan/strix.cpp) (supports all upstream GGUF formats plus ROCmFPX natively).

## License & Disclaimer

- **Base model**: Qwen3.8 Flash Next (official) — `qwen-community-1.0` (from GGUF metadata)
- **mmproj (vision tower)**: CLIP + `qwen3vl_merger` of the base model — license follows the base model
- **This quantized version**: Apache-2.0

This is a quantized version of the official model, provided "as is", without warranty of any kind. Benchmark figures depend on the exact engine build and hardware; they are measured values under the stated environment, not end-to-end business throughput.
