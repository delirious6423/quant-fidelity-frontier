---
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
  - gguf
  - llama.cpp
  - ROCm
  - Strix Halo
  - gfx1151
  - qwen35moe
  - quantization
  - rocmfpx
  - vision
library_name: gguf
---

# Ornith-1.5-35B-A3B-ROCmFPX

**ROCmFPX quantized GGUF (7 tiers + vision mmproj)** of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — the official base model, locally requantized.

> **Base model license**: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) is MIT-licensed; its copyright notice and license text are retained. The packaging of this quantization is Apache-2.0.

## Engine: strix.cpp (tuned for gfx1151 / AI MAX 395)

This repository uses [strix.cpp](https://github.com/ArtomYuan/strix.cpp), a llama.cpp fork **specifically tuned for the AMD Strix Halo `gfx1151` (Ryzen AI MAX+ 395) UMA architecture and ROCm/HIP backend**, with native support for the ROCmFPX quantization format.

- **Upstream llama.cpp cannot load the ROCmFPX files in this repository.**
- Upstream llama.cpp also does not carry this gfx1151-specific tuning.
- In the tables below, `llama.cpp-rocm` refers to the pre-rename build (build 11107); `strix.cpp` refers to the current build (build f879d83c).

## ⚡ Quick Start

Requires [strix.cpp](https://github.com/ArtomYuan/strix.cpp) (this fork supports ROCmFPX; **upstream llama.cpp cannot load these files** — format note at the end).

```bash
# Single machine (small tiers, e.g. Q4_FAST 17.65 GiB)
strix-server -m Ornith-1.5-35B-A3B_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080

# Vision: add --mmproj mmproj-Ornith-1.5-35B_BF16.gguf
# All tiers fit on a single 128 GB machine (largest: 34.74 GiB)
```

## 🎯 Which tier? (3-second decision)

- **Fastest**: Q4_0_ROCMFP4_FAST (17.65 GiB)
- **Q4 speed with stabler quality**: Q4_0_ROCMFP4_STRIX (17.81 GiB)
- **Quality/speed balance**: Q6_0_ROCMFPX (27.39 GiB)
- **Instruction following / Agent**: Q6_0_ROCMFPX_AGENT (30.86 GiB)
- **Highest quality**: Q8_0_ROCMFPX (34.17 GiB) / Q8_0_ROCMFPX_AGENT (34.74 GiB)

## Model Overview

- **Base model**: Ornith-1.5-35B-A3B (official)
- **Architecture**: `Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe`) — MoE, 35.51B total / ~3B active (A3B), 256 experts (8 active per token), 40 transformer layers + 1 MTP layer
- **Quantization**: 7 tiers — `Q4_0_ROCMFP4` (FAST / FAST_COHERENT / STRIX), `Q6_0_ROCMFPX` (base / AGENT), `Q8_0_ROCMFPX` (base / AGENT)
- **MTP**: GGUFs carry an MTP head — enable with `strix-server --spec-type draft-mtp` (off by default)
- **Vision**: bundled mmproj (CLIP + qwen3vl_merger vision tower)
- **Context**: up to 256K

## Files

| File | Size |
|:--|--:|
| Ornith-1.5-35B-A3B_Q4_0_ROCMFP4_FAST.gguf | 17.65 GiB (18,956,527,040 B) |
| Ornith-1.5-35B-A3B_Q4_0_ROCMFP4_FAST_COHERENT.gguf | 17.79 GiB (19,103,532,480 B) |
| Ornith-1.5-35B-A3B_Q4_0_ROCMFP4_STRIX.gguf | 17.81 GiB (19,119,982,016 B) |
| Ornith-1.5-35B-A3B_Q6_0_ROCMFPX.gguf | 27.39 GiB (29,410,461,120 B) |
| Ornith-1.5-35B-A3B_Q6_0_ROCMFPX_AGENT.gguf | 30.86 GiB (33,137,821,120 B) |
| Ornith-1.5-35B-A3B_Q8_0_ROCMFPX.gguf | 34.17 GiB (36,693,321,152 B) |
| Ornith-1.5-35B-A3B_Q8_0_ROCMFPX_AGENT.gguf | 34.74 GiB (37,300,118,976 B) |
| mmproj-Ornith-1.5-35B_BF16.gguf | 0.84 GiB (902,822,240 B) |

SHA-256 checksums for all files: see `sha256.txt`.

## Quantization Tiers

| Tier | Size | BPW | Hardening (from quantization logs) | Suggested use |
|:--|--:|--:|:--|:--|
| Q4_0_ROCMFP4_FAST | 17.65 GiB | 4.27 | none (bare baseline) | Fastest / fully resident |
| Q4_0_ROCMFP4_FAST_COHERENT | 17.79 GiB | 4.30 | vocab → Q6_K | Everyday use |
| Q4_0_ROCMFP4_STRIX | 17.81 GiB | 4.31 | vocab → Q6_K; attention K/V → dual-scale FP4 | Q4 speed, stabler quality |
| Q6_0_ROCMFPX | 27.39 GiB | 6.63 | first 1/16 attention, first 1/16 + last 1/8 ffn_down → 8-bit | Quality/speed balance |
| Q6_0_ROCMFPX_AGENT | 30.86 GiB | 7.47 | attention (first 1/8, sensitive layers), ffn_down (first 1/8, last 1/4, sensitive layers), gate/up (sensitive layers), vocab/output → 8-bit | Agent / instruction following |
| Q8_0_ROCMFPX | 34.17 GiB | 8.27 | none (uniform across the model) | Quality first |
| Q8_0_ROCMFPX_AGENT | 34.74 GiB | 8.40 | attention (first 1/8, latter half, sensitive layers), ffn_down (first 1/8, last 1/4, sensitive layers), gate/up (first 1/8, sensitive layers), vocab/output → standard Q8_0 | Highest quality |

## Benchmarks (llama-bench)

### Comparison: llama.cpp-rocm vs strix.cpp

#### pp256 (t/s)

| Tier | llama.cpp-rocm | strix.cpp | Change |
|:--|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 1289.8 | 2214.68 | +71.7% |
| Q4_0_ROCMFP4_FAST_COHERENT | 1374.5 | 1961.69 | +42.7% |
| Q4_0_ROCMFP4_STRIX | 1214.7 | 2109.57 | +73.7% |
| Q6_0_ROCMFPX | 694.0 | 925.88 | +33.4% |
| Q6_0_ROCMFPX_AGENT | 818.0 | 1043.81 | +27.6% |
| Q8_0_ROCMFPX | 1068.9 | 1116.91 | +4.5% |
| Q8_0_ROCMFPX_AGENT | 1101.6 | 1130.77 | +2.6% |

#### pp2048 (t/s)

| Tier | llama.cpp-rocm | strix.cpp | Change |
|:--|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 1504.9 | 2538.55 | +68.7% |
| Q4_0_ROCMFP4_FAST_COHERENT | 1499.3 | 2519.73 | +68.1% |
| Q4_0_ROCMFP4_STRIX | 1476.3 | 2407.37 | +63.1% |
| Q6_0_ROCMFPX | 918.3 | 1203.10 | +31.0% |
| Q6_0_ROCMFPX_AGENT | 1085.1 | 1343.67 | +23.8% |
| Q8_0_ROCMFPX | 1352.2 | 1551.52 | +14.7% |
| Q8_0_ROCMFPX_AGENT | 1349.0 | 1547.02 | +14.7% |

#### tg256 (t/s)

| Tier | llama.cpp-rocm | strix.cpp | Change |
|:--|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 72.6 | 74.60 | +2.8% |
| Q4_0_ROCMFP4_FAST_COHERENT | 72.5 | 74.45 | +2.7% |
| Q4_0_ROCMFP4_STRIX | 72.3 | 74.05 | +2.4% |
| Q6_0_ROCMFPX | 54.7 | 56.27 | +2.9% |
| Q6_0_ROCMFPX_AGENT | 51.6 | 52.31 | +1.4% |
| Q8_0_ROCMFPX | 48.3 | 48.79 | +1.0% |
| Q8_0_ROCMFPX_AGENT | 47.9 | 48.52 | +1.3% |

#### MTP (strix.cpp)

| Tier | t/s | Acceptance |
|:--|--:|--:|
| Q4_0_ROCMFP4_FAST | 61.52 | 24.2% |
| Q4_0_ROCMFP4_FAST_COHERENT | 58.54 | 22.3% |
| Q4_0_ROCMFP4_STRIX | 61.44 | 24.0% |
| Q6_0_ROCMFPX | 47.82 | 28.8% |
| Q6_0_ROCMFPX_AGENT | 47.25 | 31.8% |
| Q8_0_ROCMFPX | 35.31 | 19.8% |
| Q8_0_ROCMFPX_AGENT | 40.52 | 26.4% |

Notes:

- The two sides are two stages of the same project: `llama.cpp-rocm` is the pre-rename build (build 11107); `strix.cpp` (build f879d83c) is **not just a rename** — it includes substantive changes in gfx1151/ROCm scheduling, the ROCmFPX quantization format, MTP speculative decoding, and multi-sequence validation.
- The `pp2048` gain is real for this pairing, but it is **not** a pure engine-speed claim: build, `-ub 512` and measurement setup all changed.
- `pp256` has high run-to-run variance; treat short-prefill numbers as indicative, not precise.
- `tg256` improved only about 1%–3%; decode is essentially flat.
- MTP was measured separately on strix.cpp; the llama.cpp-rocm table has no MTP. On this MoE model MTP is slower than plain `tg256` at single stream, so keep it off by default.

## Usage

### ① Single machine (all tiers)

```bash
strix-server -m Ornith-1.5-35B-A3B_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080

# Vision: add --mmproj mmproj-Ornith-1.5-35B_BF16.gguf
```

- OpenAI-compatible API: `http://127.0.0.1:8080/v1`
- Vision is enabled with `--mmproj` (no separate vision switch)
- MTP: `strix-server --spec-type draft-mtp` (speculative decoding, off by default)
- Context: up to 256K

### ② MTP

```bash
strix-server -m Ornith-1.5-35B-A3B_Q4_0_ROCMFP4_FAST.gguf \
  --spec-type draft-mtp --spec-draft-n-max 6 -ngl 99 -fa on --port 8080
```

MTP is useful for latency-sensitive single-stream generation on some models; on Ornith-1.5-35B-A3B the measured single-stream result is lower than plain decoding, so benchmark before enabling it.

## Tech Notes (advanced)

- `strix.cpp` is tuned for the gfx1151 UMA architecture and ROCm/HIP backend; `-ngl 999` places all layers on the GPU.
- MTP uses the model's built-in `nextn` head, not an external draft model. Enable with `--spec-type draft-mtp`; draft depth is controlled by `--spec-draft-n-max`.
- The MTP numbers above are single-stream, 128 tokens, and include the model's thinking output; they are not the same scenario as plain `tg256`.

## Format Note (ROCmFPX)

**ROCmFPX is a proprietary quantization format (GGML types 100–107, a strix.cpp extension) — upstream llama.cpp cannot load these files. Load them with [strix.cpp](https://github.com/ArtomYuan/strix.cpp), a llama.cpp fork that supports all upstream GGUF formats plus ROCmFPX natively.**

## License & Disclaimer

- **Base model**: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — MIT
- **mmproj (vision tower)**: Ornith's CLIP + qwen3vl_merger vision tower — Apache-2.0
- **This quantization**: Apache-2.0 (ArtomYuan)

This repository is a quantized version of the official base model, provided "as is", without warranty of any kind. Users assume all responsibility.
