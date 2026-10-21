---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - gguf
  - llama.cpp
  - ROCm
  - Strix Halo
  - gfx1151
  - quantization
  - rocmfpx
  - mtp
library_name: gguf
---

# Qwen3.8-27B-ROCmFPX

**ROCmFPX quantized GGUF (7 tiers + vision mmproj)** of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — official BF16 weights, locally requantized.

## Engine: strix.cpp (tuned for gfx1151 / AI MAX 395)

This repository uses [strix.cpp](https://github.com/ArtomYuan/strix.cpp), a llama.cpp fork **specifically tuned for the AMD Strix Halo `gfx1151` (Ryzen AI MAX+ 395) UMA architecture and ROCm/HIP backend**, with native support for the ROCmFPX quantization format.

- **Upstream llama.cpp cannot load the ROCmFPX files in this repository.**
- Upstream llama.cpp also does not carry this gfx1151-specific tuning.
- `llama.cpp-rocm` in the tables is the pre-rename build (build 11107); `strix.cpp` is the current build (build f879d83c).

## ⚡ Quick Start

Requires [strix.cpp](https://github.com/ArtomYuan/strix.cpp) (this fork supports ROCmFPX; **upstream llama.cpp cannot load these files** — format note at the end).

```bash
# Single machine (smallest tier 13.56 GiB)
strix-server -m Qwen3.8-27B_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080

# Vision: add --mmproj mmproj-Qwen3.8-27B_BF16.gguf
# All tiers fit on a single 128 GB machine (largest: 26.69 GiB)
```

## 🎯 Which tier? (3-second decision)

- **Fastest**: Q4_0_ROCMFP4_FAST
- **Q4 speed with stabler quality**: Q4_0_ROCMFP4_STRIX
- **Quality/speed balance**: Q6_0_ROCMFPX
- **Instruction following / Agent**: Q6_0_ROCMFPX_AGENT / Q8_0_ROCMFPX_AGENT
- **Highest quality**: Q8_0_ROCMFPX

## Model Overview

- **Base model**: Qwen/Qwen3.8-27B
- **Architecture**: dense 27B — all parameters active; tg is inherently lower than MoE models
- **Quantization**: 7 tiers — `Q4_0_ROCMFP4` (FAST / FAST_COHERENT / STRIX), `Q6_0_ROCMFPX` (base / AGENT), `Q8_0_ROCMFPX` (base / AGENT)
- **MTP**: built-in MTP head — enable with `strix-server --spec-type draft-mtp` (off by default)
- **Vision**: bundled mmproj (CLIP + qwen3vl_merger vision tower)
- **Context**: up to 256K

## Files

| File | Size |
|:--|--:|
| Qwen3.8-27B_Q4_0_ROCMFP4_FAST.gguf | 13.56 GiB (14,562,235,840 B) |
| Qwen3.8-27B_Q4_0_ROCMFP4_FAST_COHERENT.gguf | 13.90 GiB (14,929,749,440 B) |
| Qwen3.8-27B_Q4_0_ROCMFP4_STRIX.gguf | 13.98 GiB (15,013,963,200 B) |
| Qwen3.8-27B_Q6_0_ROCMFPX.gguf | 20.98 GiB (22,528,382,400 B) |
| Qwen3.8-27B_Q6_0_ROCMFPX_AGENT.gguf | 23.55 GiB (25,283,188,160 B) |
| Qwen3.8-27B_Q8_0_ROCMFPX.gguf | 26.26 GiB (28,193,396,160 B) |
| Qwen3.8-27B_Q8_0_ROCMFPX_AGENT.gguf | 26.69 GiB (28,664,108,480 B) |
| mmproj-Qwen3.8-27B_BF16.gguf | 0.87 GiB (931,145,952 B) |

SHA-256 checksums for all files: see `sha256.txt`.

## Quantization Tiers

| Tier | BPW | Hardening (from quantization logs) | Suggested use |
|:--|--:|:--|:--|
| Q4_0_ROCMFP4_FAST | 4.25 | none (bare baseline) |
| Q4_0_ROCMFP4_FAST_COHERENT | ~4.45 | vocab → Q6_K |
| Q4_0_ROCMFP4_STRIX | ~4.49 | vocab → Q6_K; attention K/V → dual-scale FP4 |
| Q6_0_ROCMFPX | 6.50 | first 1/16 attention, first 1/16 + last 1/8 ffn_down → 8-bit |
| Q6_0_ROCMFPX_AGENT | 6.50 | attention (first 1/8, sensitive layers), ffn_down (first 1/8, last 1/4, sensitive layers), gate/up (sensitive layers), vocab/output → 8-bit |
| Q8_0_ROCMFPX | 8.25 | none (uniform across the model) |
| Q8_0_ROCMFPX_AGENT | 8.25 | attention (first 1/8, latter half, sensitive layers), ffn_down (first 1/8, last 1/4, sensitive layers), gate/up (first 1/8, sensitive layers), vocab/output → standard Q8_0 |

## Benchmarks (llama-bench)

### Comparison: llama.cpp-rocm vs strix.cpp

Measured on AMD Strix Halo (gfx1151) with the same prompt and generation lengths:

```bash
llama-bench -p 256,2048 -n 256 -r 5 -fa 1 -ngl 999 -ub 512
```

`llama.cpp-rocm` is build 11107; `strix.cpp` is build f879d83c. `llama.cpp-rocm` numbers are the mean of the last 4 of 5 shots; `strix.cpp` numbers are the mean of 5 shots (first shot included).

#### pp256 (t/s)

| Tier | llama.cpp-rocm | strix.cpp | Change |
|:--|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 418.00 | 742.23 | +77.6% |
| Q4_0_ROCMFP4_FAST_COHERENT | 404.00 | 737.62 | +82.6% |
| Q4_0_ROCMFP4_STRIX | 400.20 | 675.41 | +68.8% |
| Q6_0_ROCMFPX | 312.10 | 356.33 | +14.2% |
| Q6_0_ROCMFPX_AGENT | 335.60 | 379.16 | +13.0% |
| Q8_0_ROCMFPX | 371.80 | 406.56 | +9.3% |
| Q8_0_ROCMFPX_AGENT | 368.90 | 408.85 | +10.8% |

#### pp2048 (t/s)

| Tier | llama.cpp-rocm | strix.cpp | Change |
|:--|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 398.90 | 761.75 | +91.0% |
| Q4_0_ROCMFP4_FAST_COHERENT | 389.80 | 752.34 | +93.0% |
| Q4_0_ROCMFP4_STRIX | 385.40 | 685.64 | +77.9% |
| Q6_0_ROCMFPX | 309.30 | 359.73 | +16.3% |
| Q6_0_ROCMFPX_AGENT | 330.40 | 383.51 | +16.1% |
| Q8_0_ROCMFPX | 365.60 | 423.90 | +15.9% |
| Q8_0_ROCMFPX_AGENT | 363.00 | 416.01 | +14.6% |

#### tg256 (t/s)

| Tier | llama.cpp-rocm | strix.cpp | Change |
|:--|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 14.05 | 14.04 | -0.1% |
| Q4_0_ROCMFP4_FAST_COHERENT | 13.99 | 14.04 | +0.4% |
| Q4_0_ROCMFP4_STRIX | 14.02 | 14.04 | +0.1% |
| Q6_0_ROCMFPX | 9.45 | 9.47 | +0.2% |
| Q6_0_ROCMFPX_AGENT | 8.70 | 8.73 | +0.3% |
| Q8_0_ROCMFPX | 7.93 | 7.94 | +0.1% |
| Q8_0_ROCMFPX_AGENT | 7.83 | 7.83 | +0.0% |

#### MTP (strix.cpp)

| Tier | t/s | Acceptance |
|:--|--:|--:|
| Q4_0_ROCMFP4_FAST | 21.92 | 27.3% |
| Q4_0_ROCMFP4_FAST_COHERENT | 23.54 | 30.8% |
| Q4_0_ROCMFP4_STRIX | 23.24 | 29.7% |
| Q6_0_ROCMFPX | 17.53 | 34.0% |
| Q6_0_ROCMFPX_AGENT | 14.07 | 27.2% |
| Q8_0_ROCMFPX | 14.34 | 30.5% |
| Q8_0_ROCMFPX_AGENT | 14.78 | 32.0% |

## Usage

### ① Single machine (all tiers)

```bash
strix-server -m Qwen3.8-27B_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080

# Vision: add --mmproj mmproj-Qwen3.8-27B_BF16.gguf
```

- OpenAI-compatible API: `http://127.0.0.1:8080/v1`
- Vision is enabled with `--mmproj`
- MTP: `strix-server --spec-type draft-mtp` (off by default)
- Context: up to 256K

### ② MTP

```bash
strix-server -m Qwen3.8-27B_Q4_0_ROCMFP4_FAST.gguf \
  --spec-type draft-mtp --spec-draft-n-max 6 -ngl 99 -fa on --port 8080
```

MTP is useful for latency-sensitive single-stream generation on some models; benchmark before enabling it.

## Tech Notes (advanced)

- `strix.cpp` is tuned for the gfx1151 UMA architecture and ROCm/HIP backend; `-ngl 999` places all layers on the GPU.
- MTP uses the model's built-in `nextn` head, not an external draft model.

## Format Note (ROCmFPX)

**ROCmFPX is a proprietary quantization format (GGML types 100–107, a strix.cpp extension) — upstream llama.cpp cannot load these files. Load them with [strix.cpp](https://github.com/ArtomYuan/strix.cpp), a llama.cpp fork that supports all upstream GGUF formats plus ROCmFPX natively.**

## License & Disclaimer

- **Base model**: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — Apache-2.0
- **This quantization**: Apache-2.0 (ArtomYuan)

This is a third-party community quantization (not an official Qwen release), not affiliated with Alibaba / Qwen. Provided for research and learning; use is subject to the base model license (Apache-2.0). No warranty is provided.
