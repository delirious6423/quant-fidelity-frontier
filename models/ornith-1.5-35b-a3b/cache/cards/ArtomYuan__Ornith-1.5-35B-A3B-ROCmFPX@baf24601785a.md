---
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
  - gguf
  - llama.cpp
  - ROCm
  - Strix Halo
  - gfx1151
  - moe
  - multimodal
  - quantization
  - qwen35moe
  - rocmfpx
  - vision
library_name: gguf
---

# Ornith-1.5-35B-A3B-ROCmFPX

**ROCmFPX quantized GGUF (7 tiers + vision mmproj)** of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — the official base model, locally requantized.

## ⚡ Quick Start

Requires [strix.cpp](https://github.com/ArtomYuan/strix.cpp) (this fork supports ROCmFPX; **upstream llama.cpp cannot load these files** — format note at the end).

```bash
# Single machine (small tiers, e.g. Q4_FAST 17.65 GiB)
strix-server -m Ornith-1.5-35B-A3B_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080

# Vision: add --mmproj mmproj-Ornith-1.5-35B_BF16.gguf
# All tiers fit on a single 128 GB machine (largest: 34.74 GiB)
```

## 🎯 Which tier? (3-second decision)

- **Fastest**: Q4_0_ROCMFP4_FAST (17.65 GiB, pp2048 1505 / tg256 72.6)
- **Q4 speed with stabler quality**: Q4_0_ROCMFP4_STRIX (17.81 GiB)
- **Quality/speed balance**: Q6_0_ROCMFPX (27.39 GiB)
- **Instruction following / Agent**: Q6_0_ROCMFPX_AGENT (30.86 GiB)
- **Highest quality**: Q8_0_ROCMFPX (34.17 GiB) / Q8_0_ROCMFPX_AGENT (34.74 GiB)

## Model Overview

- **Base model**: Ornith-1.5-35B-A3B (official)
- **Architecture**: `Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe`) — MoE, 35.51B total / ~3B active (A3B), 256 experts (8 active per token), 40 transformer layers + 1 MTP layer
- **Quantization**: 7 tiers — `Q4_0_ROCMFP4` (FAST / FAST_COHERENT / STRIX), `Q6_0_ROCMFPX` (base / AGENT), `Q8_0_ROCMFPX` (base / AGENT)
- **Vision**: ✅ bundled mmproj (CLIP + qwen3vl_merger vision tower)
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

Measured on AMD Strix Halo (gfx1151) with `llama-bench -ngl 99 -p 256,2048 -n 256 -r 5`. Numbers are mean ± sample standard deviation (n−1) over the **last 4 of 5 shots** (first shot dropped as one-time warm-up).

| Tier | Size | pp256 (t/s) | pp2048 (t/s) | tg256 (t/s) |
|:--|--:|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 17.65 GiB | 1289.8 ± 176.9 | 1504.9 ± 3.8 | 72.6 ± 0.0 |
| Q4_0_ROCMFP4_FAST_COHERENT | 17.79 GiB | 1374.5 ± 2.4 | 1499.3 ± 5.4 | 72.5 ± 0.1 |
| Q4_0_ROCMFP4_STRIX | 17.81 GiB | 1214.7 ± 279.8 | 1476.3 ± 2.1 | 72.3 ± 0.0 |
| Q6_0_ROCMFPX | 27.39 GiB | 694.0 ± 122.7 | 918.3 ± 3.3 | 54.7 ± 0.3 |
| Q6_0_ROCMFPX_AGENT | 30.86 GiB | 818.0 ± 230.6 | 1085.1 ± 2.4 | 51.6 ± 0.0 |
| Q8_0_ROCMFPX | 34.17 GiB | 1068.9 ± 176.8 | 1352.2 ± 2.3 | 48.3 ± 0.0 |
| Q8_0_ROCMFPX_AGENT | 34.74 GiB | 1101.6 ± 119.4 | 1349.0 ± 3.5 | 47.9 ± 0.1 |

Notes:
- pp256 (short prefill) shows noticeably higher variance than pp2048/tg256 on most tiers (±120–280 t/s, except COHERENT) — a measurement phenomenon, not a defect.
- "8-bit" = `Q8_0_ROCMFPX` (`Q8_0_ROCMFPX_AGENT` excepted — its hardening target is standard `Q8_0`); "sensitive layers" = `use_more_bits` selection; "dual-scale FP4" = attention K/V kept in dual-scale FP4.

## Usage

### ① Single machine (all tiers)

```bash
strix-server -m Ornith-1.5-35B-A3B_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080
# Vision: add --mmproj mmproj-Ornith-1.5-35B_BF16.gguf
```

- OpenAI-compatible API: `http://127.0.0.1:8080/v1`
- Vision is enabled with `--mmproj <file>` (no separate vision switch; `--special` is a grounding output parameter, not a vision switch)
- MTP: supported via `strix-server --spec-type draft-mtp` (speculative decoding, off by default)
- Context: up to 256K

## Format Note (ROCmFPX)

**ROCmFPX is a proprietary quantization format (GGML types 100–107, a strix.cpp extension) — upstream llama.cpp cannot load these files. Load them with [strix.cpp](https://github.com/ArtomYuan/strix.cpp), a llama.cpp fork that supports all upstream GGUF formats plus ROCmFPX natively.**

## License & Disclaimer

- **Base model**: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — MIT
- **mmproj (vision tower)**: Ornith's CLIP + qwen3vl_merger vision tower — Apache-2.0
- **This quantization**: Apache-2.0 (ArtomYuan)

This repository is a quantized version of the official base model, provided "as is", without warranty of any kind. Users assume all responsibility.
