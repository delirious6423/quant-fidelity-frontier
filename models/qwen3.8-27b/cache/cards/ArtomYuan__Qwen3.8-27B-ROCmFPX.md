---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - qwen3.8
  - rocmfp4
  - rocmfpx
  - strix.cpp
  - gguf
  - mtp
library_name: gguf
---

# Qwen3.8-27B-ROCmFPX

**ROCmFPX quantized GGUF (7 tiers + vision mmproj)** of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — official BF16 weights, locally requantized.

## ⚡ Quick Start

Requires [strix.cpp](https://github.com/ArtomYuan/strix.cpp) (this fork supports ROCmFPX; **upstream llama.cpp cannot load these files** — format note at the end).

```bash
# Text only (smallest tier: Q4_FAST ~13.6 GiB)
strix-server -m Qwen3.8-27B_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080

# Vision: add --mmproj mmproj-Qwen3.8-27B_BF16.gguf
# All tiers fit on a single 128 GB machine (largest: 26.7 GiB)
```

## 🎯 Which tier? (3-second decision)

- **Fastest**: Q4_0_ROCMFP4_FAST (13.6 GiB)
- **Speed + stabler quality**: Q4_0_ROCMFP4_STRIX (14.0 GiB — recommended sweet spot)
- **Higher-bit reference**: Q6_0_ROCMFPX (21.0 GiB)
- **Agent / tool-call**: Q6_0_ROCMFPX_AGENT (23.6 GiB) / Q8_0_ROCMFPX_AGENT (26.7 GiB)
- **Quality first**: Q8_0_ROCMFPX (26.3 GiB)

## Model Overview

- **Base model**: Qwen3.8-27B (dense, official weights by Alibaba)
- **Quantization**: BF16 -> ROCmFP4 / ROCmFPx, 7 variants
- **MTP**: GGUFs carry an MTP head — enable with `strix-server --spec-type draft-mtp` (speculative decoding)
- **Vision**: multimodal — pair with the bundled mmproj (see Usage)

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

## Quantization Tiers — what's here & how to choose

All files are builds of the same official weights through `llama-quantize` (strix.cpp fork); upstream llama.cpp cannot load them (see Format Note). Quick guide:

- **Q4_0_ROCMFP4_FAST** — smallest & fastest baseline (4.25 bpw). Bulk weights in single-scale FP4.
- **Q4_0_ROCMFP4_FAST_COHERENT** — FAST + Q6_K token embeddings (better coherence, +0.2 bpw).
- **Q4_0_ROCMFP4_STRIX** — FAST + Q6_K embeddings + dual-scale attention K/V (quality-biased; same speed class as FAST — the recommended quality/speed sweet spot).
- **Q6_0_ROCMFPX / Q8_0_ROCMFPX** — higher-bit reference layouts (6.50 / 8.25 bpw). Decode is slower than the Q4 family on this dense model (Q6's unpack overhead is the heaviest).
- **Q6_0_ROCMFPX_AGENT / Q8_0_ROCMFPX_AGENT** — agent / tool-call oriented routing variants (attention & FFN boosts on sensitive layers).

The full type ladder lives in the strix.cpp repo (`llama-quantize --help`).

## Benchmarks (llama-bench)

Measured on AMD Strix Halo (gfx1151), `llama-bench -ngl 99 -p 256,2048 -n 256 -r 5`, strix.cpp build 11107. (Dense 27B — all parameters active, so tg is inherently lower than MoE models. Each figure is the mean of the last 4 of 5 reps — the first rep carries one-time warm-up cost and is excluded.)

| Variant | tg256 (t/s) | pp256 (t/s) | pp2048 (t/s) | Size | bpw |
|:--|--:|--:|--:|--:|--:|
| Q4_0_ROCMFP4_FAST | 14.05 | 418.0 | 398.9 | 13.6 GiB | 4.25 |
| Q4_0_ROCMFP4_FAST_COHERENT | 13.99 | 404.0 | 389.8 | 13.9 GiB | ~4.45 |
| Q4_0_ROCMFP4_STRIX | 14.02 | 400.2 | 385.4 | 14.0 GiB | ~4.49 |
| Q6_0_ROCMFPX | 9.45 | 312.1 | 309.3 | 21.0 GiB | 6.50 |
| Q6_0_ROCMFPX_AGENT | 8.70 | 335.6 | 330.4 | 23.6 GiB | 6.50 |
| Q8_0_ROCMFPX | 7.93 | 371.8 | 365.6 | 26.3 GiB | 8.25 |
| Q8_0_ROCMFPX_AGENT | 7.83 | 368.9 | 363.0 | 26.7 GiB | 8.25 |

## Usage

```bash
# Text only:
strix-server -m Qwen3.8-27B_Q4_0_ROCMFP4_FAST.gguf -ngl 99 -fa on --port 8080

# Vision (multimodal; --mmproj enables image input):
strix-server -m Qwen3.8-27B_Q4_0_ROCMFP4_FAST.gguf --mmproj mmproj-Qwen3.8-27B_BF16.gguf -ngl 99 -fa on --port 8080
```

- Swap `-m` to any other variant file to run that variant (see Files).
- OpenAI-compatible API served at `http://127.0.0.1:8080/v1`.
- MTP head: load with `strix-server --spec-type draft-mtp` for speculative MTP acceleration.

## Format Note (ROCmFPX)

**ROCmFPX is a quantization format family (strix.cpp extension: GGML types 100–107 / FTYPE 100–119) — upstream llama.cpp cannot load these files. Load them with [strix.cpp](https://github.com/ArtomYuan/strix.cpp), a llama.cpp fork that supports all upstream GGUF formats plus ROCmFP4/ROCmFPx natively.**

## License & Disclaimer

- **Base model**: Qwen3.8-27B (Alibaba) — Apache-2.0
- **This quantized version**: Apache-2.0 (ArtomYuan)

This is a third-party community quantization (not an official Qwen release), not affiliated with Alibaba / Qwen. Provided for research and learning; use is subject to the base model license (Apache-2.0). No warranty is provided.
