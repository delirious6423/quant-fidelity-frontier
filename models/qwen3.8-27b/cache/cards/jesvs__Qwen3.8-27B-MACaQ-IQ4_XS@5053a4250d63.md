---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
tags:
- macaq
- hadamard
- iq4_xs
- gguf
- quantized
---

# Qwen3.8-27B — MACaQ IQ4_XS

> [!WARNING]
> **FORK-ONLY QUANT.** These weights are Hadamard-rotated by [MACaQ](https://github.com/huawei-csl/MACaQ) and **will not load in stock llama.cpp**. Use the fork binaries bundled with that repo (`llamacpp/llama.cpp/build/bin/`).

A single full-precision-pipeline quant of Qwen3.8-27B produced with the MACaQ D1 method, shared alongside its complete `recipe.json` for reproducibility.

## Recipe

| | |
|---|---|
| Method | MACAQ D1 + block-Hadamard (block 64) + vector-codebook refinement |
| Format | IQ4_XS (embedding Q4_K, output Q6_K) |
| imatrix calibration | WikiText-2 (~1M tokens) |
| VQ refinement | 16,384 calibration tokens, 62 blocks |
| Size | 13.9 GiB (14,955,222,592 bytes) |
| SHA-256 | `4e678b73b58c4764dcf28ba11bd79ee4a0b3154ac7433c46fbad961121a3e5ab` |

## Files

| File | Size | Notes |
|---|---|---|
| `qwen38-27b-macaq-iq4xs.gguf` | 14 GB | Hadamard-rotated, fork-only |
| `qwen38-27b-macaq-iq4xs.recipe.json` | 549 KB | Full pipeline recipe (hashes, per-block VQ tensor shapes) |

## Measured vs llama.cpp Q4_K_M

Same seed/temperature/split, dual-GPU 16 GB + 12 GB pool, `-ngl all`, fork binary for the MACaQ side:

| Eval | MACaQ IQ4_XS | Q4_K_M |
|---|---|---|
| Quality (39 local tests) | 31 | 32 (within noise) |
| Tool use (36 agentic scenarios) | **36/36** | 34/36 |
| Generation speed | 23.0 tok/s | 21.1 tok/s |
| VRAM | 13.9 GiB | 15.9 GiB |

Single-run local evals on one machine — treat near-noise deltas accordingly. The clear signals: full tool-call recovery (6/6 retries vs 4/6) and a ~12% smaller footprint at slightly better speed.

## Usage

Build the MACaQ fork per its README (CUDA build tested on Blackwell), then serve as usual:

```bash
llama-server -m qwen38-27b-macaq-iq4xs.gguf -ngl all -sm layer -fa on
```

Standard llama.cpp will refuse the file — rotated tensor data is only meaningful to the fork's kernels.

## License

Apache 2.0, matching the [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base model.
