---
base_model:
  - Qwen/Qwen3.8-27B
  - harness-hub/Qwen3.8-27B-pi-adapted
license: apache-2.0
tags:
  - mlx
  - mtplx
  - qwen3_5
  - pi
---

# Qwen3.8-27B-pi-adapted-MTPLX-Speed-FP16

Community M1/M2 Apple Silicon MTPLX build of [`harness-hub/Qwen3.8-27B-pi-adapted`](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted-GGUF) with native `float16` non-quantized weights/scales and intact `mtp.*` speculative decoding head.

## What Was Done
1. Merged [`harness-hub/Qwen3.8-27B-pi-adapted`](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted-GGUF) (`r=32, alpha=64`, scale `2.0`) into [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) in `float32` at the `safetensors` level, cast to `float16` (avoiding M1/M2 `BF16` prefill overhead), and preserved all 15 native `mtp.*` tensors.
2. Forged with `mtplx forge build --dtype fp16 --verification full-load` using the [`Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16`](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed-FP16) mixed-precision recipe (4-bit `g32` body, 8-bit `g64` recurrence/embedding/head layers, unquantized FP16 MTP head).
3. Tuned on a 128 GB M2 Ultra (`D2` default: **51.7 tok/s** vs. 27.3 tok/s AR; ~34–45 tok/s in live `pi` sessions).

## Quick Start
Using [mtplx 2.12.3](https://github.com/youssofal/MTPLX/releases/tag/v2.12.3):
```bash
mtplx serve --model avery-forrest/Qwen3.8-27B-pi-adapted-MTPLX-Speed-FP16 --depth 2 --reasoning off

```

Keep thinking disabled (`--reasoning off`), as the `pi` adapter was RL-trained without thinking.
