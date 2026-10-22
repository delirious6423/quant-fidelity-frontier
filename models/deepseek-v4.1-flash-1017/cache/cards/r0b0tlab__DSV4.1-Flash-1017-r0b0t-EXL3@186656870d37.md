---
license: mit
base_model:
  - deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: exllamav3
tags:
  - exl3
  - quantized
  - r0b0tlab
---

# DSV4.1-Flash-1017-r0b0t-EXL3

Max-performance mixed-precision EXL3 quantization of [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by **r0b0tlab** (`@mr_r0b0t`).

## Summary

- **Base model:** `deepseek-ai/DeepSeek-V4.1-Flash-1017`
- **Format:** EXL3 (`mul1` codebook) + FP8 (`E4M3` + `E8M0` block scales) + BF16 vision tower (`300.33 GiB` across `17` shards, `190,952` tensors)
- **Layout:**
  - `head`: `4-bit` EXL3 (`K=4`)
  - Trunk MLA `wq_b`, `wo_b` & `shared_experts.{w1,w2,w3}`: `4-bit` EXL3 (`K=4`)
  - Trunk MLA latent `wq_a`, `wkv`, `wo_a`: `8-bit` FP8 (`E4M3` + `E8M0`)
  - Routed MoE experts (`40` layers × `384` experts): `1–3 bit` EXL3 (`K=1..3`, zero `K=4` outlier stragglers; `100%` of `15,360` experts fit resident in GPU memory on a single 128 GB node)
  - Speculative decoding: In-pack `3`-stage NextN (`mtp.0..2`)
  - Vision & Engram: Unquantized `BF16` vision tower (`266` tensors) and `FP8` Engram tables (`shards 16–17` + `engram_token_map.u32`)
