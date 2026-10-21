---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - gguf
  - rocmfpx
  - qwen3_5
quantized_from: unsloth/Qwen3.8-27B-GGUF (BF16)
---

# Qwen3.8-27B — ROCmFPX quants (Q8 full + 16 GB hybrid)

These are **quantizations of [`unsloth/Qwen3.8-27B-GGUF` (BF16)](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)**
(original model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), the hybrid
attention/SSM `qwen35` architecture with an MTP `nextn` head).

> [!IMPORTANT]
> **You need the [ROCmFPX fork of llama.cpp](https://github.com/charlie12345/ROCmFPX)** (or a llama.cpp
> build with ROCmFPX support). These files use the experimental `q8_0_rocmfpx` (type 103) and/or
> `q4_0_rocmfp4_fast` (type 101) weight formats, which stock llama.cpp releases do **not**
> understand — loading them elsewhere will fail with an unknown tensor type error.

## Files

| File | Recipe | Size | For |
|---|---|---|---|
| `Qwen3.8-27B-Q8_0_ROCMFPX.gguf` | pure `q8_0_rocmfpx`, all weights | 26.9 GB (8.25 bpw) | large-VRAM systems (e.g. Strix Halo) |
| `Qwen3.8-27B-Q4FAST-Q8-sensitive.gguf` | bulk `q4_0_rocmfp4_fast` + sensitive tensors at `q8_0_rocmfpx` | 16.4 GB (5.15 bpw) | ~24 GB VRAM laptops (leaves room for KV cache) |

### Hybrid recipe (16 GB file)

Sensitive-tensor selection mirrors the tiers in Unsloth's
[`UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) dynamic recipe (Q6_K tier =
most sensitive, Q5_K = next), rebalanced onto a strict two-type q4/q8 mix to hit ~16 GB:

| Tensors at `q8_0_rocmfpx` (165 tensors) | Why |
|---|---|
| `attn_q/k/v/output` (17 full-attention layers) | attention projections (Unsloth Q5_K/Q6_K tier; `attn_v` is Q6_K there) |
| `attn_gate` + `ssm_out` (48 linear-attn/SSM layers) | Q5_K tier in UD-Q4_K_XL |
| `output.weight` head | Q6_K tier in UD-Q4_K_XL |

Everything else is `q4_0_rocmfp4_fast` (340 tensors, incl. `ffn_gate/up/down`, `attn_qkv`,
`token_embd` — Unsloth keeps embeddings at Q4_K too), norms/biases stay `f32` (360 tensors).
The MTP head (`nextn.eh_proj`) is auto-protected at `q8_0` by the quantizer's draft-sensitive logic.

### How they were made

```bash
# pure Q8 (from the ROCmFPX fork; CPU-only build works fine for quantization)
llama-quantize Qwen3.8-27B-BF16-00001-of-00002.gguf \
  Qwen3.8-27B-Q8_0_ROCMFPX.gguf Q8_0_ROCMFPX

# 16 GB hybrid
llama-quantize \
  --tensor-type "attn_q.weight=q8_0_rocmfpx" \
  --tensor-type "attn_k.weight=q8_0_rocmfpx" \
  --tensor-type "attn_v.weight=q8_0_rocmfpx" \
  --tensor-type "attn_output.weight=q8_0_rocmfpx" \
  --tensor-type "attn_gate.weight=q8_0_rocmfpx" \
  --tensor-type "ssm_out.weight=q8_0_rocmfpx" \
  --tensor-type "^output.weight=q8_0_rocmfpx" \
  Qwen3.8-27B-BF16-00001-of-00002.gguf \
  Qwen3.8-27B-Q4FAST-Q8-sensitive.gguf Q4_0_ROCMFP4_FAST
```

## Usage

This quant is **text weights only**. Qwen3.8 is multimodal — for vision support, pair it with the
mmproj file: this repo includes **`mmproj-F16.gguf`** (mirrored from
[`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), which also offers
`mmproj-BF16.gguf`).

```bash
# build ROCmFPX for your GPU (see the repo README; e.g. Strix Halo):
env JOBS=16 scripts/build-strix-rocmfp4-mtp.sh

./build-strix-rocmfp4/bin/llama-cli \
  -m Qwen3.8-27B-Q8_0_ROCMFPX.gguf --mmproj mmproj-F16.gguf \
  -dev Vulkan0 -ngl 999 -fa on --jinja
```

## Benchmarks

> [!NOTE]
> All Benchmarks were run on an AI max 395+ with 128GB RAM

| Model | Size | Depth | pp512 t/s | tg128 t/s |
|---|---|---|---|---|
| Q8_0_ROCMFPX | 26.25 GiB | 4096 | 304.65 ± 1.28 | 7.47 ± 0.03 |
| Q8_0_ROCMFPX | 26.25 GiB | 16384 | 247.26 ± 1.00 | 7.37 ± 0.10 |
| Q8_0_ROCMFPX | 26.25 GiB | 65536 | 143.50 ± 0.43 | 6.84 ± 0.00 |
| Q4_0_ROCMFP4_FAST | 16.38 GiB | 4096 | 329.88 ± 4.00 | 11.15 ± 0.13 |
| Q4_0_ROCMFP4_FAST | 16.38 GiB | 16384 | 262.43 ± 1.17 | 10.95 ± 0.02 |
| Q4_0_ROCMFP4_FAST | 16.38 GiB | 65536 | 147.77 ± 0.59 | 9.49 ± 0.02 |

Quality comparison vs BF16 source (perplexity / KLD): TBD.

## Attribution & license

- Quantized from: [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (BF16 shards);
  sensitivity tiers referenced from their `UD-Q4_K_XL`
- Original model: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)
- License: Apache-2.0 (inherited)
- Quant formats by the [ROCmFPX project](https://github.com/charlie12345/ROCmFPX)
