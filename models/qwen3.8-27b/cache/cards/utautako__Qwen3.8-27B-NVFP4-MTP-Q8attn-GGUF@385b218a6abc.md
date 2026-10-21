---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: llama.cpp
pipeline_tag: image-text-to-text
tags:
  - gguf
  - qwen
  - qwen3.8
  - qwen3_5
  - nvfp4
  - modelopt
  - mtp
  - speculative-decoding
  - vision
  - vlm
  - multimodal
  - llama.cpp
  - quantized
  - blackwell
  - rtx-5090
---

GGUF conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized to
**NVFP4** in the MLP with NVIDIA **ModelOpt**, keeping the native **MTP** layer for speculative
decoding and a BF16 vision projector for image input.

This is the **accuracy-oriented Q8attn** variant: the attention and linear-attention (DeltaNet)
projections are stored at **`Q8_0`** (≈ the FP8 precision they were quantized to), rather than being
dropped to `Q4_K`. That keeps accuracy up at long context, where 4-bit attention / DeltaNet
projections hurt most. Benchmarked on an RTX 5090.

## Highlights

- **NVFP4 MLP (self-quantized)**: 193 NVFP4 MLP tensors, produced with NVIDIA ModelOpt from the
  full-precision `Qwen/Qwen3.8-27B`.
- **Q8 attention**: attention (`q/k/v/o`) and linear-attention / DeltaNet projections (`attn_qkv`,
  `attn_gate`, `ssm_out`, `ssm_alpha`, `ssm_beta`) are stored as **`Q8_0`**, for better accuracy than
  a 4-bit-attention build.
- **MTP included**: the GGUF keeps the native MTP layer for `draft-mtp` speculative decoding, roughly
  doubling generation throughput on this hardware.
- **Vision supported**: ships a BF16 `mmproj` projector for image input (Qwen3-VL vision path).
- **RTX 5090 tested**: MTP verified with a `draft-n` sweep; ~120–130 tok/s generation with MTP.

## Provenance

| Component | Source |
|---|---|
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| NVFP4 quantization | Self-quantized with [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) (recipe `w4a16_nvfp4 / fp8_attn`, calibrated on `cnn_dailymail`) |
| Runtime target | `llama.cpp` (Blackwell NVFP4 + Qwen3_5 hybrid + MTP support) |

## Files

| File                                       |                             Size | Description                                                                       |
| ------------------------------------------ | -------------------------------: | --------------------------------------------------------------------------------- |
| `Qwen3.8-27B-NVFP4-MTP-Q8attn.gguf`        | 17.81 GiB / 19,128,349,888 bytes | Main GGUF. NVFP4 MLP; attention / DeltaNet at `Q8_0`; MTP layer included. ~5.60 BPW. |
| `Qwen3.8-27B-NVFP4-MTP-Q6attn.gguf`        | 16.16 GiB / 17,350,051,520 bytes | Same NVFP4 MLP; attention / DeltaNet at `Q6_K` for smaller size (~5.08 BPW), MTP included. |
| `mmproj-Qwen3.8-27B-NVFP4-BF16.gguf`       |     888 MiB / 931,145,984 bytes  | BF16 vision projector for image input (334 tensors).                              |

## llama.cpp example

```bash
# Text-only
llama-server \
  -m Qwen3.8-27B-NVFP4-MTP-Q8attn.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  -c 213056 \
  -ngl 999

# With vision
llama-server \
  -m Qwen3.8-27B-NVFP4-MTP-Q8attn.gguf \
  --mmproj mmproj-Qwen3.8-27B-NVFP4-BF16.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  -c 163840 \
  -ngl 999
```

The context lengths above are what this repo's author runs on a 32 GB RTX 5090:
**`-c 213056` for text-only** and **`-c 163840` when the vision projector is loaded** (the mmproj plus
its image tokens take extra VRAM, so a smaller context leaves headroom).

> **A note on MTP depth.** A `draft-n` sweep on this model (below) shows generation throughput
> plateaus around `--spec-draft-n-max 2–3`. `d=2` gives the highest draft-acceptance and the most
> stable throughput; `d=3` reaches a slightly higher peak on code-like workloads. Values of `d≥4`
> gave no reliable speedup (acceptance collapses). `d=2` or `d=3` are both good choices.

## Benchmarks

Measured on RTX 5090, `-ngl 999`, `c=32768`, KV `q8_0`, generation of 512 tokens (`ignore_eos`),
draft-acceptance and generation t/s read from the server's own `timings` (ground truth).
`n` = `--spec-draft-n-max`; `n0` = autoregressive baseline (no MTP).

| MTP depth | code — gen t/s (accept) | prose — gen t/s (accept) |
|--:|--:|--:|
| n0 (AR)   | 65.2 (—)     | 67.1 (—)     |
| n1        | 94.1 (78%)   | 91.5 (70%)   |
| **n2**    | **124.0 (77%)** | **112.3 (63%)** |
| n3        | 129.2 (63%)  | 111.2 (49%)  |
| n4        | 125.3 (49%)  | 128.6 (50%)  |
| n5        | 137.0 (50%)  | 106.3 (35%)  |

MTP roughly **doubles** generation throughput over the non-speculative baseline (65 → ~124 tok/s).
Throughput plateaus at `n2`; beyond `n3` the draft-acceptance rate collapses and there is no reliable
gain. (End-to-end sanity check at `c=213056`: MTP active, ~78–86% acceptance, ~121–129 tok/s.)

## Model details

| Field | Value |
|---|---|
| GGUF architecture | `qwen35` |
| License metadata | `apache-2.0` |
| Text layers | 64 |
| MTP layers | 1 |
| `qwen35.block_count` | 65 |
| `qwen35.nextn_predict_layers` | 1 |
| Bits per weight | ~5.60 BPW |
| Vision projector architecture | `clip` / `qwen3vl_merger` |
| Vision projector precision | BF16 / F32 |

### Tensor composition of the main GGUF

| Tensor type | Count | Role |
|---|---:|---|
| NVFP4 | 193 | MLP (gate/up/down) — self-quantized with NVIDIA ModelOpt |
| Q8_0  | 308 | attention (q/k/v/o) + linear-attn / DeltaNet projections |
| Q6_K  | 3   | MTP-layer FFN |
| Q4_K  | 2   | remaining small tensors |
| F32   | 553 | norms / embeddings / scales |
| Total | 1059 | |

The conversion keeps the NVFP4 MLP tensors instead of requantizing all weights to K-quants, and keeps
attention / DeltaNet projections at `Q8_0` rather than `Q4_K`.

## License and attribution

The upstream model is licensed under Apache-2.0:

- [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)

This repository redistributes a GGUF-format, NVFP4-quantized conversion of Qwen3.8-27B under the same
Apache-2.0 license. Please also follow the terms and usage guidance of the upstream model card.

## Acknowledgements

Thanks to the Qwen team for the base model, NVIDIA for the ModelOpt NVFP4 quantization toolkit, and
the `llama.cpp` project for GGUF, NVFP4, MTP, and multimodal runtime support.
