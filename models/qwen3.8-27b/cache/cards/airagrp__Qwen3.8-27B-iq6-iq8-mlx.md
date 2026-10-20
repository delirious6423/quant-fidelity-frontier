---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx-vlm
pipeline_tag: image-text-to-text
language: en
tags:
- mlx
- mlx-vlm
- quantized
- imatrix
- speculative-decoding
- dflash
- vision-language
---

# Qwen3.8-27B-iq6-iq8-mlx (M variant)

Community MLX quantization of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(community release, `Qwen3_5ForConditionalGeneration` architecture — 64 layers:
48 gated-delta-net `linear_attention` + 16 `self_attention`, vision tower, 248k vocab).
**27.4 GiB** (mixed 6/8-bit affine, group size 32) vs 55.6 GiB bf16.

> M = "max-fidelity" variant: 6-bit MLPs with 8-bit final layers, ~27.4 GiB —
> measured **98.6% of bf16 benchmark accuracy retained** (one benchmark item
> difference overall). The sibling S variant (`Qwen3.8-27B-iq4-iq6-mlx`) is
> ~4 GiB smaller and ~1.7–2× faster per request at a −2.8pp accuracy cost.

## Recipe

Per-layer mixed precision, quantized with an imatrix-weighted affine quantizer
(weighted least-squares refit + factor ladder per group, llama.cpp
`make_qkx2_quants`-style), stock-MLX packed (bf16 scales/biases, `u*s + b`):

| component | bits | group size |
|---|---|---|
| embeddings + lm_head | bf16 | — |
| attention linears (`linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj}`, `self_attn.{q,k,v,o}_proj`) — all 64 layers | **8** | 32 |
| MLP (`mlp.{gate,up,down}_proj`) — layers 0–59 | **6** | 32 |
| MLP — layers 60–63 (final 4 layers) | **8** | 32 |
| vision tower | bf16 | — |

- Imatrix recomputed on **2,515,992 tokens** from the DS4 calibration dataset
  ([antirez/ds4 · gguf-tools/imatrix](https://github.com/antirez/ds4/tree/main/gguf-tools/imatrix);
  4690 records re-rendered with *this model's* chat template) — exact-f32
  per-input-channel activation collection on the quantization target itself,
  collected in-house with MLX (no llama.cpp/unsloth or pre-computed GGUF imatrix).
- **MTP heads have been removed** from the shipped weights — text/vision
  serving is unaffected; speculative decoding pairs instead with the companion
  DFlash2 drafter (see Serving).
- Loads as stock MLX/mlx-vlm (no custom kernels). MLX 0.32.2, mlx-vlm 0.6.17.

## Quality (wikitext2-test, 145×2048 windows)

| model | ppl | Δ ppl vs bf16 | KLD bits/tok | wSSE vs stock quant | size |
|---|---|---|---|---|---|
| bf16 reference | 7.0468 | — | — | — | 55.6 GiB |
| **this (M)** | 7.0469 | **+0.001%** | 0.0050 | **−37.4%** | 27.4 GiB |
| [Qwen3.8-27B-iq4-iq6-mlx (S)](https://huggingface.co/airagrp/Qwen3.8-27B-iq4-iq6-mlx) | 7.0607 | +0.20% | 0.0222 | −43.1% | 23.4 GiB |

## Intelligence benchmark

500 items (stratified selection, frozen seed 731): MMLU 200, ARC-Challenge 200,
GSM8K 100 · temp 0, reasoning_effort xhigh, strict `FINAL:` extraction · served
via oMLX with an 8-way concurrent client; DFlash2 drafter (bf16) attached,
block size 7.

| model | MMLU | ARC-C | GSM8K | total | gen tok/s (ARC / GSM) | median wall (s) |
|---|---|---|---|---|---|---|
| **this (M)** | 91.0% | 98.5% | 97.0% | **95.2%** | 5.9 / 6.8 | 21–39 |
| Qwen3.8-27B-iq4-iq6-mlx (S) | 89.0% | 94.0% | 97.0% | 92.6% | 10.2 / 13.7 | 13–18 |
| bf16 | 91.0% | 99.0% | 97.0% | 95.4% | 6.0 / 7.3 | 23–38 |

Draft/verify behavior (same drafter, block 7): **M 3.92 tok/cycle @ 78.6%
accept**, vs 4.75 @ 83.1% (S) and 3.81 @ 76.8% (bf16) — M matches bf16's
acceptance profile within a point.

Caveats: single seed, 500 items; total difference vs bf16 is one item
(95.2% vs 95.4%); throughput columns under 8-way batching reflect
shared-engine per-request rates (environment-specific).

## Serving

With mlx-vlm:

```python
from mlx_vlm import load, generate
model, processor = load("airagrp/Qwen3.8-27B-iq6-iq8-mlx")
```

With oMLX multi-model serving + DFlash2 speculative decoding, attach the companion drafter:

```json
{
  "mtp_enabled": false,
  "dflash_enabled": true,
  "dflash_draft_model": "<local drafter dir>",
  "dflash_draft_quant_enabled": false,
  "dflash_block_size": 7
}
```

Point `dflash_draft_model` at your local copy of the drafter — download
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
if needed. Keep the drafter unquantized (bf16). Plain mlx-vlm text/vision
serving of this artifact needs no drafter.

## Acknowledgements

- Base model: Qwen team — [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0).
- Speculative decoding drafter: [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
  (bf16, kept unquantized) — a separate download, needed only when you enable
  DFlash2 speculative decoding.
