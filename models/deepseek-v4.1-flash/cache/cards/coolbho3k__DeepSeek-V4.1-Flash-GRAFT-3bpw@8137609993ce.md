---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
inference: false
tags:
- quantization
- mxfp4
- trellis
- graft
---

# DeepSeek-V4.1-Flash, GRAFT 3-bit routed experts

The routed experts of DeepSeek-V4.1-Flash quantized with GRAFT (Grid-Aligned FP4 Trellis) at 3.006 bits per weight.
GRAFT codes MXFP4 checkpoints below 4 bits by keeping the source grid instead of rotating it away, and it decodes back
to MXFP4 blocks with a per-column scale and a sparse tail correction.

Paper: [GRAFT: Grid-Aligned FP4 Trellis Quantization for FP4-Native Language Models](https://github.com/coolbho3k/GRAFT)

| | GRAFT (this repo) | EXL3 3 bpw |
|---|---|---|
| held-out expert error (relative to EXL3) | 0.63x | 1 |
| excess NLL over the native model (nats) | 0.0085 | 0.0131 |
| KL to the native model | 0.0369 | 0.0469 |
| top-1 agreement | 92.5% | 91.7% |

## Contents

- `model-00001` ... `model-00050-of-00051.safetensors`: every layer's routed experts in GRAFT-MX format, plus the
  layers' other weights. `model-00051-of-00051.safetensors` and every tensor outside the routed experts are
  byte-identical to [coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-EXL3-3bpw).
- `config.json`: `quantization_config.codebook = "graft-mx"`; `quantization_config.graft.flagged` lists the matrices
  that carry block flags.
- `draft/` and `engrams/` hold only the configs and indexes. The drafter weights and the engram tables are unchanged
  from the EXL3 repository; copy them from there into the same folders.

Each routed-expert matrix `layers.L.ffn.experts.E.wN` (N = 1, 2, 3) is stored as:

| tensor | dtype | shape | content |
|---|---|---|---|
| `trellis` | int16 | `[k/16, n/16, 48]` | one 768-bit trellis bitstream per 16x16 tile (read as little-endian 32-bit words) |
| `suh` | float16 | `[k]` | header: the first 16 values reinterpreted as 8 float32 `(qa0, qb0, d, 0, qa1, qb1, 0, 0)`, repeated |
| `svh` | float16 | `[n]` | per-output-column scale |
| `mul1` | int32 | scalar | hash multiplier `0x83DCD12D` |
| `graft_flags` | int32 | `[k/32, n/32]` | flagged matrices only: bit `c mod 32` of word `[r/32, c/32]` flags the 32-row block of column `c` |

`k` is the input dimension and `n` the output dimension, so the decoded matrix is the transpose of the original
`weight` tensor. A reference decoder will be published separately.

## License

MIT, as the base model (Copyright (c) 2023 DeepSeek; see `LICENSE`).
