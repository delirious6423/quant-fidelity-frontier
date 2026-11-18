---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
inference: false
tags:
- quantization
- mxfp4
- trellis
- graft
---

# DeepSeek-V4.1-Flash-1017, GRAFT 3-bit routed experts

The routed experts of DeepSeek-V4.1-Flash-1017 (revision `1bae0e1088993b9b82de08f59de5be5a34e42170`) quantized with
GRAFT (Grid-Aligned FP4 Trellis) at 3.007 bits per weight. The recipe and the calibration corpora are the same
as for [coolbho3k/DeepSeek-V4.1-Flash-GRAFT-3bpw](https://huggingface.co/coolbho3k/DeepSeek-V4.1-Flash-GRAFT-3bpw),
which quantizes the earlier V4.1-Flash checkpoint; the calibration activations were recaptured on 1017.

Paper: [GRAFT: Grid-Aligned FP4 Trellis Quantization for FP4-Native Language Models](https://github.com/coolbho3k/GRAFT)

## Quality

The model-level evaluation against the 1017 checkpoint (KL, excess NLL) has not been run yet. Per expert, the
held-out reconstruction error relative to each release's own source checkpoint is the same as in the earlier release:

| | this repo (1017) | earlier release |
|---|---|---|
| bits per weight (routed experts) | 3.0066 | 3.0062 |
| held-out expert error, median ratio over the same 15,358 experts | 1.001 | 1 |

The earlier release measured KL 0.0369 and excess NLL 0.0085 nats against its source model.

## Contents

- `model-00001` ... `model-00050-of-00051.safetensors`: every layer's routed experts in GRAFT-MX format, plus the
  layers' other weights. `model-00051-of-00051.safetensors` and every tensor outside the routed experts are copied
  unchanged from DeepSeek-V4.1-Flash-1017.
- `engrams/` holds only the config and index. Its two files, `engram-layer-01.safetensors` and
  `engram-layer-14.safetensors` (101 GB each, over the Hugging Face file size limit), hold the tensors
  `layers.L.engram.embed.weight` and `layers.L.engram.embed.scale` unchanged from DeepSeek-V4.1-Flash-1017:
  `model-00047-of-00048.safetensors` for L = 1 and `model-00048-of-00048.safetensors` for L = 14. Copy those two
  tensors of each shard into the matching file.
- `draft/`: the 1017 DSpark speculative-decoding draft (`mtp.0`-`mtp.2`), copied unchanged from the 1017
  checkpoint.
- `config.json`: `quantization_config.codebook = "graft-mx"`; `quantization_config.graft.flagged` lists the matrices
  that carry block flags.

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

1017 adds an FP8 (UE8M0, block 32) round trip of the attention output before `wo_a`; a runtime must apply it as the
1017 reference code does.

## License

MIT, as the base model (Copyright (c) 2023 DeepSeek; see `LICENSE`).
