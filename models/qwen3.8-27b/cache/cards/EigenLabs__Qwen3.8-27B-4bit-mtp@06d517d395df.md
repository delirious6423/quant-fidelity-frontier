---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- qwen3.8
- vision-language
- speculative-decoding
- mtp
- quantized
- bfloat16
---

# Qwen3.8-27B MLX W4/g64 + Embedded W4 MTP

This is [`EigenLabs/Qwen3.8-27B-4bit`](https://huggingface.co/EigenLabs/Qwen3.8-27B-4bit)
at exact revision `301e9e2767fd0efcfab7883004720ba3c9a552a1` with the
previously separate MTP proposal head
([`EigenLabs/Qwen3.8-27B-MTP-4bit`](https://huggingface.co/EigenLabs/Qwen3.8-27B-MTP-4bit)
at exact revision `329261c5e0b3f9c233485e682cb3b67b88c20a55`) **embedded in the
same indexed safetensors weight set** under the `mtp.*` prefix.

- Target and vision tower tensors: byte-identical to the pinned target
  revision.
- MTP head tensors: byte-identical to the pinned head revision (W4/g64
  affine, BF16 scales/biases, one full-attention layer, ~239 MB), re-keyed
  from the standalone layout (`fc.weight`, `layers.0.*`, …) to `mtp.*`.
- Declared via `mtplx_mtp` / `mtplx_mtp_quantization` in `config.json`
  (`prefix: "mtp."`, `block_size: 3`). The head shares the target embedding
  and LM head; the target backbone remains authoritative for token
  acceptance and final output.

This repository supersedes the external-head pairing: the draft/target
revision pins that previously lived in a README are now structural — one
manifest, one weight hash, nothing to drift.

## Measured (M4 Max, 546 GB/s, temperature 0)

| Configuration | decode tok/s |
|---|---|
| target only | 20.3–20.6 |
| embedded MTP | 28–31.5 |
