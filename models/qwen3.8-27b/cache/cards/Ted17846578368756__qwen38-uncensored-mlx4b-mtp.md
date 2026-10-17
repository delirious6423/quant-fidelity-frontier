---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - orcarouter/Qwen3.8-27B-Uncensored
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - 4-bit
  - mtp
  - speculative-decoding
  - abliterated
  - uncensored
  - qwen3.8
language:
  - en
  - zh
---
# Qwen3.8-27B-Uncensored-MLX-4bit-MTP

Uncensored Qwen3.8-27B (orcarouter abliteration lineage), MLX uniform 4-bit,
with the native MTP speculative-decoding head grafted back in.

## Provenance

- Body: `orcarouter/Qwen3.8-27B-Uncensored-MLX` (root files = 4-bit).
  Shard bytes are unchanged (hardlink-verified); only filenames/index were
  rewritten from `-of-00003` to `-of-00004`.
- MTP head: `orcarouter/Qwen3.8-27B-Uncensored-MLX` `mtp/model.safetensors`
  sidecar (15 BF16 tensors), re-indexed as `language_model.mtp.*` in
  `model-00004-of-00004.safetensors` - the embedded layout oMLX native MTP
  expects (same as Jundot/Qwen3.8-27B-oQ4e-mtp).
- tokenizer.json / vocab.json: byte-verified copies from
  `mlx-community/Qwen3.8-27B-4bit` (identical LFS objects to the orcarouter repo).
- Abliteration: 129 residual-writer tensors edited vs stock MLX 4-bit
  (dequantized Frobenius verification: 16 self_attn.o_proj + 48
  linear_attn.out_proj + 64 mlp.down_proj + embed_tokens; norms and vision
  tower bit-identical to stock).

## Verification

- Index <-> shard headers consistent (2195 tensors).
- `SHA256SUMS.json`: full sha256 of the new MTP shard; body shards referenced
  by their upstream LFS sha256.

## Use

oMLX: add to a scanned model dir, load, check engine log for
`Lightning MTP enabled ... (checkpoint layout: mtp.)`. Depth 1 (prose) or 2
(code); depth 3 gains nothing on this architecture.

Requires oMLX with indexed-sidecar/MTP detection (>=0.6.x; issues #1944/#1962/#2220 fixed).

License: Apache-2.0, inherited from Qwen/Qwen3.8-27B.
