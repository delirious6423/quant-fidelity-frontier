---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-serve
- qwen4_exp
- moe
- sparse-attention
- ngram-embedding
- vision
- multimodal
- image-text-to-text
- video-text-to-text
---

# Qwen3.8-Flash-Next for mlx-serve (6-bit experts, 8-bit rest, 8-bit PLE)

mlx-serve pack of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
the Qwen4 preview architecture (`model_type: qwen4_exp`). About 96 GB
resident (128 GB Macs). Includes the MTP head and the vision tower (image and video input).

```bash
mlx-serve --model airagrp/Qwen3.8-Flash-Next-mixed-6-8-ple8-mlx-serve --serve --kv-quant 8 --mtp
```

Measured on an M5 128 GB (mlx-serve 26.9.2, mlx 0.32.2): ~96 GB peak
resident, decode ~54 tok/s serial / ~62 tok/s with MTP on code, prefill
~1.6-1.9k tok/s sustained (31k-token prompt). Short prompts pay ~1.5 s of
fixed overhead (PLE mmap walk + sparse-attention block selection).

## What is different about this model

This is not a Qwen3.5-style pack. Three things around the usual GDN + MoE trunk:

- **Gated residual streams.** The residual is 4 streams wide (4 x 2560). Every
block reads a sigmoid-mixed average of the normalized streams and writes back
through per-stream scalar gates. The final mixer replaces the usual final norm.
- **N-gram embedding (51B parameters).** A second embedding table indexed by
hashed bigrams and trigrams of the token ids: 320,001,536 rows, 160 dims per
row, injected once before layer 1. It is a lookup, no compute, which is why
Qwen quotes the model as 125B: the full checkpoint is 125B trunk + 51B n-gram
+ 4B MTP = 180B (360 GB bf16).
- **Qwen Sparse Attention.** Past 2048 tokens each attention layer only reads
the 512 most relevant 4-token blocks per query (picked by a small indexer),
plus the query's own partial block. Attention cost stays flat with context.
Native 262k context.

## How this pack stores the n-gram table

The 51B table is NOT in the safetensors shards. It is one merged 8-bit table
in `ngram_table.bin` (57.6 GB, safetensors format, `.bin` so nothing
mlx-loads it). mlx-serve mmaps the file and, per token, dequantizes the 16 rows
it needs on the CPU and uploads only the resulting 2560-vector.
The table never becomes resident: its cost is page cache, which the OS evicts
as needed. That is the difference between this pack and mlx-lm style packs
that ship the table as 128 quantized tensors and load it onto the GPU
(+~55-64 GB resident at 8-bit for this pack's table width).

Expected effect: decode speed unchanged (16 tiny reads against a ~20 ms step),
cold-cache prefill of very long prompts may pay up to ~1 s per 8k tokens of
random reads on the SSD, warm cache is free. No user-space cache is needed,
the page cache already is an LRU over exactly this access pattern.

## Widths

| tensors | width |
|---|---|
| routed experts (512 x 48 layers, the 121B) | 6-bit, group 64 |
| attention, GDN, hyper-connections, indexer, shared experts | 8-bit, group 64 |
| lm_head | 8-bit, group 64 |
| embed_tokens | 4-bit, group 64 |
| n-gram table | 8-bit, group 32 (row width 160) |
| routers, inject gates, norms, convs, SSM state | bf16 |
| MTP head | experts 6-bit group 64, projections 8-bit group 64 |

Every `(1 + w)` RMSNorm has the `+1` folded into the stored weight; depthwise
convs are transposed to MLX's `[C, K, 1]`; `experts.gate_up_proj` is split into
`switch_mlp.gate_proj` / `up_proj`. The vision tower ships dense bf16 in
`model-vision.safetensors` (~0.9 GB).

An 8-bit MTP-expert variant was also built and tested: +0.63 GB with no
measurable quality, speed, or memory difference, so this pack keeps 6-bit MTP.

## Serving notes

- **Memory.** ~96 GB resident plus KV cache. mlx-serve sizes the context to
what fits; `--kv-quant 8` halves the cache (verified, no quality loss).
- **MTP.** The checkpoint's own 1-layer speculative head is loaded from the
pack and works (`--mtp` or per-request `"enable_mtp": true`). Measured +13%
on code via the server (61.7 vs 54.4 tok/s); at parity on short prose.
`--mtp-head-kv-quant` is neutral on short prompts and saves head KV on long ones.
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
Tools use Qwen3.8's XML call format; mlx-serve parses and schema-coerces it.
- **Images and video** go through the Qwen3-VL-style tower (`model.visual.*`,
dense bf16). MTP is declined on image turns (serial decode).

## Conversion

`tests/convert_qwen38_flash_next.py` in the mlx-serve repo, run from the local
BF16 checkpoint:

```bash
python3 tests/convert_qwen38_flash_next.py --src <bf16 dir> \
  --dst <pack> --bits 6 --nonexpert-bits 8 --ngram-bits 8
python3 tests/convert_qwen38_flash_next.py --add-vision --src <bf16 dir> --dst <pack>
```

105.2 GB trunk + 57.6 GB `ngram_table.bin`, ~11 min on an M5.
