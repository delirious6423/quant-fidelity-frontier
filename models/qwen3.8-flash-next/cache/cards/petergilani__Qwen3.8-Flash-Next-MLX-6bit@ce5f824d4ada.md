---
license: other
license_name: qwen-community-license-1.0
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - mlx
  - quantization
  - 6-bit
  - mtp
  - ngram
  - qwen
---

# Qwen3.8-Flash-Next — MLX 6-bit (n-gram table served from disk)

The 128-GB-Mac build of [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next):
routed experts at **6-bit g64**, dense/attention linears at 8-bit, the **vision tower and
the MTP draft head intact** — 2,899 tensors, 106.52 GB of weights, ~106 GB resident.
What makes it fit is what is deliberately **not** in it:

> **The 102.4 GB bf16 n-gram PLE table is intentionally absent.** A plain
> `mlx_lm.load("this repo")` on mlx 0.32 / mlx-lm 0.31 fails, loudly and by design, with
> `ValueError: Missing 128 parameters: model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight, …`
> That is not damage — it is the load path refusing to run a half-fed model. The table
> lives in the paired repo
> [`petergilani/Qwen3.8-Flash-Next-bf16-ngram`](https://huggingface.co/petergilani/Qwen3.8-Flash-Next-bf16-ngram)
> and is streamed row-by-row from disk by `ngram_tier.py`, bundled below.

## Why there is no n-gram table here, and what the paired repo is for

This repository ships **no n-gram table bytes at all** — that is exactly what makes the
build fit. The weights here are ~106 GB and upstream's n-gram PLE table is a further
**102.4 GB**; on a 128-GB Mac (a ~126 GB wired ceiling) both can never be resident, so the
table is left out of the download entirely and is read from NVMe while running.
`ngram_tier.py` (bundled here) does that: a decode step pulls the few rows it needs
straight out of the table file through a 1 GiB row cache.

The table itself is what the paired repository
[`petergilani/Qwen3.8-Flash-Next-bf16-ngram`](https://huggingface.co/petergilani/Qwen3.8-Flash-Next-bf16-ngram)
is for. It is not a model and serves no other purpose: it is upstream's table, published
on its own so it can be streamed rather than loaded
(128 shards × 2,500,012 × 160 bf16, 320,001,536 rows, 102,400,491,520 bytes, byte-identical
to upstream). **You need both repos to run this model.**

Measured quality of this configuration (6-bit experts, n-gram rows served from disk):
**perplexity 4.4765** (2026-08-30).

The tier buys **context headroom, never decode speed**: a decode step touches 16 rows
(~5 KB) per token through a 1 GiB set-associative row cache; first contact is a real
NVMe read.

## How to run it (pairing recipe — verified, in this exact order)

`pip install mlx-lm`, download **both** repos (e.g. `huggingface-cli download`), and give
Metal the headroom (`mx.set_wired_limit(int(122e9))`, or `sudo sysctl iogpu.wired_limit_mb=126000`).
Then — this is the sequence that ran before this repo was published:

```python
import mlx.core as mx
from pathlib import Path
import ngram_tier as nt                      # bundled in this repo
from mlx_lm import generate
from mlx_lm.utils import load_model, load_tokenizer

MODEL_DIR = "Qwen3.8-Flash-Next-MLX-6bit"        # local snapshot of THIS repo
SRC_DIR   = "Qwen3.8-Flash-Next-bf16-ngram"      # local snapshot of the paired table

tier = nt.open_tier(MODEL_DIR, source="bf16", src_dir=SRC_DIR, out="bf16", threads=8)
model, _ = load_model(Path(MODEL_DIR), lazy=True, strict=False)  # absent table BY DESIGN
tok = load_tokenizer(Path(MODEL_DIR))
tier.install(model)                              # patch the gather BEFORE any eval
assert tier.byte_identity()["ok"], "source rows do not match the source - stop"
tier.drop()                                      # refuse the unbound table arrays
mx.eval(model.parameters())                      # exactly ONE forced eval
print(generate(model, tok, "The capital of France is", max_tokens=24, temp=0.0))
```

The order is load-bearing: the load is lazy, and `install`+`drop` before that single
`mx.eval` are what keep the table out of RAM. **Never run `strict=False` without
installing the tier first** — `strict=False` silences the helpful `Missing 128
parameters` error, and `strict=False` + no tier is the one combination that can serve
random-initialized n-gram rows. The `byte_identity` gate before `drop` exists so that
can never happen silently: it re-reads real source rows and compares to direct preads.

## Verification receipts (2026-09-08, mlx 0.32.3.dev / mlx-lm 0.31.3, 128 GB M-series)

- every shipped tensor sha256-verified byte-for-byte against the pre-publication build of
  this same conversion (2,899/2,899);
- `open_tier` geometry read from the paired repo: 128 × 2,500,012 × 160 bf16 = exactly
  102,400,491,520 bytes; `byte_identity` **passed on 64 of 128 shards × 4,096 random
  rows each**, including its row+1 non-vacuity controls;
- one forced eval after the drop: 105.63 GB active, no OOM, ~20 s on NVMe;
- greedy sanity, 18.9 s: `"The capital of France is" -> "Paris. The capital of Germany is
  Berlin. The capital of Italy is Rome. The capital of Spain is Madrid. The"`.

## Contents & config notes

- `model-00001/00004.safetensors` rebuilt without the n-gram table tensors;
  `model-00005..00014.safetensors` and all small files byte-identical to the conversion
  they came from; `model.safetensors.index.json` rebuilt (2,899 tensors);
- `config.json` is this conversion's live config (YaRN ×4, 1M rope) with the n-gram
  table's quantization-map entries removed — left in place they would tell the loader to
  quantize embeddings this repo deliberately ships no weights for. The converter's
  `config_default`/`config_1M` working variants are not shipped;
- `qwen4_exp.py` is the model definition (`model_file` in the config), byte-identical to
  the one this conversion ships; `ngram_tier.py` is the disk-tier consumer used above.

## Provenance & license

Quantized and packaged by us from
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): our own MLX
conversion, routed experts at 6-bit g64 and dense/attention linears at 8-bit, with the
n-gram PLE table externalized to the paired repository (see above) instead of shipped
here. License: **Qwen Community License 1.0** (`LICENSE` ships verbatim).