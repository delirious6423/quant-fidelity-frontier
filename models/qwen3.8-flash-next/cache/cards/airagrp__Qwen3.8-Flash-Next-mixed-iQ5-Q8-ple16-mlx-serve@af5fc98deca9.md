---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- qwen4_exp
- moe
- sparse-attention
- ngram-embedding
---

# Qwen3.8-Flash-Next for mlx-serve, mixed iQ5/iQ8 experts, Q8 heads, bf16 n-gram table

Repo: [`airagrp/Qwen3.8-Flash-Next-mixed-iQ5-Q8-ple16-mlx-serve`](https://huggingface.co/airagrp/Qwen3.8-Flash-Next-mixed-iQ5-Q8-ple16-mlx-serve)
(188.1 GiB: 91.9 GiB trunk shards + 95.4 GiB `ngram_table.bin` + 0.8 GiB vision tower).

mlx-serve pack of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next),
the Qwen4 preview architecture (`model_type: qwen4_exp`). Built and measured on an
M5 Max 128 GB; **~99.6 GB resident — the roomiest 128 GB pack of the family, and
the only one that fits the native 262,144-token context alongside an 8-bit KV
cache.** Includes the MTP head and the vision tower (image and video input).

Three dials make this the practical 128 GB pack:

- **iQ5 experts on layers 0–39, iQ8 on the last 8 (the "tail").** Early expert
  layers are 5-bit (imatrix-calibrated, group 64); the last 8 layers keep 8-bit
  experts, where residual error feeds straight into the output.
- **Everything else is 8-bit** — `embed_tokens`, `lm_head`, the whole MTP head
  (banks and projections), and all attention/GDN/indexer projections. The
  reasoning: every activation entering the output head has already passed
  through 48 layers of 8-bit projections, and a same-eval head-to-head showed
  Q8 heads tie bf16 heads on held-out next-token prediction (92.1% vs 92.4%
  top-1, within noise) while saving 3.5 GB of resident memory. That 3.5 GB is
  exactly the difference between "262k context does not fit" and "it fits."
- **The n-gram table ships dense bf16** (`ngram_table.bin`, 102.4 GB). An A/B
  pair against an otherwise-identical pack with an 8-bit table measured the
  bf16 table at **+2.5 points top-1** (92.1% vs 89.6%) — the one place 8-bit
  measurably hurts, so it stays 16-bit. It is mmapped, dequantized per token
  on the CPU, and never becomes resident, so the table costs page cache
  instead of memory.

```bash
mlx-serve --model airagrp/Qwen3.8-Flash-Next-mixed-iQ5-Q8-ple16-mlx-serve --serve
```

## What is different about this model

This is not a Qwen3.5-style pack. Three things around the usual GDN + MoE trunk:

- **Gated residual streams.** The residual is 4 streams wide (4 x 2560). Every
  block reads a sigmoid-mixed average of the normalized streams and writes back
  through per-stream scalar gates. The final mixer replaces the usual final norm.
- **N-gram embedding (51B parameters).** A second embedding table indexed by
  hashed bigrams and trigrams of the token ids: 16 heads, each a prime-sized
  bucket space of ~20M rows, 160 dims per row, injected once before layer 1.
  It is a lookup, no compute, which is why Qwen quotes the model as 125B: the
  full checkpoint is 125B trunk + 51B n-gram + 4B MTP = 180B (360 GB bf16).
- **Qwen Sparse Attention.** Past 2048 tokens each attention layer only reads
  the 512 most relevant 4-token blocks per query (picked by a small indexer),
  plus the query's own partial block. Attention cost stays flat with context.
  Native 262k context.

## How this pack stores the n-gram table

The 51B table is NOT in the safetensors shards. It is one merged bf16 table in
`ngram_table.bin` (102.4 GB, safetensors format, `.bin` so nothing mlx-loads
it). mlx-serve mmaps the file and, per token, reads the 16 rows it needs on the
CPU and uploads only the resulting 2560-vector. The table never becomes
resident: its cost is page cache, which the OS evicts as needed.

Expected effect: decode speed unchanged (16 tiny reads against a ~20 ms step),
cold-cache prefill of very long prompts may pay up to ~1 s per 8k tokens of
random reads on the SSD, warm cache is free. No user-space cache is needed,
the page cache already is an LRU over exactly this access pattern. Budget disk
headroom: the pack is 188.1 GiB (202 GB) on disk.

## Widths

| tensors | width |
|---|---|
| routed experts (512 x 48 layers) | 5-bit, group 64, imatrix-calibrated on layers 0–39; 8-bit, group 64 on layers 40–47 |
| embed_tokens | 8-bit, group 64 |
| lm_head | 8-bit, group 64 |
| MTP head — expert banks and projections | 8-bit, group 64 |
| attention, GDN, hyper-connections, indexer, shared experts | 8-bit, group 64 |
| n-gram table | bf16 (16-bit, mmap, non-resident) |
| routers, inject gates, norms, convs, SSM state | bf16 |
| vision tower | bf16 (`model-vision.safetensors`, ~0.9 GB) |

Every `(1 + w)` RMSNorm has the `+1` folded into the stored weight; depthwise
convs are transposed to MLX's `[C, K, 1]`; `experts.gate_up_proj` is split into
`switch_mlp.gate_proj` / `up_proj`. The 5-bit expert layers are
**imatrix-weighted affine (iQ-MLX)** — per-input-channel activation statistics
collected on the exact bf16 checkpoint over 162k tokens of agent-transcript,
code (SWE-bench), prose and math calibration text. The same statistics drive
the 8-bit tail, so the whole pack shares one calibration pass.

## Measured quality

Held-out evaluation from the mlx-serve test suite
(`tests/qwen38_flash_next_score.py`): bf16 next-token logits were recorded at
392 positions of held-out windows the calibration never saw; a served pack is
queried greedy (temperature 0, top-1024 logprobs) per position and scored
against the bf16 argmax. Numbers are row-matched against this pack
(n = 361–369). `ref_p` is the bf16 probability mass on the pack's pick; KLD is
full-support against the bf16 distribution.

| pack | experts / heads / n-gram table | top-1 | KLD |
|---|---|---:|---:|
| **this pack** | iQ5 + iQ8 tail / **8-bit** / bf16 | 92.1% | 0.072 |
| [mixed-iQ5-tail8-bf16heads](https://huggingface.co/airagrp/Qwen3.8-Flash-Next-mixed-iQ5-tail8-bf16heads-mlx-serve) | iQ5 + iQ8 tail / bf16 / bf16 | 92.4% | 0.068 |
| same widths, **no imatrix** (plain affine) | iQ5 + iQ8 tail / 8-bit / bf16 | 91.5% | 0.073 |
| same trunk, **8-bit n-gram table** | iQ5 + iQ8 tail / 8-bit / 8-bit | 89.6% | 0.081 |
| 4-bit-embed mixed pack (local) | iQ5 + iQ8 tail / 4-bit embed, 8-bit head / bf16 | 92.9% | 0.071 |
| ddalcu iQ-MLX-4.7bpw | 4-bit / 8-bit / 4-bit | 89.8% | 0.129 |

Readings: bf16 heads buy a KLD nicety (0.068 vs 0.072) but no reliable top-1 —
within noise of this pack, and a 4-bit `embed_tokens` scored *higher* top-1.
Dropping imatrix costs ~1 point at identical byte widths. The 8-bit n-gram
table is the worst single change measured (−2.5 points), which is why this
pack keeps it dense.

## Serving notes

- **Memory and context.** ~99.6 GB resident plus KV. With `--kv-quant 8` the
  full native 262,144-token context fits on a 128 GB machine — verified
  end-to-end: a 240,560-token prompt prefilled in ~2.5 min and a needle at 60%
  depth was retrieved exactly. Without KV quantization the auto-sized context
  is ~99k tokens. One very long session at a time.
- **MTP.** The checkpoint's own 1-layer speculative head is loaded from the
  pack and works (`--mtp` or per-request `"enable_mtp": true`); its banks and
  projections are both 8-bit here. Drafts are always verified, so MTP never
  changes the output distribution.
- **Long prompts** (past ~64k) want a smaller `--prefill-chunk` because the
  sparse-attention selection is built per chunk.
- **Thinking** is on by default (`"enable_thinking": false` turns it off).
  Tools use Qwen3.8's XML call format; mlx-serve parses and schema-coerces it.
- **Images and video** go through the Qwen3-VL-style tower (`model.visual.*`,
  dense bf16). MTP is declined on image turns (serial decode).
- Set `temperature: 0` explicitly when you need deterministic output; the
  server default samples at 1.0.

## Conversion

`tests/convert_qwen38_flash_next.py` in the mlx-serve repo. Flags:

```bash
python tests/convert_qwen38_flash_next.py --src <bf16 checkpoint> --dst <pack> \
  --imatrix imatrix.safetensors \
  --bits 8 --gate-up-bits 5 --down-bits 5 --tail-layers 8 \
  --nonexpert-bits 8 --embed-bits 8 --ngram-bits 16 --jobs 16
python tests/convert_qwen38_flash_next.py --src <bf16 checkpoint> --dst <pack> --add-vision
```

(`--lm-head-bits` and `--mtp-bits` default to the non-expert width, 8-bit
here.) Converted from the 360 GB bf16 checkpoint in 15 minutes on an M5 Max,
streaming shard by shard; the imatrix pass (32 min, 162k tokens) and the
held-out reference logits are reusable across re-quants of this architecture.
Per-shard verification: trunk `down_proj` packed rows are 100 U32 words
(5-bit) on layers 0–39 and 160 (8-bit) on 40–47; `embed_tokens`, `lm_head`
and the MTP `switch_mlp` banks are U32 at 8-bit widths; the table header
reads `{"bits": "16"}`.
