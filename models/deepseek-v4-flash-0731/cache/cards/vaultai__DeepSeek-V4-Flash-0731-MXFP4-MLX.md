---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
  - mlx
  - deepseek
  - deepseek_v4
  - moe
  - mxfp4
  - mxfp8
pipeline_tag: text-generation
library_name: mlx
---

# DeepSeek-V4-Flash-0731 — MXFP4 (MLX)

MXFP4 experts + MXFP8 attention, in MLX format. This is a **bit-exact** conversion of
[deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
(305B params, 13B active). Every weight dequantizes to **exactly** the same value as
the original checkpoint — verified `maxdiff = 0.000e+00` — at **+0.12% size**.

> [!IMPORTANT]
> **This checkpoint keeps the DSpark MTP heads. Most community builds strip them.**
>
> oMLX 0.5.4rc2+ ships native DSpark Lightning MTP with Metal kernels for the
> DSpark projections, routed experts, DSA indexer scoring and block verification.
> It is inert without the embedded `dspark_*` weights. Several MLX conversions
> drop them — quantization pipelines commonly strip auxiliary heads, since they
> are dead weight unless the runtime knows how to use them. Check for `mtp.*`
> tensors in any build before relying on native MTP.
>
> On this checkpoint, native MTP measures **41.7 tok/s at 23K context** versus
> **26.1 tok/s** with speculation off — on the same machine, same harness. That
> difference is the whole reason to prefer this build.
>
> Set `mtp_enabled` for the model (oMLX stores it in
> `~/.omlx/model_settings.json`) and you are done.

> [!WARNING]
> **On oMLX 0.5.4rc2+, do not install the companion speed patches.**
> [The patch repo](https://github.com/ashhart/DeepSeekV4-Flash-0731-MXFP4-MLX)
> predates native MTP. Its speculation marker puts a second speculative decoder
> on the same KV cache — output stays fluent while repeatedly restarting the same
> sentence — and its cache patches abort rc2 with `Cache corruption not
> recoverable after retries`. Upstream is also faster at long context. The repo
> is kept as a record of how this checkpoint and its measurements were produced.

> [!TIP]
> **Check the prompt cache before anything else.** Under oMLX,
> `hot_cache_max_size` has shipped at `"0"` (disabled). Enabling it took a
> repeated 23.7K-token prompt from 51.0s to 4.6s — an 11x improvement worth more
> than every kernel change in this project combined. It lives under Memory
> Management, not the CACHE panel.

| | size | fidelity |
|---|---|---|
| Source checkpoint | 166.9 GB | — |
| **This repo** | **167.1 GB** | **bit-identical** |
| MLX affine 8-bit | 324.7 GB | 0.69% mean rel. error |
| MLX affine 4-bit (g32) | ~193 GB (projected) | 8.47% mean rel. error |

## Measured performance

Single Mac Studio, **M3 Ultra** (80-core GPU, 256 GB unified memory, 819 GB/s),
macOS 26.6, mlx 0.32.0 / mlx-lm 0.31.3 under oMLX. Weights resident: 145.5 GiB.

### Decode

Measured by *slope* (generate N1 and N2 tokens from the same prompt, take
`(N2-N1)/(t2-t1)`), which cancels prefill. M3 Ultra, 23K cached context.

| configuration | 23K context | short prompt |
| --- | --- | --- |
| **oMLX 0.5.4rc2, native DSpark MTP** | **41.7 tok/s** | 40.6 |
| speculation off | 26.1 | ~29 |

**Recommended: stock oMLX 0.5.4rc2 with `mtp_enabled`.** No patches, no markers.

For the record, the pre-rc2 patch stack on 0.5.4rc1 measured 37.6–40.7 at 23K
and 47.7–50.1 short. Ranges, not points: the same configuration measured at the
top of those ranges shortly after a reboot and at the bottom after sustained
load, with settings verified identical. Machine state moves these numbers by
~8% on this hardware; quote the low end of anything you rely on.

Published third-party figures for this model on the same hardware, all without
MTP (their checkpoints strip it): 26.2 tok/s (oMLX's own benchmark), 28.8–29.8
warm (al-engr), 35.5 short / 26.6 at 12K (community Q4).

#### For comparison: dual DGX Spark

A published dual-Spark run of this model reports **72.8 tok/s single-stream** —
faster, on two nodes with TP=2 halving per-node weight traffic. At concurrency
the picture changes:

| concurrency | aggregate | per-stream | TTFT |
|---|---|---|---|
| x1 | 72.8 | 72.8 | 237ms |
| x2 | 102.0 | 53.1 | 1.55s |
| x4 | 120.0 | 35.4 | 7.68s |
| x8 | 147.0 | 23.8 | 4.91s |

Per-stream drops to 35.4 at x4, so a single M3 Ultra is level or ahead once
there is any concurrency — and its cached prefill at 23K is ~1.2s.

> Earlier revisions of this card quoted 47.1, 2.06x, 43-48 and 938 tok/s.
> **All withdrawn** — variously measured against a corrupt KV cache, by a
> prefill-contaminated method, or on pathologically repetitive text. The patch
> repo documents each.

### If you run this under oMLX, check the prompt cache first

oMLX ships `hot_cache_max_size` at `"0"`, which means **disabled**. With "Hot
Cache Only" set (RAM, no SSD spill) nothing is cached and every request
re-prefills its whole prompt. The admin UI's CACHE panel does not show the size
field — it is under **Memory Management → "Memory Limit (In-Memory Hot Cache)"**.

On a repeated 23.7K-token prompt, setting it: **51.0s → 4.6s**. For agent
traffic that resends a growing context each turn this matters far more than any
decode tuning.

### Output head

The checkpoint leaves `head` unquantized, so it is 129280x4096 bf16 = **1.06 GB
read per forward**. Quantizing to 8-bit halves that, measured teacher-forced
over 2047 positions:

| `lm_head` | perplexity | vs bf16 | top-1 agree |
|---|---|---|---|
| bf16 (stock) | 8.3103 | — | — |
| **8-bit** | 8.3019 | **-0.10%** | **98.78%** |
| 6-bit | 8.3088 | -0.02% | 97.07% |
| 4-bit | 8.5392 | **+2.75%** | 91.26% |

8-bit is effectively lossless. 4-bit is clearly degraded.

### DSpark acceptance

Measured against ground-truth autoregressive decoding, no rollback involved, at
the drafter's trained width on clean code:

| content | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | E[prefix] |
|---|---|---|---|---|---|---|
| structured code | 100% | 96% | 96% | 92% | 96% | **4.75** |
| open-ended prose | 71% | 21% | 8% | 17% | 8% | 1.00 |

Strongly content-dependent. In a real generation loop against mixed chat traffic
it runs nearer 2.0/3, which is what the decode table reflects.

### Prefill

> A previously published 938 tok/s figure is **withdrawn** — it was measured on
> one three-line function repeated 800 times, which collapses MoE routing. Real
> content measures ~430 tok/s at 25K.

Attention builds a dense `(L, L)` mask despite a 128-token sliding window, so
prefill carries an `O(L^2)` term. The blocked/windowed path in the patch repo
removes it. The A/B below was taken on *random tokens* — worst case for the
dense path — so read the shape rather than the ratio:

| L | dense | windowed |
|---|---|---|
| 4096 | 425 | 536 |
| 8192 | 345 | 603 |
| 16384 | 183 | 595 |

The dense path halves from 8K to 16K; the windowed path does not move. The
equivalent comparison on real text has not been measured.

### Reference point

The 2x DGX Spark recipe
([tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark))
reports 84.3 tok/s peak decode on structured prompts and ~22 tok/s per stream on
mixed agent traffic, across two TP-sharded nodes. The numbers above are one
Mac Studio.

## What makes this build different

**The MTP heads are present.** `mtp.*` tensors are included, so oMLX 0.5.4rc2+
can drive native DSpark Lightning MTP against this checkpoint. That is the
difference between 41.7 and 26.1 tok/s at 23K context on an M3 Ultra. If you
want to verify before downloading 167 GB, look for `mtp.` keys in
`model.safetensors.index.json`.

**The experts are a byte transplant, not a requantization.** DeepSeek shipped
this model already quantized (166.9 GB for 305B parameters — about 4.38 bits per
weight). This repo re-containers those MXFP4 bytes into MLX's format without
touching the values: every weight dequantizes to exactly the original number,
verified `maxdiff = 0.000e+00`. Requantization error on the routed experts —
98% of the parameters — is therefore zero by construction rather than small.

Any conversion that dequantizes and re-quantizes adds error on top of DeepSeek's
own quantization. How much depends entirely on the scheme and is worth measuring
rather than assuming.

**Perplexity is not yet published for this repo.** The bit-exactness argument
above is a statement about the weights, not a measured quality claim, and should
not be read as one until a wikitext-2 number with a stated protocol appears here.

## Loading

> [!IMPORTANT]
> Earlier revisions of this repo did **not** load. `config.json` keyed its
> `quantization` overrides by *checkpoint tensor* name (`layers.0.attn.wq_a`),
> but MLX's `nn.quantize` passes `class_predicate` a *module path*
> (`model.layers.0.attn.wq_a`) — `sanitize` renames tensors on the way in and
> nothing applied the same renaming to the quantization keys. Every lookup missed
> and all 390 MXFP8 attention/shared-expert modules silently fell back to the
> MXFP4 default, failing with:
>
> ```
> ValueError: Expected shape (1024, 512) but received shape (1024, 1024)
>              for parameter model.layers.0.attn.wq_a.weight
> ```
>
> (`4096/8 = 512` lanes at 4 bits vs the true `4096/4 = 1024` at 8 bits.)
>
> **Fixed as of this revision.** `config.json` now carries both spellings, so it
> resolves under either convention. If you pinned an older revision, re-pull it.

You still need a `deepseek_v4` modeling module — stock `mlx-lm` has none
([#1233](https://github.com/ml-explore/mlx-lm/issues/1233),
[#1281](https://github.com/ml-explore/mlx-lm/issues/1281)). oMLX ships one.

### Note on the DSpark heads

`config.json` says `num_nextn_predict_layers: 1`, but the checkpoint ships
**three heterogeneous DSpark stages** under `mtp.*` (`inference/config.json` has
it right as `n_mtp_layers: 3`):

- `mtp.0` — `main_proj` (`3*dim -> dim`, fusing hidden states from layers 40/41/42
  per `dspark_target_layer_ids`) + `main_norm`, then a block
- `mtp.1` — block only
- `mtp.2` — block + `norm` + `hc_head` + `markov_head` (low-rank bigram prior)
  + `confidence_head`

This is **not** the DeepSeek-V3 MTP head (`e_proj`/`h_proj`, `enorm`/`hnorm`), so
V3-shaped MTP implementations will not load it. The field is deliberately left at
`1`: raising it to `3` makes such implementations build three V3-shaped heads and
fail. Speculative decoding needs a DSpark-aware implementation.

## Why this is lossless

The source is already stored in formats MLX supports natively — it just isn't
labelled that way:

| Source tensors | Stored as | MLX mode |
|---|---|---|
| 256 routed experts | `e2m1` nibbles + `e8m0` scale per 32 values | **`mxfp4`** (exact) |
| Attention, shared experts, indexer | `e4m3` bytes + `e8m0` scale per 128×128 block | **`mxfp8`** (exact) |
| Router gate, embed, head, norms, `hc_*` | BF16 / F32 | unchanged |

DeepSeek's expert format *is* OCP MXFP4. So rather than dequantizing to bfloat16 and
re-quantizing — which loses precision and inflates size — this conversion
**transplants the payload bytes directly** into MLX's layout:

- **MXFP4** — 4 source bytes (8 `e2m1` nibbles) reinterpret as one `uint32` lane; the
  `e8m0` scales already have exactly the per-row, per-32 granularity MLX wants, so they
  pass through untouched. **Zero transformation.**
- **MXFP8** — 4 `e4m3` bytes per `uint32` lane; the one difference is that MLX keys
  scales per 32 values while the source keys them per 128×128 block, so each block
  scale is broadcast to its 4 groups. Same values, finer indexing.

The **only** size change in the whole model is that MXFP8 broadcast: 1 byte per 32
values instead of 1 byte per 16,384. That is the entire +0.12%.

Note `mx.quantize(mode="mxfp4")` is *not* bit-exact here (~2.2% error) — it re-derives
its own scale exponent from bfloat16 rather than reusing DeepSeek's. The byte
transplant sidesteps that.

## Verification

- `e4m3` and `e8m0` decode tables match `ml_dtypes` exactly on all 256 codes.
- Both dequant paths reproduce an independent NumPy reference bit-for-bit.
- 200 randomly sampled modules across both modes: `mx.dequantize` of this repo equals
  the source dequantization **exactly**, zero mismatches.
- Full tensor accounting: every source module and passthrough tensor present.
- Quantization metadata independently re-derived from the safetensors headers via
  the packing ratio (`scales_last == lanes/4 -> mxfp4`, `lanes/8 -> mxfp8`):
  **390 MXFP8 + 35,328 MXFP4, zero mismatches, zero stray keys.**

## Layout

Source tensor naming is preserved (`layers.N.attn.wq_a`,
`layers.N.ffn.experts.E.w1`, `embed`, `head`). Each quantized module emits
`{mod}.weight` (`uint32`) and `{mod}.scales` (`uint8`, `e8m0`) — no biases, since
MXFP4/MXFP8 use pure power-of-2 scales. Experts are stored per-expert, not stacked.

`config.json` carries `quantization` with `mxfp4` as the default (the ~35k expert
modules) and explicit per-module overrides for the `mxfp8` and unquantized tensors,
in both checkpoint-tensor and MLX-module-path spellings.

## License

MIT, inherited from the base model.
