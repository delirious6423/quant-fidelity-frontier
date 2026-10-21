---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- qwen
- q27
- quantized
- dflash2
- speculative-decoding
---

# Qwen3.8-27B-MTP -- q27 tiers (v2 recipes)

Quantized `.q27` artifacts of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
for the [q27 engine](https://github.com/signalnine/q27). Converted from the
official safetensors via BF16 GGUF; the trained MTP head is preserved (15
tensors at blk.64), so q27's speculative ladder works.

**These recipes were derived fresh for this checkpoint** by a per-tensor
sensitivity sweep (q27 BUILDLOG 2026-08-14), not ported from the Qwen3.6
tiers. The 3.6 recipes measurably do not transfer: on 3.8 the q4-head
error-cancellation family inverts, and the `ssm_out` Q8 promotion the 3.6
default/q6/q6k tiers carry is actively harmful at depth on this checkpoint
(+14% dPPL at 24K-token single-pass vs +0.26% chunked -- it compounds
through the GDN recurrence).

| tier | file | GB | wikitext PPL | HumanEval+ | pick it when |
|---|---|--:|--:|--:|---|
| q4s (v2) | `qwen38-27b-mtp-q4s.q27` | 15.70 | 7.3765 | 30/30 | max context on 24 GB cards |
| **default (v2)** | `qwen38-27b-mtp.q27` | 17.00 | 7.3121 | 30/30 | the reference tier |
| q6 (v2) | `qwen38-27b-mtp-q6.q27` | 19.76 | 7.2233 | 28/30 | quality on 24 GB, tighter context |
| q6k (v2) | `qwen38-27b-mtp-q6k.q27` | 22.52 | 7.1718 | 29/30 | best quality that fits 32 GB |

All four are smaller AND lower-PPL than the corresponding 3.6-recipe builds
of this checkpoint. Long-context tail behavior verified per-position against
a q6k reference (catastrophic counts 1-10 of 24,575 positions); needle and
per-tier canonical digests below were derived on the exact files uploaded.

Recipes (repack.py): q4s-v2 = Q4 head + Q8 attn_output; v2.0 = Q8
attn_output; q6-v2 = + Q8 ffn_down; q6k-v2 = + Q8 ffn_up. Tokenizer is
byte-identical to Qwen3.6's.

Serving notes: Qwen3.8 agentic use wants `--think` (thinking off, the model
fails hidden tests it passes with thinking on) and q27 >= commit eb4a6b0,
which auto-selects the model's trained XML tool dialect.

## DFlash2 drafter packs (q27 >= v0.11.0)

q27's production decode path since 2026-09-08 is a DFlash2 block drafter
(K=7). The drafter is
[z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
(Apache-2.0); the two files below are that checkpoint packed into q27's
weight format by `tools/dflash2_pack.py`, drafter weights only -- the
engine supplies the embedding and the head from its own tensors, so the
packs work with every tier above.

| file | GB | matmuls | notes |
|---|--:|---|---|
| `qwen38-dflash2-q8-serve.d2w` | 2.08 | Q8_G128 | production; byte-identical to the fp16 drafter on the acceptance gate |
| `qwen38-dflash2-q4-serve.d2w` | 1.21 | Q4_G64 | smaller; 1.5 to 3.6% fewer tokens per round measured |

Serve with `Q27_DFLASH2=<pack> Q27_BATCH=0 Q27_DFLASH2_RESERVE_GB=3` (single
slot; the reserve keeps the KV pool off the drafter's VRAM) and the model
card sampler; q27's `tools/launch_q27_38.sh d2-pfx` is the exact production
recipe. Measured on Claude Code traffic (12 SWE-bench instances, 2026-09-09):
207 t/s aggregate decode, 3.89 tokens per round, 96.9% prefix reuse. To
rebuild a pack yourself: `python3 tools/dflash2_pack.py <drafter_dir>
out.d2w [--q8]` (torch + safetensors + numpy).

## Gates (derived on the exact uploaded files)

Needle-in-haystack: 6/6 at ~120K tokens (q4s, default), ~100K (q6), ~40K (q6k), fp8 KV.

Per-tier greedy canonicals (`--tokens "760,6511,314,9338,369" -n 128 --ctx 2048 --spec`, RTX 5090/sm_120):

| tier | canonical md5 of the `generated:` line |
|---|---|
| q4s | `10e654eb9c9f2aeb47ea8e003aa77030` |
| default | `8cf639c25276228800cfca834db628db` |
| q6 | `067d81464ed4573b9e52841a503e111e` |
| q6k | `a965a72295800ce70ffac9113ad7ec3b` |

Note: canonicals gate *change*, not identity -- nearby tiers can share a
trajectory on this prompt (q6-v2's digest matches the superseded v1 q6/q6k
builds despite different tensors). File md5s are in `CHECKSUMS.md5`.
