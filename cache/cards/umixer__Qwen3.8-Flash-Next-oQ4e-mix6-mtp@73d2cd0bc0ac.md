---
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
- mlx
- safetensors
- qwen4_exp
- oq
- quantized
- mtp
- mixed-precision
---

# Qwen3.8-Flash-Next-oQ4e-mix6-mtp

[繁體中文版 →](README_zh.md)

**Goal: oQ4e speed and memory footprint, oQ5e-or-better quality reaching toward
"Q6-class".** Runs at ~73–75 GB resident with ~82→66 tok/s decode on a 128 GB Apple
Silicon host — the speed/RAM economics of a 4-bit build, at an accuracy tier
equal to or above uniform oQ5e.

*How:* a mixed-precision MLX build of Qwen3.8-Flash-Next, assembled by
**component-wise byte-copy** from Jundot's native oQ quantizations —
**zero re-quantization in the entire package**. Every tensor is byte-identical
to its source, so quality is traceable, not re-derived.

> **v2 rebuild:** the middle-40-layer donor is switching to
> Jundot's **new-mechanism oQ4e** ([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385):
> importance-matrix calibration, 1024 samples × 512 tokens, plus a weighted
> least-squares refit of every group's scale/bias). Same U-shaped allocation,
> same name — the trunk 4-bit tensors are now LSQ-refit instead of plain absmax,
> and the 6-bit skeleton comes from Jundot's oQ6e (also PR 4385). Both source
> packs carry the new quantizer; the byte-copy property is unchanged.

## Why this build exists

Uniform-bit quantization pays accuracy where it isn't needed (middle layers) and
loses it where it hurts (first/last layers, n-gram table, attention).
Layer-sensitivity literature (SliderQuant arXiv:2603.25284) shows
quantization sensitivity is U-shaped across depth: expensive edges, cheap
middle. This package prices that curve with **native
bits** — the high-value components are lifted verbatim from higher-bit packs
instead of re-quantized.

## Recipe & comparison

| Component | This build (v2) | uniform oQ4e | Resident Δ vs oQ4e | Source (byte-identical) |
|---|---|---|---|---|
| Trunk MoE experts, layers 0–3 & 44–47 (U-shaped edges) | **6-bit g64** (LSQ-refit) | 4-bit | **+4.8 GB** (hot, fully resident) | [Jundot/Qwen3.8-Flash-Next-oQ6e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ6e-mtp) (PR 4385) |
| Trunk MoE experts, middle 40 layers | 4-bit g64, **imatrix + LSQ refit** | 4-bit (v1, absmax) | 0 | [Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) (PR 4385) |
| Attention / GDN / hyper-connections / conv | **all 48 layers ≥6-bit**: 6-bit on 40 layers, 8-bit at L0/L31/L32/L35/L36/L39/L44/L46 | mostly 4-bit, sensitivity layers raised to 5–8 | **+0.4 GB** (hot, resident) | oQ6e |
| N-gram (PLE) embedding table | **6-bit g32** (41.7 GiB) | 4-bit g32 | ≈0 (SSD-offloaded on hosts; page-cache only) | oQ6e |
| Shared experts (all 48 layers) | 8-bit g128 | 8-bit g128 | 0 | oQ6e |
| embed_tokens / lm_head | 8-bit g64 | 8-bit g64 | 0 | oQ6e |
| MTP drafter (experts 6-bit, attention/hyper-conn 8-bit) | **6/8-bit** | 4-bit experts / 6-bit attn | **+0.6 GB** (hot, resident) | oQ6e |
| Norms / routers / vision tower | bf16, unquantized | bf16, unquantized | 0 | oQ6e |
| **Total at load** | **75.49 GB** | **69.53 GB** | **+5.96 GB** | `engine_pool` actual, same host (v2 measured) |

Every component where this build differs from uniform oQ4e is one the router or
the hot path touches every token — the extra 5.8 GB of resident weights buys
6-bit exactly where quantization error is realized.

Skeleton: `bits=4, group_size=64, mode=affine`; every high-bit component is a
per-tensor entry in `config.json` (mirrored across `quantization` and
`quantization_config`). Build integrity verified: safetensors packed geometry
matches all switch-mlp config declarations (0 mismatches); swapped tensors
verified `mx.array_equal` against source packages; untouched shards hardlinked
from base (bits not written, bytes proven equal).

## Source packs — new-mechanism quantization (PR 4385)

Both donors were re-released on 2026-10-11 with Jundot's new quantizer
([jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385)): sensitivity-
measured bit allocation, imatrix calibration (1024 samples × 512 tokens), and a
**weighted least-squares refit of each group's scale/bias** replacing plain
absmax rounding.

| Pack | Effective bits | Package size | Bit ladder |
|---|---|---|---|
| [Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) (new) | **4.724** | **99.0 GiB** (identical size & bit ladder to v1; the 10/11 release re-bakes every quantized tensor — probed: trunk weights ~86–88% of words differ, trunk scales 97.6%, shared experts ~58%, MTP experts ~87%; unquantized bf16 tensors byte-identical) | 4-bit g64 trunk 69.3% · 4-bit g32 n-gram 28.4% · 5/6/8-bit sensitivity layers ~1.9% · BF16 0.3% |
| [Jundot/Qwen3.8-Flash-Next-oQ6e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ6e-mtp) | **6.691** | **140.2 GiB** | 6-bit g64 trunk 69.8% · 6-bit g32 n-gram 28.4% · 8-bit sensitivity ~1.1% · BF16 0.3% |

v2 size: skeleton 140.2 GiB minus the L4–43 trunk downgraded to the
oQ4e tensors ≈ **115 GiB** (final number stamped after the rebuild; expected
resident at load stays ~75 GB — the trunk is 4-bit exactly as before, the extra
bytes sit in the page-cache-only n-gram table).

## Specs & measurements (M5 Max 128 GB, oMLX 0.7.0 @PR4335)

| Item | Value |
|---|---|
| Architecture | qwen4_exp — MoE + n-gram PLE + MTP; 48 layers, 512 experts, top-10 routed; 125B total / 6B active / 51B n-gram / 4B MTP |
| Package | 116.8 GiB, 24 shards — dedup layout: 22 of 24 shards are byte-identical (same SHA) to Jundot's public oQ4e/oQ6e packs; only 2 mixed shards are new bytes |
| Resident at load | ~73–75 GB (oMLX, fresh load) |
| vs uniform oQ4e (same host) | **+5.96 GB initial resident** (75.49 vs 69.53 GB, both `engine_pool` actual loads on this M5 Max) — the premium buys 6-bit experts on the U-curve edges (L0–3/L44–47), the 6-bit g32 n-gram table and a 6/8-bit MTP drafter |
| vs uniform oQ5e (same family) | oQ5e package ≈128 GiB, initial resident ≈85 GB (measured on this M5 Max 128 GB host) — this build starts ~10–12 GB lower while matching or beating oQ5e accuracy (see Quality) |
| Context tested | 131k |
| MTP | native 6-bit, depth 3; acceptance 65–75% on agent/edit traffic; measured +6~8% tg @64k vs 4-bit drafter |

### Speed ladder — v2 vs uniform oQ4e (same host)

M5 Max, fan 80%, cold start 40 °C, second of two back-to-back runs per build,
oMLX 0.7.1.dev1. Same harness for both builds.

| Test | oQ4e tg | v2 tg | oQ4e pp | v2 pp | Δpp |
|---|---|---|---|---|---|
| pp1198/tg256 | 88.5 | 98.1 | 1,701.5 | 1,722.9 | +1.3% |
| pp4410/tg256 | 73.6 | 79.1 | 2,622.8 | 2,409.8 | −8.1% |
| pp8155/tg256 | 81.4 | 76.2 | 2,778.3 | 2,498.4 | −10.1% |
| pp15745/tg256 | 79.5 | 72.2 | 2,837.8 | 2,580.6 | −9.1% |
| pp30945/tg256 | 75.5 | 72.8 | 2,788.2 | 2,544.4 | −8.7% |
| pp62032/tg256 | 73.2 | 73.8 | 2,717.3 | 2,492.0 | −8.3% |
| pp122464/tg256 | 58.0 | 70.5 | 2,623.7 | 2,406.1 | −8.3% |
| **Mean** | **75.7** | **77.5** | **2,581.4** | **2,379.2** | **−7.8%** |

tg: no regression vs oQ4e. pp: honest ~8% cost — the 6-bit hot-path weights are
read on the prefill path too.

## Usage

Requires **oMLX 0.7.0+** (or mlx_vlm builds that support `qwen4_exp` plus
per-tensor mixed-precision quantization entries). Enable MTP speculative
decoding — the drafter is native 6/8-bit (inherited byte-for-byte from the oQ6e pack)
and pays for itself at long context.

```python
from mlx_vlm import load, generate
model, processor = load("umixer/Qwen3.8-Flash-Next-oQ4e-mix6-mtp")
```

## Quality

Task-level benchmarks on this v2 build (sampled-300 protocol, thinking on,
reasoning effort: **auto**, truncation accounted). Accuracy on finished answers
is the quality signal:

| Benchmark | Accuracy | Correct/Total | Accuracy on finished | Think time (s) |
|---|---|---|---|---|
| MMLU-Pro | 86.3% | 259/300 | **91.4%** (n=279; v1: 88.6%) | 5852.8 |
| MATHQA | 92.3% | 277/300 | **96.8%** (n=282; v1: 92.3%) | 3773.8 |
| MBPP | 89.3% | 268/300 | **99.3%** (n=267; v1: 97.3%) | 7323.1 |

## License & provenance

Derived from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(Qwen community license, `license: other`, applies to all weight derivatives).
Assembled byte-wise from community oQ quantizations —
[Jundot/Qwen3.8-Flash-Next-oQ6e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ6e-mtp)
and [Jundot/Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp)
— whose assembly and metadata rights are respected; weight lineage remains Qwen's.

## Acknowledgments

- [Jundot](https://huggingface.co/Jundot) — for [Qwen3.8-Flash-Next-oQ6e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ6e-mtp) (the 6-bit skeleton supplying every hot-path component) and [Qwen3.8-Flash-Next-oQ4e-mtp](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp) (the LSQ-refit 4-bit expert donor), both via the new-mechanism quantizer in [jundot/omlx#4385](https://github.com/jundot/omlx/pull/4385) — and for developing [oMLX](https://github.com/jundot/omlx) / the oQ quantizer that makes per-tensor mixed-precision builds possible on Apple Silicon.
- The U-shaped sensitivity literature: SliderQuant (arXiv:2603.25284), "Beyond Outliers: A Data-Free Layer-wise Mixed-Precision Quantization Approach Driven by Numerical and Structural Dual-Sensitivity" (arXiv:2603.17354).
- Community benchmarkers validating the mix vs uniform builds on the same hardware.
