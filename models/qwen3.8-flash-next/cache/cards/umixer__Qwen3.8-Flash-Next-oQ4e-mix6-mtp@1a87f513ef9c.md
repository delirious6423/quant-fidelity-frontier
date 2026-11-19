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
**component-wise byte-copy** of native oQ quantization tensors —
**zero re-quantization in the entire package**. Every tensor is byte-identical
to its source, so quality is traceable, not re-derived.

> **Rebuild (2026-10-30):** the GDN `out_proj` trio — 36 tensors — moves
> **6-bit g128 → 8-bit g64** (3 of 24 shards carry new bytes). What that buys: out_proj noise drops to roughly ¼ (2 bits plus 2×
> finer groups). Measured pp ≈ 2,840–3,040 tok/s @8k (M5 Max, same bench
> harness, PLE page-cache). Lineage unchanged: middle-40 trunk from
> new-mechanism oQ4e (importance-matrix calibration 1024 samples × 512 tokens,
> per-group weighted least-squares refit), 6-bit skeleton from oQ6e.

## Why this build exists

Uniform-bit quantization pays accuracy where it isn't needed (middle layers) and
loses it where it hurts (first/last layers, n-gram table, attention).
Layer-sensitivity literature (SliderQuant arXiv:2603.25284) shows
quantization sensitivity is U-shaped across depth: expensive edges, cheap
middle. This package prices that curve with **native
bits** — the high-value components are lifted verbatim from higher-bit packs
instead of re-quantized.

## Recipe & comparison

| Component | This build | uniform oQ4e | Resident Δ vs oQ4e |
|---|---|---|---|
| Trunk MoE experts, layers 0–3 & 44–47 (U-shaped edges) | **6-bit g64** (LSQ-refit) | 4-bit | **+4.8 GB** (hot, fully resident) |
| Trunk MoE experts, middle 40 layers | 4-bit g64, **imatrix + LSQ refit** | 4-bit (absmax) | 0 |
| Attention / GDN / hyper-connections / conv | **all 48 layers ≥6-bit**: 6-bit on 40 layers, 8-bit at L0/L31/L32/L35/L36/L39/L44/L46; **GDN out_proj trio (36 tensors) 8-bit g64** since 10-30 (was 6-bit g128) | mostly 4-bit, sensitivity layers raised to 5–8 | **+0.4 GB** (hot, resident) |
| N-gram (PLE) embedding table | **6-bit g32** (41.7 GiB) | 4-bit g32 | ≈0 (SSD-offloaded on hosts; page-cache only) |
| Shared experts (all 48 layers) | 8-bit g128 | 8-bit g128 | 0 |
| embed_tokens / lm_head | 8-bit g64 | 8-bit g64 | 0 |
| MTP drafter (experts 6-bit, attention/hyper-conn 8-bit) | **6/8-bit** | 4-bit experts / 6-bit attn | **+0.6 GB** (hot, resident) |
| Norms / routers / vision tower | bf16, unquantized | bf16, unquantized | 0 |
| **Total at load** | **75.49 GB** | **69.53 GB** | **+5.96 GB** |

Every component where this build differs from uniform oQ4e is one the router or
the hot path touches every token — the extra 5.8 GB of resident weights buys
6-bit exactly where quantization error is realized.

## Quantization mechanism

A `bits=4, group_size=64, mode=affine` skeleton with every high-bit component
carried as a per-tensor entry in `config.json`. The trunk's 4-bit tensors use a
new-mechanism quantizer: sensitivity-measured bit allocation, imatrix
calibration (1024 samples × 512 tokens), and a **weighted least-squares refit
of each group's scale/bias** replacing plain absmax rounding. The high-bit
components ride the same affine format at 6/8-bit with group sizes 32–128.

## Specs & measurements (M5 Max 128 GB, oMLX 0.7.0 @PR4335)

| Item | Value |
|---|---|
| Architecture | qwen4_exp — MoE + n-gram PLE + MTP; 48 layers, 512 experts, top-10 routed; 125B total / 6B active / 51B n-gram / 4B MTP |
| Package | 117.5 GiB, 24 shards — dedup layout: 21 of 24 shards byte-identical to the upstream quantized packs; 3 mixed shards carry new bytes (the 10-30 out_proj lift touched shards 11,19,20) |
| Resident at load | ~73–75 GB (oMLX, fresh load) |
| vs uniform oQ4e (same host) | **+5.96 GB initial resident** (75.49 vs 69.53 GB, both `engine_pool` actual loads on this M5 Max) — the premium buys 6-bit experts on the U-curve edges (L0–3/L44–47), the 6-bit g32 n-gram table and a 6/8-bit MTP drafter |
| vs uniform oQ5e (same family) | oQ5e package ≈128 GiB, initial resident ≈85 GB (measured on this M5 Max 128 GB host) — this build starts ~10–12 GB lower while matching or beating oQ5e accuracy (see Quality) |
| Context tested | 131k |
| MTP | native 6-bit, depth 3; acceptance 65–75% on agent/edit traffic; measured +6~8% tg @64k vs 4-bit drafter |

### Speed ladder — this build vs uniform oQ4e (same host)

M5 Max, fan 80%, cold start 40 °C, second of two back-to-back runs per build,
oMLX 0.7.1.dev1. Same harness for both builds.

| Test | oQ4e tg | mix6 tg | oQ4e pp | mix6 pp | Δpp |
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
decoding — the drafter is native 6/8-bit
and pays for itself at long context. On oMLX ≥ 0.7.3 set
`qwen4_gdn_decode_wide_proj = true` for this pack — its 8-bit g64 out_proj
enters the fused GDN decode path only through that switch.

```python
from mlx_vlm import load, generate
model, processor = load("umixer/Qwen3.8-Flash-Next-oQ4e-mix6-mtp")
```

## Quality

Task-level benchmarks on this build (sampled-300 protocol, thinking on,
reasoning effort: **auto**, truncation accounted). Accuracy on finished answers
is the quality signal:

| Benchmark | Accuracy | Correct/Total | Accuracy on finished | Think time (s) |
|---|---|---|---|---|
| MMLU-Pro | 86.3% | 259/300 | **91.4%** (n=279; v1: 88.6%) | 5852.8 |
| MATHQA | 92.3% | 277/300 | **96.8%** (n=282; v1: 92.3%) | 3773.8 |
| MBPP | 89.3% | 268/300 | **99.3%** (n=267; v1: 97.3%) | 7323.1 |

## License

Derived from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(Qwen community license, `license: other`, applies to all weight derivatives).
Weight lineage remains Qwen's.

## Acknowledgments

- The U-shaped sensitivity literature: SliderQuant (arXiv:2603.25284), "Beyond Outliers: A Data-Free Layer-wise Mixed-Precision Quantization Approach Driven by Numerical and Structural Dual-Sensitivity" (arXiv:2603.17354).
- Community benchmarkers validating the mix vs uniform builds on the same hardware.
