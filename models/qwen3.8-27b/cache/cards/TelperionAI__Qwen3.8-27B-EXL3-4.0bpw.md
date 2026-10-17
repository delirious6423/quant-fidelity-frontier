---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
tags:
  - exl3
  - exllamav3
  - quantized
  - awq
---

# Qwen3.8-27B — EXL3 4.0 bpw (AWQ-smoothed)

Trellis quantization ([EXL3](https://github.com/turboderp-org/exllamav3), a QTIP
variant) of `Qwen/Qwen3.8-27B` at **4.0 bits per weight**, with an **AWQ
smoothing pre-pass** applied to the BF16 weights first.

**16.0 GB**, `lm_head` at 6 bpw, MTP layers quantized inline.

> Requires **exllamav3 / TabbyAPI**. EXL3 cannot be served by vLLM.

## Measured quality

Every row measured identically: 231 documents, 142,727 scored token positions,
against the BF16 base. `this model` rows are the three checkpoints in this
series; the others are for context.

| build | bits/wt | size | top-1 ↑ | near-tie ↓ | moderate ↓ | **confident ↓** | certain ↓ | divmed ↑ | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| TelperionAI **INT4**-AWQ-GPTQ (vLLM) | 4.63 | 25.1 GB | 96.30% | 22.29% | 3.52% | **0.93%** | **0.09%** | 48 | 4617 |
| EXL3 6.5 bpw | 6.50 | 23.0 GB | **97.10%** | **17.03%** | **2.41%** | 1.23% | 0.15% | **59** | 492* |
| EXL3 5.5 bpw | 5.50 | 20.0 GB | 96.12% | 22.81% | 3.52% | 1.34% | 0.16% | 55 | 516* |
| Qwen **FP8** (reference, vLLM) | 8.00 | ~30 GB | 96.15% | 22.70% | 3.48% | 1.45% | 0.08% | 47 | 8711 |
| TelperionAI **NVFP4** (vLLM) | 4.50 | 24.7 GB | 93.62% | 32.25% | 8.59% | 1.85% | 0.16% | 29 | 10521 |
| **➤ this model — EXL3 4.0 bpw** | 4.00 | 16.0 GB | 93.82% | 31.66% | 6.94% | 2.40% | 0.58% | 28 | 570* |
| EXL3 4.0 bpw, no AWQ pre-pass | 4.00 | 16.0 GB | 93.17% | 34.05% | 8.41% | 3.08% | 0.16% | 27 | 561* |

\* EXL3 throughput is single-stream argmax on one GPU via exllamav3 — **not**
comparable to the batched vLLM figures in the other rows. It was not tuned.

### How to read this

Columns are **disagreement rates against the BF16 base**, split by how confident
the base model was at that position (its top1−top2 logprob margin):

| bucket | margin | meaning |
|---|---|---|
| `near-tie` | < 0.5 | numerical noise — harmless |
| `moderate` | 0.5–2 | mild |
| **`confident`** | 2–5 | **real damage** |
| **`certain`** | > 5 | **real damage** |

**Only `confident` and `certain` matter.** `divmed` is the median token index at
which free greedy generation first diverges from the base model (higher = holds
a trajectory longer).

## Bitrate scaling in this series

| bitrate | size | confident ↓ | change |
|---|---|---|---|
| 4.0 bpw, no AWQ | 16.0 GB | 3.08% | — |
| 4.0 bpw + AWQ | 16.0 GB | 2.40% | AWQ pre-pass, z=+6.91 |
| 5.5 bpw + AWQ | 20.0 GB | 1.34% | +1.5 bits, z=+14.29 |
| 6.5 bpw + AWQ | 23.0 GB | 1.23% | +1.0 bit, z=+2.89 (diminishing) |

**5.5 bpw is the knee of the curve.** Above it, extra bits buy an order of
magnitude less per bit.

## Recipe

1. **AWQ smoothing** on the BF16 weights — a pure, function-preserving weight
   transform (fold `1/s` into the preceding norm, `s` into the next linear's
   input dim), calibrated on 256 sequences. Verified function-preserving: the
   ratio to the original weights is a separable per-row × per-column scale.
2. **EXL3 conversion** at 4.0 bpw on the smoothed weights.

AWQ stacks because it leaves an ordinary BF16 checkpoint. **GPTQ and AutoRound
do not stack** — they are quantizers whose error compensation is tied to
specific rounding decisions on a specific grid, which EXL3 discards while
keeping the damage.

## Honest comparison

- Beats every NVFP4 build measured, at smaller size (5.5 bpw: 1.34% at 20.0 GB
  vs NVFP4's 1.85% at 24.7 GB, z=+7.11).
- 6.5 bpw takes the best `top-1`, `near-tie`, `moderate` and `divmed` in the
  field — it agrees with the base model more often *overall*.
- **But INT4 group-32 asymmetric is still more faithful where it counts**:
  0.93% confident vs 1.23% (z=−5.06), and 0.09% vs 0.15% certain (z=−3.18).
  That build serves in vLLM at 4,617 tok/s.

Significance is paired McNemar over matched token positions, n = 33,812.

## KV-cache quantization (calibrated scales included)

This checkpoint ships per-layer `k_scale` / `v_scale` in
`model-kv-scales.safetensors`, with `kv_cache_scheme` recorded under
`quantization_config` — the on-disk contract llm-compressor emits for
`kv_cache_scheme` and vLLM reads in `BaseKVCacheMethod`. Without them vLLM falls
back to a scale of 1.0.

```
--kv-cache-dtype fp8      # 32 KiB/token, from 64 KiB at fp16
--kv-cache-dtype nvfp4    # 18 KiB/token, FlashInfer + TRTLLM attention only
```

Qwen3.8-27B is a hybrid stack: only 16 of its 64 layers are `full_attention` and
carry a KV cache at all, so this is 32 scalars in a 4 KB file.

Scales were observed on **this quantized model**, post-RoPE, where vLLM caches
K/V, over a 2,353-document / 3,711,425-token corpus, taking a running absmax per
layer (`memoryless_minmax`, llm-compressor's default KV observer);
`k_scale = amax/448`.

Corpus size turned out to matter, and only for V. absmax is a tail statistic —
it sits flat and then jumps when a rare document appears, so a small calibration
set looks converged when it is not. A 48-document pass under-estimated V by up
to 1.9x, and a 231-document pass was still up to 1.5x low; at this size no
tensor set a new maximum over the final 50 documents. Under-estimating clips
exactly the outliers a cache most needs to keep. K is converged within ~100
documents and needs none of this; V's range also grows ~13x from the first
attention layer to the last, which is why the scales are per-layer.

### Measured cost

Disagreement vs the BF16 base bucketed by the base model's top1−top2 margin,
same corpus and harness as the rest of the matrix. `confident` is the
pre-registered decider; z is paired McNemar against the fp16-cache row,
n = 33,812, positive = fp16 cache better.

| KV cache | KiB/token | top-1 ↑ | near-tie ↓ | moderate ↓ | **confident ↓** | certain ↓ | vs fp16 |
|---|---|---|---|---|---|---|---|
| fp16 (none) | 64 | 93.82% | 31.66% | 6.94% | 2.40% | 0.58% | — |
| **fp8 + these scales** | **32** | 93.82% | 31.73% | 7.07% | 2.40% | 0.45% | z=+0.20, n.s. |
| nvfp4 + these scales | **18** | 93.47% | 33.26% | 7.34% | 2.58% | 0.61% | z=+4.05 |

Note this harness scores next-token agreement, which is not sensitive to what a
clipped KV outlier does to a long reasoning chain. It supports "fp8 KV is
cheap here"; it is not a substitute for a reasoning benchmark.
