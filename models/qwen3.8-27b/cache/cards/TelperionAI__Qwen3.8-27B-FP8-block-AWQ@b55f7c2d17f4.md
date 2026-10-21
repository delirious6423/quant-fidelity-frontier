---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
tags:
  - fp8
  - awq
  - vllm
  - compressed-tensors
---

# Qwen3.8-27B — FP8 block + AWQ

FP8 block-scaled `[128, 128]` quantization of `Qwen/Qwen3.8-27B` with dynamic
activations — **the same format and the same 882 excluded modules as Qwen's own
FP8 release** — with one addition they do not use: an **AWQ smoothing pre-pass**
on the BF16 weights.

That single change cuts real damage by **45%**: 1.45% → **0.79%** confident
disagreement, z = +9.80.

Drop-in for vLLM. ~30 GB, 8,656 tok/s (TP=2, B300).

## Measured quality

Every row measured identically: 231 documents, 142,727 scored token positions,
against the BF16 base.

| build | size | top-1 ↑ | near-tie ↓ | moderate ↓ | **confident ↓** | certain ↓ | divmed ↑ | tok/s |
|---|---|---|---|---|---|---|---|---|
| **➤ this model — FP8 block + AWQ** | 30 GB | 96.22% | 22.83% | 3.76% | **0.79%** | **0.08%** | 51 | **8656** |
| TelperionAI INT4-AWQ-GPTQ | 25.1 GB | **96.30%** | **22.29%** | 3.52% | 0.93% | 0.09% | 48 | 4617 |
| TelperionAI EXL3 6.5 bpw + AWQ | 23.0 GB | 97.10% | 17.03% | 2.41% | 1.23% | 0.15% | 59 | — |
| TelperionAI EXL3 5.5 bpw + AWQ | 20.0 GB | 96.12% | 22.81% | 3.52% | 1.34% | 0.16% | 55 | — |
| Qwen FP8 (same format, no AWQ) | 30 GB | 96.15% | 22.70% | 3.48% | 1.45% | **0.08%** | 47 | 8711 |
| TelperionAI NVFP4 | 24.7 GB | 93.62% | 32.25% | 8.59% | 1.85% | 0.16% | 29 | 10521 |
| RadixArk NVFP4 | 24.7 GB | 90.23% | 43.80% | 14.49% | 3.29% | 0.70% | 11 | 11436 |
| unsloth NVFP4 | 24.7 GB | 91.75% | 40.12% | 10.32% | 3.91% | 0.25% | 19 | 11069 |
| Inferact NVFP4 | 20.1 GB | 84.11% | 52.01% | 25.70% | 10.39% | 2.27% | 7 | 11457 |

EXL3 rows have no comparable `tok/s` — they run on exllamav3, not vLLM.

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
which free greedy generation first diverges from the base.

## Significance

Paired McNemar over matched token positions, n = 33,812:

| vs | their confident | z | verdict |
|---|---|---|---|
| TelperionAI NVFP4 | 1.85% | **+14.80** | this model better |
| **Qwen FP8** (same format) | 1.45% | **+9.80** | this model better |
| TelperionAI EXL3 5.5bpw+AWQ | 1.34% | **+9.14** | this model better |
| TelperionAI EXL3 6.5bpw+AWQ | 1.23% | **+7.43** | this model better |
| TelperionAI INT4-AWQ-GPTQ | 0.93% | **+2.46** | this model better |

It also ties the best `certain` figure in the field (0.08%).

## Recipe

1. **AWQ smoothing** on the BF16 weights — a pure, function-preserving transform
   (fold `1/s` into the preceding norm, `s` into the next linear's input dim),
   calibrated on 256 sequences. Verified function-preserving: the ratio to the
   original weights is a separable per-row × per-column scale.
2. **FP8 block `[128,128]`** weights with dynamic per-group activations,
   excluding the same 882 modules Qwen excludes. This stage needs **no
   calibration** — weight scales come from the weight tensor alone and
   activations are scaled at runtime.

Note that the 882 exclusions are not conservatism: FP8 block requires every
quantized dimension to be divisible by 128, and the router / `shared_expert_gate`
layers are not.

## Why AWQ helps at 8 bits

AWQ corrects **activation outliers**, which are a property of the model rather
than of the quantizer — so it transfers across formats. Measured on this model:
NVFP4 3.97% → 2.69%, a 4-bit trellis 3.08% → 2.40%, and block FP8 1.45% → 0.79%.
A `[128,128]` block still spans a wide dynamic range, so outlier channels cost
real precision even at 8 bits.

GPTQ and AutoRound do **not** transfer this way: they are quantizers whose error
compensation is tied to specific rounding decisions on a specific grid.

## KV-cache quantization (calibrated scales included)

This checkpoint carries per-layer `k_scale` / `v_scale` in
`model-kv-scales.safetensors`, with `kv_cache_scheme` recorded under
`quantization_config` — the on-disk contract llm-compressor emits and vLLM reads
in `BaseKVCacheMethod`. With no scales present vLLM falls back to 1.0.

```
--kv-cache-dtype fp8      # 32 KiB/token, from 64 KiB at fp16
```

Qwen3.8-27B is a hybrid stack: only 16 of its 64 layers are `full_attention` and
carry a KV cache at all, so this is 32 scalars in a 4 KB file.

Scales were observed on **this quantized model**, at the point vLLM caches K/V
(post-RoPE, at the attention dispatch), over a 2353-document / 3,713,778-token
corpus, taking a running absmax per layer (`memoryless_minmax`,
llm-compressor's default KV observer); `k_scale = amax/448`.

Corpus size matters more than it looks, and only for V. absmax is a tail
statistic: it sits flat, then jumps when a rare document appears, so a small
calibration set looks converged when it is not. A 48-document pass
under-estimated V by up to 1.9x and a 231-document pass was still up to 1.5x
low; at this size **no tensor set a new maximum over the final 50 documents**.
Under-estimating clips exactly the outliers a cache most needs to keep. K is
converged by ~100 documents and needs none of this; V's range also grows ~13x
from the first attention layer to the last, which is why the scales are
per-layer rather than global.
