---
license: mit
base_model:
  - ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
language:
  - en
library_name: mlx
tags:
  - mlx
  - omlx
  - mtplx
  - mtp
  - multi-token-prediction
  - speculative-decoding
  - qwen35moe
  - moe
  - 4-bit
pipeline_tag: image-text-to-text
---

# Ornith-1.5-35B-A3B-Vision-MTPLX-4bit

A validated oMLX 0.6.3 derivative of the 4-bit
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
combining the official vision tower with the native one-layer MTPLX head.

Ornith-1.5-35B-A3B is a 35B-parameter mixture-of-experts model activating approximately 3B parameters per token. This release preserves its native one-layer MTP head and includes the vision tower.

## Quantization

- **Main model:** 4-bit affine, group size 64
- **MTP runtime sidecar:** 4-bit affine, group size 64 (`mtp-omlx.safetensors`)
- **MTP norms:** BF16 (non-quantized 1-D tensors)
- **Vision runtime sidecar:** BF16 (`model-vision-omlx.safetensors`)
- **Architecture:** `Qwen3_5MoeForConditionalGeneration`
- **MTP contract:** Qwen3-Next / delta norm encoding

The MTP sidecar contains 46 tensors, including 13 quantized matrix-weight
tensors and 7 BF16 normalization tensors. All seven raw zero-centered RMSNorm gammas were converted to the MLX
representation by adding 1.0. This conversion raised measured draft acceptance
from 0% to 94-100% without changing deterministic output.

## Validated oMLX settings

- Runtime: oMLX 0.6.3
- Native MTP: enabled
- Draft depth: 2 (recommended mixed vision/text default)
- Thinking: disabled for the controlled benchmark

## Measured performance

### Test environment

- Mac Studio with Apple M3 Ultra (28-core CPU) and 96 GB unified memory
- macOS 26.6.2
- oMLX 0.6.3
- Deterministic decoding with thinking disabled
- MTP depth 2 unless otherwise stated

All results below are local measurements. `pp` is the prompt-token count and
`tg` is the requested generation-token count.

### Single-request synthetic benchmark

| MTP | Test | TTFT (ms) | TPOT (ms/token) | Prefill (tok/s) | Generation (tok/s) | End-to-end (s) | Aggregate throughput (tok/s) | Peak memory (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Off | pp1024/tg128 | 474.8 | 9.55 | 2156.5 | 105.5 | 1.696 | 679.2 | 19.35 |
| Depth 2 | pp1024/tg128 | 480.2 | 7.82 | 2132.3 | 129.0 | 1.480 | 778.4 | 21.05 |
| Off | pp4096/tg128 | 1603.9 | 9.63 | 2553.8 | 104.6 | 2.835 | 1489.7 | 21.26 |
| Depth 2 | pp4096/tg128 | 1621.1 | 8.01 | 2526.7 | 125.8 | 2.646 | 1596.2 | 23.01 |

Compared with MTP disabled, depth 2 increased single-request generation speed
by 22.3% for pp1024 and 20.3% for pp4096. TPOT decreased by 18.1% and 16.8%,
while end-to-end latency decreased by 12.7% and 6.7%, respectively. The tradeoff
was approximately 1.70-1.75 GB more peak memory. TTFT and prefill throughput
were effectively unchanged, with differences of about 1.2% or less.

### Extended prompt-length sweep (single configuration — MTP Depth 2)

The rows below extend the single-request sweep to longer prompts at a fixed
`tg128`. They use a single configuration (MTP Depth 2) and reproduce the Depth 2
baseline already shown above: the pp1024 / pp4096 single-config values (481.9 ms /
2,124.7 / 133.8 / 1.447 s / 21.05 GB, and 1,629.8 ms / 2,513.2 / 113.1 / 2.769 s /
23.01 GB) match the Depth 2 rows in the comparison table, to within measurement
variation.

| pp / tg | TTFT | TPOT | Prefill (tok/s) | Generation (tok/s) | End-to-end (s) | Aggregate throughput (tok/s) | Peak memory (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| pp8192 / tg128 | 3,432.7 | 8.38 | 2,386.5 | 120.2 | 4.505 | 1,846.8 | 23.20 |
| pp16384 / tg128 | 7,669.5 | 7.78 | 2,136.3 | 129.6 | 8.667 | 1,905.2 | 23.54 |
| pp32768 / tg128 | 18,701.6 | 9.55 | 1,752.2 | 105.6 | 19.925 | 1,651.0 | 24.23 |

### Continuous batching (pp1024/tg128)

The scale factor is relative to batch size 1 within the same run.

| MTP setting | Batch | Generation (tok/s) | Scale | Prefill (tok/s) | Prefill/request (tok/s) | Mean TTFT (ms) | End-to-end (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Off | 1 | 105.5 | 1.00x | 2156.5 | 2156.5 | 474.8 | 1.696 |
| Off | 2 | 195.0 | 1.85x | 1766.3 | 883.1 | 1159.4 | 2.472 |
| Off | 4 | 291.4 | 2.76x | 1760.2 | 440.1 | 2200.8 | 4.084 |
| Depth 2 | 1 | 129.0 | 1.00x | 2132.3 | 2132.3 | 480.2 | 1.480 |
| Depth 2 | 2 | 177.2 | 1.37x | 1747.3 | 873.6 | 1172.0 | 2.617 |
| Depth 2 | 4 | 275.3 | 2.13x | 1753.5 | 438.4 | 2206.1 | 4.196 |

The multi-request rows are observational rather than a controlled measurement
of MTP itself. During the run, oMLX reported that MTP was inactive for the
two-row batch because standard batched decode was faster, and the enabled and
disabled runs used different VLM/language-only load paths. Consequently, the
batch-size 2 and 4 rows should not be interpreted as evidence that MTP causes a
concurrency regression.

### Continuous-batching sweep (single configuration — MTP Depth 2)

The sweep below re-runs a single batch-size series at the MTP Depth 2 configuration.
The batch-size 1 row (129.0 → 133.8 tok/s generation, 480.2 → 481.9 ms TTFT, 1.480 →
1.447 s end-to-end) reproduces the Depth 2 baseline in the comparison table, to within
measurement variation. The 2× and 4× rows carry the same caveat as the comparison table:
they are a single configuration sweep, not a controlled Off-vs-Depth comparison, and the
batched rows were run under a single load path.

| Batch | Generation (tok/s) | Scale | Prefill (tok/s) | Prefill/request (tok/s) | Mean TTFT (ms) | End-to-end (s) |
|---|---:|---:|---:|---:|---:|---:|
| 1x（基准线） | 133.8 | 1.00x | 2,124.7 | 2,124.7 | 481.9 | 1.447 |
| 2x | 175.8 | 1.31x | 1,730.3 | 865.1 | 1,183.6 | 2.640 |
| 4x | 272.2 | 2.03x | 1,731.6 | 432.9 | 2,228.8 | 4.246 |

### Fixed-task warm depth sweep

| Task | MTP off | Depth 1 | Depth 2 | Depth 3 | Depth-2 change vs off |
|---|---:|---:|---:|---:|---:|
| Visual chart reading | 3.173 s | 2.982 s | 2.904 s | 2.911 s | 8.5% faster |
| Short-prompt, long-output text | 2.404 s | 2.128 s | 1.696 s | 1.604 s | 29.5% faster |

The controlled text microbenchmark favored depth 3, but subsequent real-workload
testing regressed after switching to depth 3. No exact result was retained for
that later workload, so it is not quantified here. Depth 2 remains the final
recommended default because it performed consistently across the measured text
and visual tasks. These results are workload- and machine-specific, not a
general performance guarantee.

### Performance highlights (single configuration — MTP Depth 2)

The following observations describe the Depth 2 sweep above. They are local point
measurements on this machine and oMLX setup, not a cross-hardware ranking.

- **Throughput scales with prompt length.** End-to-end aggregate throughput rises from
  ~796 tok/s at pp1024 to a peak of ~1,905 tok/s around pp16384, then settles at
  ~1,651 tok/s at pp32768. The fixed per-request generation cost (tg128) amortizes more
  effectively as the prompt grows, so long-prompt single requests are the strongest case.
- **Memory stays bounded across prompt lengths.** Peak memory climbs from ~21 GB at
  pp1024 to ~24 GB at pp32768 — roughly a 3 GB headroom over a 32 K token prompt —
  consistent with the 4-bit language body and the preserved BF16 vision tower.
- **Fast first token.** First-token latency stays sub-millisecond at short prompts
  (~482 ms TTFT at pp1024), and generation throughput holds in the 105–134 tok/s range
  across all prompt lengths.
- **Continuous-batching headroom narrowed.** The 4× batch now achieves ~2.03× aggregate
  generation throughput, lower than the ~2.13× recorded in the comparison table above.
  This is expected once per-request throughput is already high: there is less free
  throughput to recover by packing requests, so the batching speedup compresses. It is
  not a regression in per-request behavior.

### Model intelligence benchmarks

The benchmarks above measure oMLX **latency, throughput, and memory**. This section
measures **model intelligence** — accuracy on general-knowledge and coding
benchmarks — which is a different axis, and was measured with Think reasoning
enabled throughout.

Sampling windows: 100/14042 (MMLU), 100/11582 (CMMLU), 100/817 (TruthfulQA),
100/164 (HumanEval), 100/500 (MBPP), 100/1055 (LiveCodeBench). Recorded
2026-08-29.

#### Cross-model comparison (accuracy %)

| Model | MMLU | CMMLU | TruthfulQA | HumanEval | MBPP | LiveCodeBench |
|------|------|-------|------------|-----------|------|---------------|
| Qwen3.8-27B-oQ4e-mtp | 87 | 71 | 90 | 99 | 81 | 43.3 |
| Ornith-1.0-35B-Vision (Think on) | 87 | 87 | 92 | 97 | 87 | 53.3 |
| Ornith-1.0-35B-Vision (Think off) | 85 | 82 | 91 | 91 | 84 | 43.3 |
| ornith-1.5-35b-a3b-abliterated-attention8-bf16recurrence-vision-mtplx | 47 | 53 | 69 | 78 | 67 | 16.7 |
| Ornith-1.5-35B-A3B-MLX-4bit | 71 | 37 | 87 | 78 | 79 | 23.3 |
| Ornith-1.5-35B-A3B-MLX-6bit | 63 | 53 | 87 | 87 | 78 | 30.0 |
| **Ornith-1.5-35B-A3B-Vision-MTPLX-4bit** | **85** | **74** | **89** | **93** | **91** | **61.0** |
| Qwen3.8-27B-oQ4e-fp16-mtp | 85 | 72 | 90 | 99 | 83 | 50 |

#### Ornith-1.5-35B-A3B-Vision-MTPLX-4bit (Think on)

| Benchmark | Accuracy | Time(s) |
|-----------|----------|---------|
| MMLU | 85.0% | 596.3 |
| CMMLU | 74.0% | 623.7 |
| TruthfulQA | 89.0% | 171.7 |
| HumanEval | 93.0% | 336.7 |
| MBPP | 91.0% | 785.6 |
| LiveCodeBench | 61.0% | 4346.2 |

**Highlights:**

- **Complex-code leader within the 1.5 series.** The only 1.5-series model to
  exceed 60% on LiveCodeBench (61.0%), well ahead of the other 1.5 variants
  (abliterated 16.7%, MLX-4bit 23.3%, MLX-6bit 30.0%) and above Ornith-1.0
  (53.3%). LiveCodeBench measures real-world, hard coding tasks, so the gap is
  meaningful.
- **Strong code generation.** HumanEval 93% and MBPP 91% — MBPP alone
  outperforms Qwen3.8-27B (81%).
- **Balanced general benchmarks.** MMLU 85% and TruthfulQA 89% sit close to
  Ornith-1.0 (Think on); Chinese knowledge (CMMLU 74%) is weaker than 1.0 (87%)
  but stronger than the other 1.5 variants (37–53%).
- **Fair cross-comparison.** vs Qwen3.8-27B-oQ4e-fp16-mtp (Think off): beats it
  on LiveCodeBench (61% vs 50%) and MBPP (91% vs 83%), ties MMLU (85%), slightly
  behind on HumanEval (93% vs 99%) and TruthfulQA (89% vs 90%). Net: stronger on
  hard real-time coding, comparable on general common sense.

**Caveats:**

- LiveCodeBench cost dominates: 4346.2s, roughly 10× MMLU — the Think overhead on
  complex-code reasoning is the main cost. Other benchmarks run at moderate
  latency.
- LiveCodeBench here uses a 100-sample window, larger than the 30-sample runs of
  the other 1.5 variants, so its 61.0% is statistically more reliable — but
  direct numeric comparison against those 30-sample figures should carry that
  caveat.
- CMMLU (74%) is meaningfully lower than Ornith-1.0 (87%); the 4-bit quantization
  trades some Chinese-knowledge fidelity.
- These are point measurements on this machine and oMLX build, not a general
  guarantee.

## Usage

```bash
# Select this directory in oMLX 0.6.3, enable native MTP,
# and set the MTP draft-token count to 2.
```

The model can also be loaded by MLX-LM for autoregressive text generation, without MTPLX speculative decoding.

## Provenance and licensing

- **Base model:** [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) — MIT
- **4-bit MTPLX source:** [wang-yang/Ornith-1.5-35B-A3B-MTPLX-4bit](https://huggingface.co/wang-yang/Ornith-1.5-35B-A3B-MTPLX-4bit)
- **Vision/config compatibility reference:** [scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp](https://huggingface.co/scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp)
- **Runtime:** [MTPLX](https://github.com/youssofal/MTPLX) — Apache-2.0
- **Conversion:** MLX affine quantization, group size 64

This repository contains a quantized derivative of the base model. Please follow the upstream model license and usage conditions. MTPLX attribution is required when shipping a product or service built on MTPLX:

> Powered by MTPLX  
> https://github.com/youssofal/MTPLX
