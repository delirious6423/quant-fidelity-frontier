---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-generation
tags:
- mlx
- oq
- quantized
- qwen3.8
---

# Qwen3.8-27B-oQ4e-fp16-mtp

4-bit MLX quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (hybrid Gated DeltaNet + Gated Attention, native vision, 262K context, thinking mode with preserved reasoning) — the **fp16 variant of the 4-bit build, tuned for M1/M2-generation Apple Silicon**, where fp16 non-quant weights prefill about 20% faster than bf16. It was the family's **first ANE-eligible build**: oMLX 0.6.1's experimental Qwen ANE prompt processing accepted only this q4 layout, measured at **+21% prefill throughput** on an M2 Ultra. oMLX 0.6.3's reworked path (fused MLP/down offload + FP16 CPU sharing) measures **+25.6% at 64K** on this build at a higher resident cost (\~11 GB, up from \~4 GB — details below), and extends ANE eligibility to the [8-bit sibling](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-fp16-mtp) (+19.5% at 64K, ~16 GB — see its card). Made with [oQ](https://github.com/jundot/omlx) (oMLX v0.6.1) mixed-precision quantization.

**Sibling repos:**
- [Qwen3.8-27B-oQ4e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ4e-mtp) — 4-bit / bf16, the same weights in the safe dtype for M3/M4/M5 machines
- [Qwen3.8-27B-oQ6e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ6e-mtp) — 6-bit / bf16, ~22 GB, for 64 GB-class machines
- [Qwen3.8-27B-oQ8e-fp16-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-fp16-mtp) — 8-bit / fp16, ~27.5 GiB, highest fidelity
- [Qwen3.8-27B-oQ8e-mtp](https://huggingface.co/evsinlb/Qwen3.8-27B-oQ8e-mtp) — 8-bit / bf16, ~27.9 GiB; same quantized tensors and imatrix as the oQ8e-fp16 build, bf16 floating tensors for M3/M4+ numerical safety

## Quantization details

- **Model type**: qwen3_5
- **Bits**: 4 (effective **~4.9 bpw** mixed precision), group size 64 — ~16.7 GiB weights
- **Enhanced quantization (oQe)**: imatrix-calibrated (1024 samples, shared calibration with the family) — affine quantization weighted by activation importance
- **MTP weights preserved** (`mtp.*` tensors + config) — multi-token-prediction / Lightning MTP works after quantization
- **Non-quant weight dtype**: float16 — ~20% faster prefill on M1/M2; on M3+ prefer the bf16 sibling. No fp16 activation-overflow issues observed with Qwen3.8 in extended agentic use (this is a Gemma-class problem, not a Qwen one)
- **Vision components included** (not a text-only strip)
- **Format**: MLX safetensors

## Recommended sampling (per the Qwen3.8 model card)

| Mode | temperature | top_p |
|---|---|---|
| Thinking (default) | 1.0 | 0.95 |
| Instruct (non-thinking) | 0.7 | 0.80 |

Thinking controls via `chat_template_kwargs`:

- `enable_thinking` (default `true`)
- `preserve_thinking` (default `true`) — keeps reasoning traces across multi-turn history
- `reasoning_effort`: `xhigh` (default) / `medium` / `low` — in our testing (on the 8-bit sibling), `medium` reduced thinking volume ~25% with no loss on agentic tasks

## Measured: ANE prompt processing (experimental)

### oMLX 0.6.1/0.6.2

This layout (q4, group size 64) is eligible for oMLX 0.6.1's dual-ANE/GPU prefill split. Measured on an **M2 Ultra Mac Studio** (128 GB), ~78K-token cold prefills, 3 unique-prompt runs per leg, prefix cache verified cold:

| Configuration | Prefill tok/s |
|---|---:|
| GPU only (ANE off) | ~222 |
| **ANE on (dual-ANE + GPU)** | **~269 (+21%)** |
| 8-bit sibling, GPU only (ANE-ineligible in ≤0.6.2; eligible from 0.6.3 — see its card), same machine | ~207† |

†Measured at ~88K, depth-adjusted to ~78K. The +21% slightly exceeds the M3 Ultra reference figures (+18.9% at 32K) because longer prompts amortize more complete 2,048-token ANE blocks — the win grows with prompt length and applies to **cold prefills only** (prefix-cached turns are unaffected).

oMLX 0.6.2's built-in split tuner, swept on the same M2 Ultra, confirmed **53% MLP / 50% GDN as the optimal split** and measured **+37.6% (444.9 prompt tok/s)** at its shorter benchmark prompt shape. The percentage gain varies with prompt depth: it rises with block count out of the small-prompt regime, peaks mid-range, then tapers at very long contexts as the full-attention layers (not ANE-accelerated) claim a growing share of prefill time — hence +37.6% at the tuner's shape vs. +21% at 78K.

Costs to know before enabling: ~4.15 GB extra resident memory for the model's lifetime (≈68K tokens' worth of KV cache), ~40 s eager ANE program compile added to every model load, and an approximate INT8 weight copy on the ANE path (upstream reports matching output hashes at 16K/32K vs. the GPU path). On 64 GB machines, leave it off — the resident cost comes out of exactly the headroom that long-context prefill needs.

### oMLX 0.6.3 update (measured 2026-08-27)

0.6.3 reworked the ANE path: bit-aware affine quantization (replacing the INT8 approximation), fused MLP/down offload, an FP16 CPU branch (this fp16 build feeds it natively), and calibrated tail padding. Tuner V2 on the same M2 Ultra selected MLP 50% / GDN 37.5% on ANE with CPU shares of 10% gate / 20% down / 10% GDN (8 workers), measuring **+40.9% (451.6 pp tok/s)** at its shape — about 3 points over the old 0.6.2 split. Version-matched on/off pair, oMLX built-in benchmark, ANE-aligned prompts off (realistic ragged lengths), one run per leg:

| Prompt depth | GPU only (pp tok/s) | ANE on (pp tok/s) | Gain |
|---|---:|---:|---:|
| 1,024 | 308.5 | 308.9 | 0% |
| 4,096 | 322.9 | 437.3 | +35.4% |
| 65,536 | 242.2 | 304.2 | **+25.6%** |

Decode is unaffected at every depth (e.g. 23.8 → 23.0 tg tok/s at 64K, within variance), and continuous-batching behavior is identical on and off. **The 0.6.3 path costs ~11 GB resident** (measured peak 25.3 → 36.2 GB at 64K) — nearly triple the 0.6.1 path's ~4.15 GB; the earlier figure no longer applies on 0.6.3. The 64 GB-machine advice above stands, more firmly than before: prompts ≤1K gain nothing (they never fill a 2,048-token ANE block), and the gain applies to cold prefills only.

## Measured context windows

The 64 GB ladder was run on the bf16 sibling (M5 Pro, oMLX, default wired-memory limit): **64K OK, 131K OK, 160K crashed the oMLX server** (allocation spike during prefill; no graceful rejection) — **131,072 is the recommended client window on 64 GB machines**. Real windows on this hybrid architecture are **prefill-overhead-bound, not KV-bound**, so don't size your window from KV math, and **cap the context client-side** — an over-window request may crash the server rather than return an error. The full 262,144 native window was validated with the 8-bit sibling on a 128 GB Mac Studio.

## Fidelity note

4-bit is the aggressive end for a 27B reasoning model; the imatrix calibration mitigates but does not eliminate low-bit degradation, so expect the largest quality gap of the family on long reasoning chains. **This exact build passed the family's multi-turn agentic battery** (2026-08-18, M2 Mac Studio, thinking fed back into history verbatim): 3/3 scenarios completed with correct answers, 0 stalled turns across 8 agentic tool-call turns. The battery's scenarios are short; the caveat about very long reasoning chains stands. A good deployment pattern from our own use: run this build as a **fast long-context instruct tier** (batch summarization/extraction, one-shot long-document Q&A — cold-prefill-dominated work where the ANE gain lands and quant sensitivity is lowest) and keep an 8-bit sibling for thinking-grade tasks.
