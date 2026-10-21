---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags: [exl3, exllamav3, quantization, awq]
---

# Qwen3.8-27B — EXL3 5.5 bpw (v2)

EXL3 trellis quantization of `Qwen/Qwen3.8-27B` at **5.5 bpw**, 20 GB, with AWQ
smoothing and calibrated FP8 KV-cache scales.

## What changed in v2

v1 was calibrated on exllamav3's bundled generic text, because
`convert_model.py` exposes only `-cr`/`-cc` and has no option to supply a
corpus. v2 changes three things:

1. **Calibrated on our own corpus** — 2,353 documents / 3.7M tokens, self-distilled
   from the target model plus a domain blend, replacing the bundled
   wiki/c4/code text (exllamav3's random-token rows are kept, so full-vocabulary
   coverage is not lost).
2. **4x the calibration rows** (`-cr 1000`). We measured the statistics the
   quantizers actually consume: at v1's 250 rows the Hessian `XᵀX` sits at ~0.66
   of its converged value, and AWQ's per-channel means at ~0.94.
3. **AWQ smoothing re-run on 8x the data** (2.1M tokens vs 262K).

At 4.0 bpw that is worth a full bitrate tier — every disagreement bucket improved
significantly over v1 (confident z=−11.99, certain z=−14.02, moderate z=−17.12,
near-tie z=−7.30). The gain is concentrated at low bitrates: the identical change
at 5.5 bpw moves only `moderate` significantly, because a quantizer with capacity
to spare tolerates a worse Hessian.

## Measured

Disagreement against the BF16 base, bucketed by how confident the base model was
at each position (its top1−top2 logprob margin). 231 documents, 142,727 scored
positions. **`confident` is the pre-registered decider** — `near-tie` is
numerical noise and not a quality signal. `divmed` is the median
first-divergence token under free greedy decode, higher is better.

| build | size | top-1 ↑ | near-tie ↓ | moderate ↓ | **confident ↓** | certain ↓ | divmed ↑ |
|---|---|---|---|---|---|---|---|
| **TelperionAI EXL3 5.5bpw-v2** | 20 GB | 96.26% | 22.34% | 3.25% | 1.29% | 0.15% | 55  |
| TelperionAI FP8-block-AWQ | 30 GB | 96.22% | 22.83% | 3.76% | 0.79% | 0.08% | 51  |
| TelperionAI INT4-AWQ-GPTQ | 25.1 GB | 96.30% | 22.29% | 3.52% | 0.93% | 0.09% | 48  |
| TelperionAI EXL3 4.0bpw-v2 | 16 GB | 95.16% | 28.51% | 4.64% | 1.47% | 0.14% | 41  |
| Qwen3.8-27B-FP8 (official) | 30 GB | 96.15% | 22.70% | 3.48% | 1.45% | 0.08% | 47  |
| TelperionAI EXL3 4.0bpw (v1) | 16 GB | 93.82% | 31.66% | 6.94% | 2.40% | 0.58% | 28  |
| TelperionAI EXL3 5.5bpw (v1) | 20 GB | 96.12% | 22.81% | 3.52% | 1.34% | 0.16% | 55  |
| TelperionAI EXL3 6.5bpw (v1) | 23 GB | 97.10% | 17.03% | 2.41% | 1.23% | 0.15% | 59  |
| TelperionAI NVFP4 (mixed) | 24.7 GB | 93.62% | 32.25% | 8.59% | 1.85% | 0.16% | 29  |
| TelperionAI NVFP4 (full) | 24.7 GB | 93.05% | 36.65% | 8.70% | 2.00% | 0.16% | 22  |

Significance is paired McNemar over matched token positions, n = 33,812 in the
confident bucket.

Throughput is deliberately omitted: the v2 builds were benchmarked while another
quantization job held the second GPU, so their single-stream numbers understate
them and are not comparable to the v1 rows. The architecture is identical to v1
at the same bitrate.

## KV cache

Ships per-layer `k_scale` / `v_scale` in `model-kv-scales.safetensors` with
`kv_cache_scheme` under `quantization_config` — the contract llm-compressor emits
and vLLM reads. Measured on **this** checkpoint, post-RoPE, over the full 3.7M-token
corpus, which is where the V absmax stops moving. Only 16 of 64 layers are
`full_attention` and carry a cache, so this is 32 scalars in a 4 KB file.

```
--kv-cache-dtype fp8      # 32 KiB/token, from 64 KiB at fp16
```

## Notes

- **3.5 bpw was built and rejected.** It scores 2.63% confident for 1 GB less
  than 4.0 bpw's 1.47%. Only 70% of an EXL3 checkpoint is quantized layers —
  embeddings and head are 23% and fixed with respect to bitrate, because the
  248,320-token vocabulary makes them ~1.3B parameters. Below 4.0 bpw the size
  curve flattens while the damage curve keeps climbing.
- Perplexity is not shown. It is anti-correlated with behavioural damage on this
  model, so it is excluded from decisions rather than printed beside numbers that
  drive them.
