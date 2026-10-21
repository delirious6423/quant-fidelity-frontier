---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - quantization
  - moe
  - speculative-decoding
  - mtp
  - apple-silicon
---

# Qwen3.8-Flash-Next — iq5/iq8 mixed MLX quant

An Apple-Silicon (MLX) quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(bf16, ~335 GB) built with omlx's `quantize_oq_streaming`. Bit allocation is
tuned for the MoE structure: the routed experts — ~95% of the parameters —
carry the compression, while everything quality-critical stays high precision.

> [!Important]
> This is the **gs64/gs32 expert** recipe (see table below). It is distinct
> from [airagrp/Qwen3.8-Flash-Next-iq5-iq8-mixed-mlx](https://huggingface.co/airagrp/Qwen3.8-Flash-Next-iq5-iq8-mixed-mlx),
> which carries the earlier gs128-expert allocation.

- **Size:** 185 GB on disk (~77 GB resident; the 102 GB n-gram PLE table is
  SSD-offloaded)
- **Context:** full native **262,144 tokens** (verified: 249,688-token prompt
  prefilled at ~1,366 tok/s with deep needle retrieval)
- **Speculative decoding:** MTP head preserved — native multi-token prediction
  active (2.5–3 tok/cycle observed)
- **Hardware:** tested on a 128 GB Apple Silicon Mac; 96 GB+ recommended

## Quantization recipe

| Component | Precision |
|---|---|
| Routed experts, layers 0–43 | 5-bit, gs64 (oQ) |
| Routed experts, layers 44–47 | 8-bit, gs32 (oQ) |
| Attention / indexer / mixers | 8-bit (stock oQ5 predicate: mxfp8 + sensitive layers +1 bit) |
| Shared experts | 8-bit |
| `lm_head`, `embed_tokens` | BF16 |
| N-gram PLE table | BF16 (SSD-offloadable) |
| MTP head | preserved |

- **Calibration:** activation imatrix over 4,690 mixed-domain prompts
  (~1024 × 512-token windows), oQe enhanced mode.
- **MTP:** the multi-token-prediction head is kept intact, so native
  speculative decoding works out of the box.

## Benchmarks

500 items (MMLU 200 / ARC-Challenge 200 / GSM8K 100), temp 0, extended
reasoning, 16 concurrent requests on 128 GB Apple Silicon, 8-bit TurboQuant
KV cache, MTP on:

| Task | Accuracy | 95% CI |
|---|---|---|
| MMLU | 187/200 (93.5%) | ±2.7pp |
| ARC-Challenge | 198/200 (99.0%) | ±0.7pp |
| GSM8K | 97/100 (97.0%) | ±2.0pp |
| **Total** | **482/500 (96.4%)** | |

Throughput: ~5 tok/s decode per request under 16-way load; **~62–65 tok/s
aggregate**. Single-stream decode ~63 tok/s, TTFT 0.2–0.4 s.

This quant was selected as the best quality/speed trade-off against two
sibling recipes on the same suite: an all-4-bit expert variant (481/500) and a
4-bit/6-bit/8-bit variant (478/500) — both scored lower *and* decoded slower.

## Usage

Load with any MLX runtime (mlx-lm, omlx, LM Studio):

```python
from mlx_lm import load, generate

model, tokenizer = load("airagrp/Qwen3.8-Flash-Next-iq5-mixed-mlx")
```

For omlx serving, the recommended model settings are:

```json
{
  "turboquant_kv_enabled": true,
  "turboquant_kv_bits": 8,
  "max_context_window": 262144,
  "mtp_enabled": true
}
```

> [!Note]
> The repo requires a runtime with `qwen4_exp` support (mlx-lm / omlx ≥ 0.7.x).
> With 8-bit KV cache, full 262k context fits comfortably on a 128 GB machine;
> allow ~110 GB of free SSD for the n-gram table cache.

## License

Qwen Community License 1.0 — see [LICENSE](LICENSE). Inherits the terms of the
base model, [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
