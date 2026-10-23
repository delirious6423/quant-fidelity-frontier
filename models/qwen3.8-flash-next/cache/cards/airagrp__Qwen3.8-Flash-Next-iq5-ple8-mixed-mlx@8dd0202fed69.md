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

# Qwen3.8-Flash-Next — iq5 mixed MLX quant (8-bit n-gram table + 8-bit attention)

An Apple-Silicon (MLX) quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(bf16, ~335 GB) built with omlx's `quantize_oq_streaming`. The routed experts
carry 5-bit with fine-grained gs32 groups, and both the 51.2B-parameter n-gram
PLE table and attention are quantized to 8-bit — the fastest quant of this
model family on aggregate throughput.

- **Size:** 148 GB on disk
- **Context:** full native **262,144 tokens**
- **Speculative decoding:** MTP head preserved — native multi-token prediction
  active
- **Hardware:** tested on a 128 GB Apple Silicon Mac

## Quantization recipe

| Component | Precision |
|---|---|
| Routed experts (`mlp.switch_mlp`), layers 0–43 | 5-bit, gs32 (oQ) |
| Routed experts, layers 44–47 | 6-bit, gs32 (oQ) |
| N-gram PLE table | 8-bit, gs32 |
| Attention / linear_attn | 8-bit, gs32 |
| `lm_head`, `embed_tokens`, vision tower | BF16 |
| MTP head | preserved |

- **Calibration:** activation imatrix over 4,690 mixed-domain prompts,
  oQe enhanced mode.

## Benchmarks

500 items (MMLU 200 / ARC-Challenge 200 / GSM8K 100), temp 0, extended
reasoning, 8 concurrent workers on 128 GB Apple Silicon, 8-bit TurboQuant KV
cache, MTP on. Answers whose reasoning failed to parse count as incorrect.

| Task | Accuracy | 95% CI | Aggregate tok/s |
|---|---|---|---|
| MMLU | 185/200 (92.5%) | ±2.9pp | 66.8 |
| ARC-Challenge | 197/200 (98.5%) | ±1.0pp | 21.6 |
| GSM8K | 96/100 (96.0%) | ±2.4pp | 56.7 |
| **Total** | **478/500 (95.6%)** | | |

Best aggregate throughput of the family: 66.8 tok/s on MMLU vs 61.8 for the
larger iq5-mixed at twice the worker count.

### Comparison with sibling quants

Same suite; the iq5-mixed run used 16 concurrent workers, so aggregate tok/s
is not directly comparable across rows.

| | [iq5-mixed](https://huggingface.co/airagrp/Qwen3.8-Flash-Next-iq5-mixed-mlx) | **iq5-ple8-mixed (this repo)** | [iq4-mixed](https://huggingface.co/airagrp/Qwen3.8-Flash-Next-iq4-mixed-mlx) |
|---|---|---|---|
| Size | 185 GB | **148 GB** | 133 GB |
| MMLU | **187/200 (93.5%)** | 185/200 (92.5%) | 183/200 (91.5%) |
| ARC-Challenge | 198/200 (99.0%) | 197/200 (98.5%) | **199/200 (99.5%)** |
| GSM8K | 97/100 (97.0%) | 96/100 (96.0%) | 97/100 (97.0%) |
| **Total** | 482/500 | 478/500 | 479/500 |
| Aggregate tok/s (MMLU/ARC/GSM8K) | 61.8 / 13.3 / 64.6 (16 workers) | **66.8 / 21.6 / 56.7** (8 workers) | — |
| Hard-set (25 items any quant erred) | 10/22 MMLU + 0/3 ARC | 13/22 + 0/3 | 11/22 + 2/3 |

All three are statistical ties on the full suite (95% CIs ±2.5–3pp); the
hard-set is a 25-item subset where single-pass margins are within run-to-run
flip noise. Pick this repo for throughput, iq4-mixed for minimum size,
iq5-mixed for the best measured full-suite total.

## Usage

Load with any MLX runtime (mlx-lm, omlx, LM Studio):

```python
from mlx_lm import load, generate

model, tokenizer = load("airagrp/Qwen3.8-Flash-Next-iq5-ple8-mixed-mlx")
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

## License

Qwen Community License 1.0 — see [LICENSE](LICENSE). Inherits the terms of the
base model, [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
