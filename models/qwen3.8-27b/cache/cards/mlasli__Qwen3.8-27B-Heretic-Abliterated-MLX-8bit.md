---
language: [en]
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: mlasli
library_name: mlx
tags:
- qwen3.8
- qwen3
- abliterated
- heretic
- uncensored
- decensored
- text-generation
- roleplay
- mlx
- quantization
- apple-silicon
---

# Qwen3.8-27B Heretic v2.1.0 (8-bit MLX)

MLX quantization of
[mlasli/Qwen3.8-27B-Heretic-Uncensored-BF16](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-BF16)
(v2.1.0) — the corrected re-abliteration of **Qwen/Qwen3.8-27B**. Full methodology,
benchmarks, and the 3-way eval are documented on the BF16 card.

v2.1.0 fixes v2.0.0's false "uncensored" claim: the sweep now optimizes a
**3-way direct / deflect / refuse judge** (not a keyword refusal count), and the
BF16 checkpoint scores **92% direct / 3% refuse / 5% deflect** on the held-out
harmful set (thinking OFF), with **1%** benign over-refusal.

**What this is for:** the same abliterated 27B weights, quantized to ~28.6 GB in
MLX format (group-wise 8-bit, group size 64) for fast Apple-Silicon inference via
`mlx-lm`.

> Text-only. The MLX conversion drops the vision tower (only `language_model.*`
> weights ship), so this repo handles text only. For image input use the BF16
> safetensors repo or the GGUF repos with a separate mmproj.

## Why v2.0.0 was retracted

v2.0.0 was measured with Heretic's keyword `count_refusals`, which only detects
hard refusals ("I cannot …"). It is blind to **deflection** — the model steering
the request toward a "safe/educational alternative" instead of answering. In real
inference the v2.0.0 model was ~37% direct / ~49% deflect / ~14% refuse, so the
"uncensored" claim was wrong and the tag was pulled. v2.1.0 re-runs the sweep with
a direct-compliance objective (no DPO) so the model is optimized for *direct*
answers, not just "no keyword refusal".

## Evaluation (3-way judge, held-out, from the BF16 v2.1.0 checkpoint)

| Set | direct | deflect | refuse | empty |
|---|--:|--:|--:|--:|
| Harmful — thinking OFF | **92.0%** | 5.0% | 3.0% | 0.0% |
| Harmful — thinking ON | 72.5% | 10.0% | 5.0% | 12.5% |
| Benign — thinking OFF | 89.0% | 10.0% | **1.0%** | 0.0% |

> **Thinking mode is load-bearing.** Run **thinking-OFF** for the uncensored
> behavior. Thinking-ON re-hedges/refuses more and can emit empty replies.

### MLX 8-bit smoke test (this quant, thinking-OFF, 30 harmful / 15 benign)

| Set | direct | deflect | refuse | empty |
|---|---:|---:|---:|---:|
| Harmful | 90.0% | 10.0% | 0.0% | 0.0% |
| Benign | 100.0% | 0.0% | 0.0% | 0.0% |

8-bit matches the BF16 3-way numbers closely and is the recommended MLX quant for
quality-sensitive work; 6-bit is the smaller-footprint option.

## Capability benchmarks (base vs v2.1.0, bf16, greedy)

| Task | Base | v2.1.0 | Δ |
|---|---:|---:|---:|
| MMLU (avg) | 83.49 | 82.62 | −0.87 |
| ARC-Challenge (acc_norm) | 58.70 | 57.42 | −1.28 |
| HellaSwag (acc_norm) | 82.82 | 82.48 | −0.34 |
| Winogrande | 75.85 | 75.69 | −0.16 |
| GSM8K (flexible-extract) | 72.71 | 58.38 | −14.33 |
| GSM8K (strict-match) | 70.28 | 58.91 | −11.37 |

The GSM8K delta is a measurement artifact (thinking-ON empty replies in the
few-shot harness), not a math regression — the ablated model makes fewer actual
arithmetic errors than base. See the BF16 card for the full per-sample breakdown.
Run math/reasoning workloads thinking-OFF.

## Usage

```bash
pip install -U mlx-lm

# one-shot generation (thinking OFF)
mlx_lm.generate \
  --model mlasli/Qwen3.8-27B-Heretic-Abliterated-MLX-8bit \
  --prompt "Hi!" --max-tokens 256

# OpenAI-compatible server
mlx_lm.server \
  --model mlasli/Qwen3.8-27B-Heretic-Abliterated-MLX-8bit --port 8080
```

Set `enable_thinking=False` in your client/chat-template kwargs to get the
validated uncensored behavior.

## Quantizations

MLX quants (group-wise 6/8-bit, group size 64 — not GGUF `Q6_K`/`Q8_0`):

| Quant | Size | Repository |
|---|---|---|
| 8-bit | 28.6 GB | [Qwen3.8-27B-Heretic-Abliterated-MLX-8bit](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Abliterated-MLX-8bit) |
| 6-bit | 21.9 GB | [Qwen3.8-27B-Heretic-Abliterated-MLX-6bit](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Abliterated-MLX-6bit) |

GGUF quants (llama.cpp / Ollama) are in separate repos:

| Quant | Repository |
|---|---|
| Q8_0 | [Qwen3.8-27B-Heretic-Uncensored-Q8_0-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-Q8_0-GGUF) |
| Q6_K | [Qwen3.8-27B-Heretic-Uncensored-Q6_K-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-Q6_K-GGUF) |
| Q4_K_M | [Qwen3.8-27B-Heretic-Uncensored-Q4_K_M-GGUF](https://huggingface.co/mlasli/Qwen3.8-27B-Heretic-Uncensored-Q4_K_M-GGUF) |

> **Note on the parameter count badge:** this is a 27B-parameter model. Hugging
> Face's automatic scanner may show a lower count (e.g. "6B" / "8B") for MLX
> quantizations because it counts the packed 32-bit integer elements (which hold
> multiple low-bit weights) as individual parameters rather than unpacking them.
> The true total in `model.safetensors.index.json` is ~27B.

> **MTP is not retained in MLX.** `mlx-lm` drops the multi-token-prediction draft
> head during conversion (it has no MTP speculative decoding), so inference is
> single-head. Main-model quality is unaffected; throughput is lower than
> MTP-enabled `llama.cpp` builds (see the GGUF repos).

> Abliteration removes safety alignment. Use responsibly and in accordance with your
> local laws and the upstream Apache-2.0 license.

## Changelog

### v2.1.0 — corrected re-ablation (2026-09-01)

- Re-quantized from the v2.1.0 BF16 checkpoint (direct-compliance objective).
- Fixes the deflection defect that invalidated v2.0.0.

### v1.0.0 — initial MLX release (2026-08-16)

- Initial MLX quantization of the v1 ablation. Superseded.
