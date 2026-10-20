---
library_name: mlx
base_model: google/gemma-4-26B-A4B-it
tags:
- mlx
- mlx-vlm
- moe
- multimodal
- vision
- gemma
- basequant-xl
pipeline_tag: image-text-to-text
---

# leonsarmiento/gemma-4-26B-A4B-it-8bit-XL-mlx

This model was converted to MLX format from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) using **BaseQuant_XL 8-bit mixed quantization** — the maximum-fidelity XL variant. The vision encoder is preserved (full multimodal). The unquantized BF16 base was used as the source for maximum fidelity.

**BaseQuant_XL** keeps the most routing-critical layers in full bf16 precision — the MoE `router.proj` (gate) and the `mlp` (shared/dense expert that processes all tokens) — while applying 8-bit quantization to everything else. Compared to the [6-bit XL variant](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-6bit-XL-mlx), only the routed experts (`experts.switch_glu`) move from 6-bit to 8-bit; all other layer assignments are identical.

> **⚠️ Honest assessment:** While this variant ties for best on MMLU_PRO (82.0%) — confirming the 8-bit routed experts improve knowledge recall — it **underperforms the 6-bit XL on math reasoning** (see Custom MATHQA below). On the failure-enriched MATHQA test, the 8-bit XL scored 56.7% vs the 6-bit XL's 66.7%, and produced the highest average response length (8,789 chars vs 5,894). For reasoning-focused workloads, the [6-bit XL](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-6bit-XL-mlx) remains the recommended XL variant. This model is the **knowledge specialist** — use it when broad factual recall matters more than step-by-step math.

Note: `tie_word_embeddings=True` means there is no separate `lm_head` (tied to `embed_tokens`).

Gemma-4-26B-A4B is a 25.2B-parameter MoE (Mixture of Experts) model with 128 experts per layer (avg. 3.8B active parameters per token) and a vision encoder.


## About XL Quantization

**BaseQuant_XL** is a fully **data-agnostic**, static quantization. No calibration dataset, no sensitivity analysis, no importance matrix. Precision is allocated purely by architectural role — routing-critical layers get higher precision, bulk expert parameters get lower precision. The result is a transparent, faithful capture of the source model.

Data-dependent calibration quantizations (iMatrix, AWQ, GPTQ, oQ, oQ4e, etc.) use a calibration set to guide bit allocation. This can produce a **skewed representation** of the model: domains well-represented in the calibration data (English, popular topics, public or leaked benchmarks) are preserved better, while underrepresented domains (non-English languages, niche use cases, your own data) are preserved worse. XL avoids this trade-off entirely — it generalizes honestly because it is never fit to any particular data distribution.

## Intelligence Benchmarks

All variants of `google/gemma-4-26B-A4B-it` quantized to MLX, run under identical conditions on the same hardware.

### Instruct-mode benchmarks (no thinking)

| Benchmark | Sample | [QAT 4-bit](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-QAT-MLX-4bit) | [6-bit (6/8 mixed)](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-MLX-6bit) | [8-bit uniform](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-MLX-8bit) | [6-bit XL](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-6bit-XL-mlx) | **8-bit XL (this)** |
|-----------|--------|------|------|------|------|------|
| MMLU            | 50/14042 | 76.0% | 76.0% | 76.0% | 76.0% | **76.0%** |
| MMLU_PRO        | 50/12032 | 72.0% | 78.0% | **82.0%** | 80.0% | **82.0%** |
| ARC_CHALLENGE   | 50/1172  | **90.0%** | **90.0%** | **90.0%** | **90.0%** | 88.0% |
| HUMANEVAL       | 30/164   | 96.7% | 96.7% | 96.7% | 96.7% | **96.7%** |
| MBPP            | 30/500   | **86.7%** | 76.7% | 80.0% | 80.0% | 80.0% |
| LIVECODEBENCH   | 30/1055  | 43.3% | **60.0%** | 50.0% | 56.7% | 56.7% |

### Custom MATHQA (thinking mode, relaxed 8192-token budget)

A failure-enriched MATHQA test set (n=30) run with thinking enabled and relaxed token limits to isolate reasoning quality from throughput artifacts.

| Model | Accuracy | Avg Time | Avg Response Len |
|-------|:-:|:-:|:-:|
| [QAT 4-bit](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-QAT-MLX-4bit) | **70.0%** (21/30) | 69.9s | 5,916 |
| [6-bit XL](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-6bit-XL-mlx) | **66.7%** (20/30) | 100.3s | 5,894 |
| [8-bit uniform](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-MLX-8bit) | 60.0% (18/30) | 102.1s | 7,070 |
| [6-bit uniform](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-MLX-6bit) | 56.7% (17/30) | 81.9s | 5,969 |
| **8-bit XL (this)** | 56.7% (17/30) | 92.6s | **8,789** |

> **⚠️ Dataset quality caveat:** This benchmark was run with the failure-enriched MATHQA test set before corrections. A full re-audit (2026-07-21) found 9 of 19 "hard" questions had corrupted text, wrong expected answers, or unsolvable constraints (e.g., dropped digits, wrong combinatorial formulas, P(A∩B)>P(A)). All 9 have since been corrected in `gemma-4-results/test_sets.json` — the test set is now mathematically sound. The scores below were obtained with the uncorrected set and carry this caveat. See the [6-bit XL model card](https://huggingface.co/leonsarmiento/gemma-4-26B-A4B-it-6bit-XL-mlx) for methodology details.

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model leonsarmiento/gemma-4-26B-A4B-it-8bit-XL-mlx --max-tokens 256 --temperature 1.0 --top-p 1.0 --prompt "Hello"
```

## BaseQuant_XL Quantization Strategy

| Bit Depth | Layers | Rationale |
|-----------|--------|-----------|
| **bf16 (unquantized)** | `router.proj` (gate), `mlp` (shared/dense expert), vision tower | Routing decisions and the dense expert that processes every token — errors here cascade through all tokens |
| **8-bit** | `embed_tokens`, `self_attn`, `experts.switch_glu` (routed experts) | Near-lossless — the max-fidelity XL recipe. Only the routing-critical bf16 paths are kept at full precision |

### Quantization Details

| Layer | Bits | Group Size |
|-------|------|------------|
| `router.proj` (gate) | bf16 | — |
| `mlp` (shared/dense expert) | bf16 | — |
| Vision tower (full) | bf16 | — |
| `embed_tokens` | 8 | 64 |
| `self_attn` | 8 | 64 |
| `experts.switch_glu` (routed experts) | 8 | 64 |
| Default fallback | 8 | 64 |

- **Quantization type**: BaseQuant_XL mixed (multimodal, vision preserved)
- **Bits per weight**: 8.833
- **Total size**: ~27 GB (6 shards)
- **Group size**: 64
- **Method**: Custom `quant_predicate` via `mlx_vlm`
- **Note**: `tie_word_embeddings=True` — no separate lm_head

## Recommended Inference Parameters

| Parameter | Value |
|-----------|-------|
| `temperature` | 1.0 |
| `top_p` | 1.0 |
| `top_k` | 64 |
| `min_p` | 0.01 |
| `repeat_penalty` | 1.1 |

## Chat Template

The chat template is synced to the **canonical Google upstream** version from [`google/gemma-4-26B-A4B-it/chat_template.jinja`](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/chat_template.jinja) (last verified 2025-07-09). Do not rely on the HF cache copy — it may be stale.

Key markers in the latest template: `preserve_thinking`, `continues_into_next`, `prev_non_tool_role`, null handling in `format_argument`.

The template is injected into both `chat_template.jinja` (standalone file) and `tokenizer_config.json` (`chat_template` JSON field).
