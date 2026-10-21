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

## Flight Simulator — Coded by This Model

A real-world coding benchmark: each model was prompted to write a complete flight simulator from scratch. The resulting code was rendered and recorded.

![Flight Simulator](flightsimulator.gif)


## About XL Quantization

**BaseQuant_XL** is a fully **data-agnostic**, static quantization. No calibration dataset, no sensitivity analysis, no importance matrix. Precision is allocated purely by architectural role — routing-critical layers get higher precision, bulk expert parameters get lower precision. The result is a transparent, faithful capture of the source model.

Data-dependent calibration quantizations (iMatrix, AWQ, GPTQ, oQ, oQ4e, etc.) use a calibration set to guide bit allocation. This can produce a **skewed representation** of the model: domains well-represented in the calibration data (English, popular topics, public or leaked benchmarks) are preserved better, while underrepresented domains (non-English languages, niche use cases, your own data) are preserved worse. XL avoids this trade-off entirely — it generalizes honestly because it is never fit to any particular data distribution.

## Thinking-Mode MATHQA: Gemma vs Qwen3.6 Token Efficiency

Both models were tested on the corrected custom MATHQA test set (n=30, thinking mode, 8192-token budget). Both verified at **30/30 (100%)** after re-test of failures. The decisive difference is *how efficiently* they arrive at correct answers.

![Gemma vs Qwen Efficiency](gemma_vs_qwen_efficiency.png)

| Model | Accuracy (verified) | Reasoning Chars / Correct |
|-------|:---:|---------------------------|
| **Gemma-4 26B 6bit-XL** | **30/30 (100%)** | **3,527** |
| Qwen3.6-35B 6bit-XL | **30/30 (100%)** | 4,948 |

**Gemma uses 29% fewer reasoning chars per correct answer** than Qwen3.6 (verified results after re-test of failures). Gemma's shorter reasoning chains mean:

- **Less context window consumed** — more space for multi-turn conversation or complex prompts
- **Less energy per answer** — fewer GPU cycles wasted on self-verification
- **More decisive reasoning** — Gemma commits to answers sooner; Qwen3.6 tends to verify and re-derive

The per-question scatter shows Gemma's reasoning clustered at 1K–4K chars while Qwen3.6 spreads from 2K–12K. On the hardest questions (Q2232, Q2932), both models spike to 11K–19K chars — the efficiency gap closes when the problem genuinely requires extended reasoning.

> ⚠️ Raw accuracy differences are dominated by sampling variance (temp=1.0) and answer-extraction artifacts. Both models are capable of 30/30. The real differentiator is reasoning efficiency — **correct answers per thousand reasoning chars**.

## Local SOTA for 48GB Macs — Intelligence Benchmark Comparison

This model is part of a benchmark comparison of the best local MLX-quantized LLMs that fit in 48GB unified memory on Apple Silicon. All benchmarks run in instruct mode (no thinking) with n=50 samples per benchmark.

![SOTA Comparison](sota_comparison_instruct.png)

| Benchmark | Samples | Agents-A1 6bit-XL | Gemma-4 26B 6bit-XL | Huihui-Qwen3.6 6bit-XL | Ornith-35B 6bit-XL | Qwen3.6-27B oQ4e | Qwen3.6-35B 6bit-XL | Qwen3.6-35B oQ4e | Qwen3.6-35B oQ4e-XL | Qwen3.6-35B oQ6 |
|-----------|:-------:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MMLU | 50/14042 | 66% | **76%** | 74% | 64% | 74% | 64% | 66% | 72% | 64% |
| MMLU_PRO | 50/12032 | 58% | **82%** | 66% | 66% | 56% | 64% | 60% | 64% | 60% |
| ARC_CHALLENGE | 50/1172 | 90% | 90% | **92%** | **92%** | 88% | 90% | **92%** | **92%** | 90% |
| HUMANEVAL | 50/164 | 90% | **98%** | 84% | 78% | **92%** | 78% | **92%** | 90% | 66% |
| MBPP | 50/500 | 70% | 82% | 78% | 78% | **86%** | 78% | 80% | 76% | 76% |
| **Average** | | 74.8% | **85.6%** | 78.8% | 75.6% | 79.2% | 74.8% | 78.0% | 78.8% | 71.2% |

**Collection:** [Local SOTA for 48GB Macs](https://huggingface.co/collections/leonsarmiento/local-sota-for-48gb-macs-6a5fb58390dd01e1fc35d55e)

> ⚠️ n=50 sampling means wide confidence intervals (±~13% at 95% CI). Differences under ~6 points may not be statistically significant. Models using data-aware quantization (oQ/oQe) may be calibrated on benchmark-like data — their scores carry a benchmaxxing caveat. The BaseQuant_XL variants (data-agnostic) provide the most honest generalization estimates.

# leonsarmiento/gemma-4-26B-A4B-it-6bit-XL-mlx

This model was converted to MLX format from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) using **BaseQuant_XL 6/8-bit mixed quantization** optimized for Apple Silicon. The vision encoder is preserved and quantized at 6-bit, making this a full multimodal model.

**BaseQuant_XL** keeps the most routing-critical layers in full bf16 precision — the MoE `router.proj` (gate) and the `mlp` (shared/dense expert that processes all tokens) — while applying aggressive quantization to the bulk parameters. Note: `tie_word_embeddings=True` means there is no separate `lm_head` (tied to `embed_tokens`).

Gemma-4-26B-A4B is a 25.2B-parameter MoE (Mixture of Experts) model with 128 experts per layer (avg. 3.8B active parameters per token) and a vision encoder.

## Chat Template

The `chat_template.jinja` and `tokenizer_config.json` were synced to the latest canonical Gemma 4 chat template from Google (published 2025-07-09) at quantization time. This update fixes tool-calling loops, turn closures, and thinking content-ordering issues. See [google/gemma-4-26B-A4B-it/chat_template.jinja](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/chat_template.jinja) for the upstream source.

## Quantization Details

| Layer | Bits | Group Size |
|-------|------|------------|
| `router.proj` (gate) | bf16 | — |
| `mlp` (shared/dense expert) | bf16 | — |
| Vision tower (full) | bf16 | — |
| `embed_tokens` | 8 | 64 |
| `self_attn` | 8 | 64 |
| `experts.switch_glu` (routed experts) | 6 | 64 |
| Default fallback | 8 | 64 |

- **Quantization type**: BaseQuant_XL mixed (multimodal, vision preserved)
- **Bits per weight**: 7.061
- **Total size**: ~21 GB (5 shards)
- **Group size**: 64
- **Method**: Custom `quant_predicate` via `mlx_vlm`
- **Note**: `tie_word_embeddings=True` — no separate lm_head

## BaseQuant_XL Strategy

| Bit Depth | Layers | Rationale |
|-----------|--------|-----------|
| **bf16 (unquantized)** | `router.proj` (gate), `mlp` (shared/dense expert), vision tower | Routing decisions and the dense expert that processes every token — errors here cascade through all tokens |
| **8-bit** | `embed_tokens`, `self_attn` | Every-token layers with moderate sensitivity — 8-bit is near-lossless |
| **6-bit** | `experts.switch_glu` (routed experts) | Bulk of parameters — only a subset of experts active per token |

## Intelligence Benchmarks

| Benchmark | n | XL 6/8 (this) | 6-bit uniform | 8-bit uniform | QAT 4-bit |
|-----------|---|:-------------:|:-------------:|:-------------:|:---------:|
| MMLU | 50 | 76.0% | 76.0% | 76.0% | 76.0% |
| MMLU_PRO | 50 | **80.0%** | 78.0% | **82.0%** | 72.0% |
| ARC Challenge | 50 | 90.0% | 90.0% | 90.0% | 90.0% |
| HUMANEVAL | 30 | 96.7% | 96.7% | 96.7% | 96.7% |
| MBPP | 30 | **80.0%** | 76.7% | **80.0%** | **86.7%** |
| LIVECODEBENCH | 30 | 56.7% | **60.0%** | 50.0% | 43.3% |
| MATHQA (custom)¹ | 30 | **66.7%** | 56.7% | 60.0% | **70.0%** |

> ¹ Custom relaxed MATHQA test (see methodology below). Standard benchmark MATHQA/GSM8K results were omitted because time-constrained math benchmarks conflate throughput with reasoning quality — see [Curated MATHQA Methodology](#custom-mathqa-methodology).
>
> **Analysis**: XL 6/8 at 21GB matches or beats the 8-bit uniform (26GB) on knowledge and code-MC benchmarks — MMLU_PRO (+2 over 6-bit), MBPP (+3.3 over 6-bit), LIVECODEBENCH (+6.7 over 8-bit). On the custom MATHQA test with relaxed token budget, the XL trails only the QAT 4-bit and beats both uniform variants.

## Custom MATHQA Methodology

Standard math benchmarks (run via LM Studio's evaluation harness with fixed time/token budgets) produced results that conflated **generation throughput** with **reasoning quality**. The XL's bf16 router/shared-expert layers make it 1.2x–5.5x slower per answer than the uniform variants, causing it to exhaust the benchmark budget on harder problems — **100% of the XL's failures were timeouts, with zero arithmetic, logic, or formatting errors**.

To get a fairer picture of reasoning quality, a custom curated MATHQA test was constructed:

### Test Set Construction (n=30)

1. **Failure-enriched sample**: All 19 MATHQA questions where either the XL or the 6-bit uniform produced an empty/timeout response in the original benchmark — these are the "hard" cases where time limits matter most.
2. **Control group**: 11 questions both models solved correctly in the original run, sampled randomly. If a model starts missing these, it indicates a real reasoning issue.
3. **Relaxed constraints**: 8192 max tokens (vs. the original ~2048 thinking limit), no server-side timeout. Each model generates as long as it needs.
4. **All 4 variants tested**: QAT 4-bit, 6-bit uniform, 8-bit uniform, XL 6/8 — same questions, same prompt, same parameters (temperature 1.0, top_p 0.95).

### Detailed Results

| Model | Accuracy | Avg Time | Avg Response Len |
|-------|:-:|:-:|:-:|
| QAT 4-bit | **70.0%** (21/30) | 69.9s | 5,916 |
| XL 6/8 (this) | **66.7%** (20/30) | 100.3s | 5,894 |
| 8-bit uniform | 60.0% (18/30) | 102.1s | 7,070 |
| 6-bit uniform | 56.7% (17/30) | 81.9s | 5,969 |

All 11 control-group questions were answered correctly by all 4 models — none of the variants exhibit a real reasoning deficit on solvable problems.

## Recommended Inference Parameters

| Parameter | Value |
|-----------|-------|
| `temperature` | 1.0 |
| `top_p` | 1.0 |
| `top_k` | 64 |
| `min_p` | 0.01 |

## Reasoning Parsing in LM Studio and Others

- **Start string**: `<|channel>thought` 
- **End string**: `<channel|>`

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model leonsarmiento/gemma-4-26B-A4B-it-6bit-XL-mlx --max-tokens 256 --temperature 1.0 --top-p 1.0 --prompt "Hello"
```
