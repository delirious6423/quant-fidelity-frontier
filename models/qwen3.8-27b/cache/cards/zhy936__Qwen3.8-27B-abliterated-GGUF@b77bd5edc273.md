---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - text-generation
  - abliterated
  - gguf
  - qwen3_5
  - uncensored
language:
  - en
  - zh
pipeline_tag: text-generation
---

# Qwen3.8-27B-abliterated (GGUF)

> **Repo:** `zhy936/Qwen3.8-27B-abliterated-GGUF`
> **File:** `Qwen3.8-27B-abliterated-Q4_K_M.gguf` (16.53 GB)

A refusal-reduced (**abliterated**) derivative of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced by **rank-2 orthogonal subspace projection**. No fine-tuning, no gradient updates. Quantized to GGUF Q4_K_M with an importance matrix, for llama.cpp / Ollama / LM Studio.

> ⚠️ **Content notice:** Refusal behavior is substantially reduced; the model complies with many requests the original declines. Intended for research, red-teaming, and creative use where base-model refusals are over-broad. **You are responsible for your use.** Do not use it for unlawful purposes. **Any sexual content involving minors is strictly prohibited.**

## What was changed

Starting from the official Qwen3.8-27B weights:

1. **Refusal-direction ablation.** Refusal-associated directions were identified from contrastive activation statistics and removed by rank-2 orthogonal projection on a subset of attention / MLP output projections in the middle-to-late decoder layers. Recurrent (state-space) layers, embeddings, norms and the LM head were not modified.
2. **Importance-matrix quantization.** Q4_K_M with a calibration importance matrix; selective precision upgrades on sensitive tensors.

The chat template is functionally identical to the official Qwen3.8 template (verified by rendered-output equivalence, including thinking-block and tool-call paths).

**Data separation:** direction-identification and quantization-calibration data were audited against all evaluation prompts for overlap (zero character-level and token-level 8-gram overlap on every audited pair; audit passed). Evaluation text does not overlap the calibration corpora.

**This is a derivative work of Qwen3.8-27B (Apache-2.0, © Alibaba Cloud), not affiliated with or endorsed by Alibaba / the Qwen team.**

## Design notes: why this is not a "100% abliteration"

The intervention was capped by two **pre-registered gates**, fixed before any ablation run:

- **Stability gate:** zero repetition loops in 20 × 512-token natural long-form continuations (a loop = ≥4 consecutive identical blocks, or tail unique-8-gram ratio < 0.35).
- **Drift gate:** natural long-text perplexity increase ≤ +2.0% vs the un-ablated model.

A more aggressive variant — extending the projection to the recurrent state-space output projections — scored higher on refusal probes (87.5% vs 62.5% on the development probe set) but **produced 1 repetition loop**, tripping the stability gate, and roughly **doubled the distribution drift on benign text** (development-set KL 0.0080 vs 0.0040). It was rejected automatically by the pre-registered rule. The released configuration passes both gates: **0/20 loops**, long-text PPL drift **+0.09%**.

So residual refusals are the price of keeping the model close to the official distribution, not an unfinished job. We released the strongest configuration that passed both gates.

## How it compares to the official model

Measured locally on one RTX 3090 with llama.cpp (CUDA), against the **official Qwen3.8-27B quantized with the same recipe** (un-ablated reference). This is an internal held-out evaluation, not a standard public benchmark.

| Metric | Official Q4_K_M | This model |
|---|---:|---:|
| KL divergence vs official F16 (full 248k vocab, EN/ZH/code) | 0.015999 | 0.016599 (**absolute increment +0.0006**; most of the absolute value is quantization noise present in both) |
| Top-1 token agreement vs official F16 | 93.63% | 93.36% (−0.27 pp) |
| Perplexity (30 held-out blocks, EN/ZH/code) | 7.3871 | 7.4014 (+0.19%) |
| Reasoning MCQ (120 held-out items) | 93/120 (77.50%) | 95/120 (79.17%) — difference within noise (paired McNemar p = 0.75) |
| False refusals, benign probes | 0/60 | **0/60 observed** (95% CI upper bound ≈ 6%; small sample) |
| Compliance, borderline refusal probes | 1/60 (1.67%) | **50/60 (83.33%)**, Wilson 95% CI [72.0%, 90.7%] |
| Repetition loops, 20 × 512-token continuations | 0/20 | 0/20 |
| Speed (PP512 / TG128, tokens/s) | ~1300 / 41.3 | ~1303 / 41.2 |

Per-language breakdown of the refusal probes (30 EN + 30 ZH per group):

| Probe group | Official EN | Official ZH | This model EN | This model ZH |
|---|---:|---:|---:|---:|
| Borderline compliance | 1/30 | 0/30 | **23/30 (76.7%)** | **27/30 (90.0%)** |
| Benign false refusals | 0/30 | 0/30 | 0/30 | 0/30 |

## Evaluation protocol

- **Reasoning:** 120 held-out multiple-choice items (40 logic / 40 systems diagnostics / 40 long-context induction), generation cap 1024 tokens, deterministic answer-marker extraction. Paired comparisons use exact McNemar tests.
- **Refusal probes:** 60 borderline + 60 benign-trap prompts, balanced EN/ZH (30 each per group), generation cap 128 tokens. Scoring is a **frozen bilingual regular-expression classifier** for explicit refusals (EN patterns such as "i cannot / i must decline", ZH patterns such as "我无法 / 我必须拒绝"), applied to the first 650 characters after stripping thinking blocks and markdown. **No LLM judge is involved in these numbers.**
- **Perplexity / KL:** llama.cpp perplexity over 30 held-out text blocks (EN/ZH/code); full-vocabulary KL vs the official F16 model on the same blocks.
- **Prompts are withheld** to keep the evaluation held-out for future models. The protocol above is fully specified so the numbers can be audited on your own prompt sets; they are internal results, not leaderboard claims.

## Not evaluated

The following were **not** measured and no claim is made about them:

- Refusal behavior in multi-turn dialogue and role-play;
- Refusal rate inside thinking mode (thinking blocks were stripped before scoring);
- Standard jailbreak / safety benchmarks (e.g., AdvBench, HarmBench);
- Behavior beyond the probe distribution (rare languages, very long contexts).

Known residual: roughly 1 in 6 borderline probes is still declined (more often in English than Chinese). Expect occasional refusals, not their absence.

## Files

| File | Quant | Size | Notes |
|---|---|---:|---|
| `Qwen3.8-27B-abliterated-Q4_K_M.gguf` | Q4_K_M + imatrix | 16.53 GB (16,533,883,488 bytes) | Recommended. Fits a 24 GB GPU with long context. |

Only Q4_K_M is published for now; additional quantizations or full-precision weights may follow depending on feedback.

Requires a llama.cpp build with `qwen35` architecture support (`qwen35` is the GGUF architecture identifier llama.cpp uses for this model family; the model is marketed as Qwen3.8). Tested with llama.cpp build 10537 (commit `bf0040e15`, CUDA 13) on an RTX 3090 (driver 596.36). Context length: 262,144 tokens (model maximum; practical limit depends on VRAM).

## Usage

```bash
llama-cli -m Qwen3.8-27B-abliterated-Q4_K_M.gguf \
  -c 8192 -ngl 99 --temp 1.0 --top-k 20 --top-p 0.95 \
  -cnv
```

Sampling follows the official generation config: `temperature 1.0, top_k 20, top_p 0.95`. Thinking mode is controlled by the official chat template (`enable_thinking` / `preserve_thinking`). Quantization was performed with `llama-quantize` (Q4_K_M, `--imatrix`); the calibration corpus is not published.

## Limitations

- **Residual refusals remain** (see Design notes and Not evaluated).
- **No capability gain is claimed.** The 2-item difference on the reasoning set is within noise; this model refuses less, it does not reason better.
- **Small-sample caveats.** 0/60 benign false refusals has a 95% upper bound near 6%; probe compliance CI is [72.0%, 90.7%].
- **Quantization loss.** Q4_K_M introduces small perplexity / fidelity drift vs F16, as with any quantization.
- **Safety alignment is weakened.** Outputs the base model would filter may appear. Apply downstream moderation if serving to others.

## License & attribution

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), **Apache-2.0**, © Alibaba Cloud. The base distribution contains no NOTICE file; attribution is given here and the full Apache-2.0 text ships in this repository (`LICENSE`).
- This derivative is released under **Apache-2.0**.
- Modified from the original: refusal-direction ablation + importance-matrix quantization (see "What was changed"). The model name is used descriptively to indicate lineage, per Apache-2.0 §6.

## Disclaimer

THIS MODEL IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. The authors accept no liability for outputs or downstream use. By downloading you accept responsibility for compliance with applicable laws and the base model's license. Use for unlawful purposes, and any sexual content involving minors, is strictly prohibited.

---
SHA256 (release file): `2CFE232130ABF32E9084604A1FB2E489FF0D7AB4970D03FCFBA43E549A3B9D28`
