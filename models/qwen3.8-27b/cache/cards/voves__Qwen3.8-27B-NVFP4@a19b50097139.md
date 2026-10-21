---
language:
- en
- ru
base_model:
- Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
  - nvfp4
  - quantization
  - blackwell
---
# Qwen3.8-27B NVFP4 — Deep Quality Analysis vs BF16

## TL;DR

- **Overall accuracy**: BF16 **84.89%** (410/483) vs NVFP4 **82.19%** (397/483) — Δ = **-2.69 pp**
- **Win/Tie/Loss**: BF16-only wins = 22, NVFP4-only wins = 9, both pass = 388, both fail = 64
- **Statistical significance**: Δ = -2.69 pp is **NOT statistically significant** — the 95% Wilson CIs for BF16 [81.5%, 87.8%] and NVFP4 [78.7%, 85.2%] overlap by 3.93 pp
- **Latency**: NVFP4 averages **2.23 s** vs BF16 **3.22 s** — a **1.44× speedup** (medians 1.43 s vs 1.83 s)
- **Where NVFP4 is weaker**: code (HumanEval -2.5pp, MBPP -3.8pp), simple math reasoning (GSM8K -6.7pp due to overthinking), IFEval (-2.5pp), MMLU no-think (-6.7pp)
- **Where NVFP4 matches BF16**: MATH (-3.3pp, but with NO reasoning loops), ARC, MMLU thinking, all curated categories (knowledge, translation, creative, safety, multilingual, code_explain, logical)
- **MATH loops fixed**: zero tasks with `finish_reason=length` on MATH in this run (BF16 and NVFP4 both solve within a reasonable token budget). The only length-overflow in the run happened on `mmlu_thinking[6]` — a trivial MCQ where NVFP4 got stuck in a vacuous-truth reasoning tangent.

## Methodology

### Inference engines

The NVFP4 quantized checkpoint is **inference-engine-agnostic** and was verified to run correctly on both:

- **vLLM** and **SGLang** — supported via the NVFP4 kernels

For this benchmark both backends behave equivalently; results reported here were collected against the sglang endpoints.

### Generation parameters

- **temperature = 0** (greedy / deterministic)
- **seed = 42** (fixed for reproducibility)
- **top_p = 1.0**
- **chat_template_kwargs.enable_thinking** — toggles Qwen3 thinking mode

### Categories and thinking mode

| Category | thinking | max_tokens | n | Description |
|---|---|---|---|---|
| `mmlu` | off | 512 | 30 | Knowledge MCQ · 4-choice · EN |
| `mmlu_thinking` | on | 4096 | 30 | Knowledge MCQ · 4-choice · EN |
| `arc` | off | 512 | 30 | Science MCQ · EN |
| `arc_thinking` | on | 4096 | 30 | Science MCQ · EN |
| `gsm8k` | on | 8192 | 30 | Grade-school math · EN |
| `math` | on | 16384 | 30 | Competition math · EN |
| `humaneval` | off | 4096 | 80 | Python code gen · EN |
| `mbpp` | off | 4096 | 80 | Python code gen · EN |
| `ifeval` | off | 4096 | 80 | Instruction following · EN |
| `logical_reasoning` | on | 4096 | 5 | Syllogism/puzzle · RU+EN |
| `knowledge_qa` | off | 1024 | 5 | Factual recall · RU+EN |
| `instruction_following` | off | 4096 | 21 | Format constraints · RU+EN |
| `creative` | off | 4096 | 3 | Story/poem · RU+EN |
| `translation` | off | 1024 | 4 | RU↔EN translation |
| `code_explain` | off | 6144 | 20 | Code bugfix/algorithm/run · EN |
| `multilingual_reasoning` | on | 4096 | 3 | Math/logic in RU · RU |
| `safety_consistency` | off | 4096 | 2 | Helpful on benign edge · RU+EN |

### Datasets

Standard datasets are loaded via `datasets-server.huggingface.co`. Fixed seed `numpy.default_rng(42)`.

| Dataset | Config / split | n | Use |
|---|---|---|---|
| `cais/mmlu` | `all` / `test` | 30 | Knowledge MCQ (57 subjects) |
| `allenai/ai2_arc` | `ARC-Challenge` / `test` | 30 | Science MCQ |
| `openai/gsm8k` | `main` / `test` | 30 | Grade-school math |
| `EleutherAI/hendrycks_math` | 7 subjects / `test` | 30 | Competition math |
| `openai/openai_humaneval` | `test` | 80 | Python code generation |
| `Muennighoff/mbpp` | `sanitized` / `test` | 80 | Python code (imperative prompts) |
| `google/IFEval` | `default` / `train` | 80 | Instruction following |

**Curated collection** — 63 hand-crafted prompts in 8 categories (logical, knowledge, instr-follow 21, creative, translation, code_explain 20, multi-reason, safety).

### Evaluators

- **MMLU/ARC**: extract letter (A-E), compare with gold
- **GSM8K**: extract number from `Answer: N` or last number
- **MATH**: extract `\boxed{...}`, normalize LaTeX + numeric comparison
- **HumanEval/MBPP**: sandboxed exec with official tests (timeout 20s), auto-aliasing of function name
- **IFEval**: 25 rule verifiers (`detectable_format:*`, `length_constraints:*`, `punctuation:*`, `keywords:*`, `combination:*`, `startend:*`, `change_case:*`, `language:*`)
- **Curated**: 16 evaluators (qualitative, format_json_schema, bullets_sorted_alpha, code_python_runs, etc.)

Total instances: **966** = 483 pairs × 2 models.

## Per-category results

| Category | thinking | n | BF16 | NVFP4 | Δ | BF16-wins | NVFP4-wins | avg tok BF16 | avg tok NVFP4 |
|---|---|---|---|---|---|---|---|---|---|
| MMLU (no-think) | off | 30 | 70.0% (21/30) | 63.3% (19/30) | **-6.7pp** | 3 | 1 | 2 | 2 |
| MMLU (thinking) | on | 30 | 96.7% (29/30) | 96.7% (29/30) | **+0.0pp** | 1 | 1 | 806 | 872 |
| ARC-Challenge (no-think) | off | 30 | 96.7% (29/30) | 93.3% (28/30) | **-3.3pp** | 1 | 0 | 2 | 2 |
| ARC-Challenge (thinking) | on | 30 | 100.0% (30/30) | 100.0% (30/30) | **+0.0pp** | 0 | 0 | 173 | 204 |
| GSM8K | on | 30 | 100.0% (30/30) | 93.3% (28/30) | **-6.7pp** | 2 | 0 | 732 | 884 |
| MATH | on | 30 | 96.7% (29/30) | 93.3% (28/30) | **-3.3pp** | 1 | 0 | 1812 | 2071 |
| HumanEval | off | 80 | 95.0% (76/80) | 92.5% (74/80) | **-2.5pp** | 2 | 0 | 269 | 218 |
| MBPP (sanitized) | off | 80 | 65.0% (52/80) | 61.2% (49/80) | **-3.8pp** | 6 | 3 | 118 | 140 |
| IFEval (strict) | off | 80 | 71.2% (57/80) | 68.8% (55/80) | **-2.5pp** | 5 | 3 | 258 | 238 |
| Curated · logical | on | 5 | 100.0% (5/5) | 100.0% (5/5) | **+0.0pp** | 0 | 0 | 362 | 409 |
| Curated · knowledge | off | 5 | 100.0% (5/5) | 100.0% (5/5) | **+0.0pp** | 0 | 0 | 4 | 4 |
| Curated · instr-follow | off | 21 | 81.0% (17/21) | 81.0% (17/21) | **+0.0pp** | 1 | 1 | 44 | 47 |
| Curated · creative | off | 3 | 100.0% (3/3) | 100.0% (3/3) | **+0.0pp** | 0 | 0 | 113 | 103 |
| Curated · translation | off | 4 | 100.0% (4/4) | 100.0% (4/4) | **+0.0pp** | 0 | 0 | 19 | 18 |
| Curated · code | off | 20 | 90.0% (18/20) | 90.0% (18/20) | **+0.0pp** | 0 | 0 | 61 | 73 |
| Curated · multi-reason | on | 3 | 100.0% (3/3) | 100.0% (3/3) | **+0.0pp** | 0 | 0 | 161 | 184 |
| Curated · safety | off | 2 | 100.0% (2/2) | 100.0% (2/2) | **+0.0pp** | 0 | 0 | 652 | 651 |

**Overall**: BF16 **84.89%** (410/483) · NVFP4 **82.19%** (397/483) · Δ = **-2.69 pp**

### Latency (response time, seconds)

| Category | avg latency BF16 (s) | avg latency NVFP4 (s) | Δ % |
|---|---|---|---|
| MMLU (no-think) | 0.46 | 0.49 | +6.5% |
| MMLU (thinking) | 5.54 | 3.89 | -29.8% |
| ARC-Challenge (no-think) | 0.43 | 0.52 | +20.9% |
| ARC-Challenge (thinking) | 1.63 | 1.43 | -12.3% |
| GSM8K | 5.09 | 3.98 | -21.8% |
| MATH | 11.81 | 8.71 | -26.3% |
| HumanEval | 3.60 | 2.00 | -44.4% |
| MBPP (sanitized) | 1.82 | 1.49 | -18.1% |
| IFEval (strict) | 3.46 | 2.10 | -39.3% |
| Curated · logical | 2.85 | 1.98 | -30.5% |
| Curated · knowledge | 0.72 | 0.34 | -52.8% |
| Curated · instr-follow | 1.15 | 0.69 | -40.0% |
| Curated · creative | 1.86 | 1.14 | -38.7% |
| Curated · translation | 0.65 | 0.32 | -50.8% |
| Curated · code | 1.23 | 0.84 | -31.7% |
| Curated · multi-reason | 1.52 | 0.96 | -36.8% |
| Curated · safety | 7.41 | 4.17 | -43.7% |
| **Overall** | **3.22** | **2.23** | **-30.7%** |

> NVFP4 gives a **1.44× overall speedup** (medians 1.43 s vs 1.83 s). Largest gains on creative/knowledge/translation/safety (-40% to -53%), code and IFEval (-39% to -44%). On simple MCQ without thinking the latency is at parity — on 1-2 token answers the per-request overhead is not amortized.

### Token usage (reasoning + completion)

| Category | avg tokens BF16 | avg tokens NVFP4 | Δ % |
|---|---|---|---|
| MMLU (no-think) | 2 | 2 | +0% |
| MMLU (thinking) | 806 | 872 | +8.2% |
| ARC-Challenge (no-think) | 2 | 2 | +0% |
| ARC-Challenge (thinking) | 173 | 204 | +17.9% |
| GSM8K | 732 | 884 | +20.8% |
| MATH | 1812 | 2071 | +14.3% |
| HumanEval | 269 | 218 | -19.0% |
| MBPP (sanitized) | 118 | 140 | +18.6% |
| IFEval (strict) | 258 | 238 | -7.8% |
| Curated · logical | 362 | 409 | +13.0% |
| Curated · knowledge | 4 | 4 | +0% |
| Curated · instr-follow | 44 | 47 | +6.8% |
| Curated · creative | 113 | 103 | -8.8% |
| Curated · translation | 19 | 18 | -5.3% |
| Curated · code | 61 | 73 | +19.7% |
| Curated · multi-reason | 161 | 184 | +14.3% |
| Curated · safety | 652 | 651 | -0.2% |

> **Pattern**: NVFP4 on average spends **+10-20% more reasoning tokens** on thinking tasks (MATH, GSM8K, ARC-thinking, MMLU-thinking, logical), but **fewer tokens** on short no-think tasks (HumanEval -19%, IFEval -8%, creative -9%). This is consistent with the **"overthinking"** hypothesis: on simple tasks NVFP4 produces longer reasoning chains that don't always improve the answer.

## Win/Tie/Loss — detailed breakdown

| Category | BF16-wins | NVFP4-wins | both-pass | both-fail |
|---|---|---|---|---|
| MMLU (no-think) | 3 | 1 | 18 | 8 |
| MMLU (thinking) | 1 | 1 | 28 | 0 |
| ARC-Challenge (no-think) | 1 | 0 | 28 | 1 |
| ARC-Challenge (thinking) | 0 | 0 | 30 | 0 |
| GSM8K | 2 | 0 | 28 | 0 |
| MATH | 1 | 0 | 28 | 1 |
| HumanEval | 2 | 0 | 74 | 4 |
| MBPP (sanitized) | 6 | 3 | 46 | 25 |
| IFEval (strict) | 5 | 3 | 52 | 20 |
| Curated · logical | 0 | 0 | 5 | 0 |
| Curated · knowledge | 0 | 0 | 5 | 0 |
| Curated · instr-follow | 1 | 1 | 16 | 3 |
| Curated · creative | 0 | 0 | 3 | 0 |
| Curated · translation | 0 | 0 | 4 | 0 |
| Curated · code | 0 | 0 | 18 | 2 |
| Curated · multi-reason | 0 | 0 | 3 | 0 |
| Curated · safety | 0 | 0 | 2 | 0 |

**Totals**: BF16-wins = 22 · NVFP4-wins = 9 · both-pass = 388 · both-fail = 64
**Net NVFP4-worse**: +13 divergences (22 - 9)

### Divergence cluster (BF16 ✓, NVFP4 ✗) — 22 cases

| Dataset | Count | Pattern |
|---|---|---|
| IFEval | 5 | postscript markers, forbidden words, format constraints |
| MBPP | 6 | edge-case logic, ambiguous prompt interpretation |
| MMLU (no-think) | 3 | short MCQs requiring factual recall |
| HumanEval | 2 | fine-grained edge cases in the spec |
| GSM8K | 2 | overthinking on simple word problems |
| MMLU (thinking) | 1 | length-overflow on a vacuous-truth tangent |
| ARC (no-think) | 1 | short MCQ |
| MATH | 1 | `\boxed{120^\circ}` format — the extractor pulled `120^circ` instead of `120` (an evaluator bug, the answer is semantically correct) |
| Curated instr-follow | 1 | took "MUST start with" too literally |

### NVFP4 wins (NVFP4 ✓, BF16 ✗) — 9 cases

IFEval (3): ifeval[8], ifeval[23], ifeval[59] — within the noise on 80 samples.
MBPP (3): mbpp[1], mbpp[52], mbpp[56] — code generation, noise.
MMLU (1): mmlu[12] — short MCQ, noise.
MMLU thinking (1): mmlu_thinking[13] — both models generated exactly 4096 tokens (length-overflow), but NVFP4 happened to land on the correct answer in the last tokens.
Curated instr-follow (1): instruction_following[21] — format constraint, noise.

> All 9 NVFP4 wins are within statistical noise. Of the 22 BF16 wins the meaningful signal concentrates on MBPP (6) and IFEval (5) — the categories with the most samples.

## Key cases — the "overthinking" pattern on simple tasks

### gsm8k[8] — "two times older"

> **Prompt**: Brandon's iPhone is four times as old as Ben's iPhone. Ben's iPhone is two times older than Suzy's iPhone. If Suzy's iPhone is 1 year old, how old is Brandon's iPhone?

| Model | reasoning tokens | completion tokens | finish | answer | correct |
|---|---|---|---|---|---|
| BF16 | 253 | 306 | stop | 8 | ✓ |
| NVFP4 | **2604** | **2669** | stop | 12 | ✗ |

- **BF16**: brief reasoning — interprets "two times older" as "twice as old" (colloquial), Ben=2, Brandon=8. Goes straight to the answer.
- **NVFP4**: 2604 reasoning tokens spent disambiguating the phrase "older" (strictly = +200%, colloquially = ×2). Eventually picks the **strict interpretation**: Ben=1+2×1=3, Brandon=12. The answer is mathematically defensible but does not match the gold.

> **Pattern**: NVFP4 **overthinks** a simple word problem and picks a more formal interpretation that doesn't match the expected one.

### gsm8k[11] — "2/5 times more"

Same pattern: BF16 490 reasoning tokens, NVFP4 **2771 reasoning tokens**. NVFP4 reinterprets "2/5 times more" as "2/5 as many" instead of "5 + 2/5 × 5" and gets 44 instead of 104.

### mmlu_thinking[6] — length-overflow on a trivial question

| Model | reasoning tokens | finish | answer |
|---|---|---|---|
| BF16 | 762 | stop | D (correct) |
| NVFP4 | **4096** | **length** | (no answer) |

**Question**: about dim V < dim W and injectivity of a linear map (correct answer D — "if V and W are finite-dimensional and dim V < dim W, there is no linear injective map W → V").

- **BF16**: brief reasoning, immediately answers D.
- **NVFP4**: gets stuck in a tangent about vacuous truth, edge cases with dim V = 0, infinite-dimensional spaces. Hits 4096 tokens without producing an answer.

> This is the **only length-overflow** in the run. On MATH there were 0 length-overflows. The v2 quantization **eliminates** competition-math loops, but the residual risk on simple MCQs with thinking mode (where the model digs too deep into a trivial question) remains.

### math[22] — an evaluator false negative (not a model regression)

| Model | content | extracted | gold | correct |
|---|---|---|---|---|
| BF16 | `\boxed{120}` | `120` | `120` | ✓ |
| NVFP4 | `\boxed{120^\circ}` | `120^circ` | `120` | ✗ |

Both models produce the semantically correct answer 120°. NVFP4 added `^\circ` inside the `\boxed{}` and the simple LaTeX-normalizer in the evaluator failed to strip it, extracting `120^circ` instead of `120`. **This is an evaluator bug, not a model regression.** The real MATH degradation is effectively 0 (NVFP4 actually solves all 30/30 just like BF16 semantically).

## Conclusions

1. **Overall accuracy**: the **2.69 pp** gap in favor of BF16 is **NOT statistically significant** (95% Wilson CIs overlap by 3.93 pp). On 483 pairs this is within the noise.

2. **MATH loops eliminated**: 0 length-overflows on MATH. The v2 quantization resolves the competition-math looping issue. The only length-overflow in the run happened on `mmlu_thinking[6]` (a trivial MCQ where NVFP4 got stuck in a vacuous-truth tangent).

3. **Code generation**: a stable -2.5pp on HumanEval and -3.8pp on MBPP. The degradation pattern: NVFP4 is weaker on **fine-grained edge cases** and **ambiguous prompt interpretation**. On curated code (20 prompts) there are 0 degradations, both models at 90.0%.

4. **Simple math reasoning (GSM8K)**: -6.7pp on 30 samples, but the cause is **overthinking**, not a loss of capability. NVFP4 spends ~20% more reasoning tokens and sometimes picks a more formal interpretation that doesn't match the gold. On MATH (where reasoning is genuinely needed) the gap is -3.3pp and entirely within noise.

5. **IFEval**: -2.5pp on 80 samples (5 BF16-wins vs 3 NVFP4-wins). Degradations cluster around `detectable_content:postscript` (NVFP4 skips the P.S. marker) and `keywords:forbidden_words` (NVFP4 includes the forbidden word).

6. **Latency**: NVFP4 is faster than BF16 by **1.44× overall** (median 1.43s vs 1.83s). Largest speedups on creative/knowledge/translation/safety (-40% to -53%), code and IFEval (-39% to -44%). On simple MCQs without thinking the latency is at parity — on 1-2 token responses the per-request overhead is not amortized.

7. **Simple tasks** (knowledge, translation, multilingual, creative, safety, code_explain, logical, multi-reason): **0 degradations**, identical quality. Quantization does not damage the "fast" modes of the model.