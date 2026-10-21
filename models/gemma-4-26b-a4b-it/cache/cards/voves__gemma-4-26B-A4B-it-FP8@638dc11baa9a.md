---
base_model:
- google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
license: apache-2.0
language:
- en
- ru
tags:
- fp8
- quantization
---

# Gemma-4-26B-A4B FP8 — Quality Benchmark vs BF16

## TL;DR

- **Overall accuracy**: BF16 **81.5%** vs FP8 **81.5%** (Δ = **0.00 pp**) — parity at n=270
- **Win/Tie/Loss** (across 270 task pairs): BF16-only wins = **7**, FP8-only wins = **7**, both pass = **213**, both fail = **34**, evaluator skipped = **9** (IFEval rules not implemented) — **the win/loss ratio is perfectly symmetric**
- **Largest degradation**: **MATH -6.7 pp** and **GSM8K -6.7 pp** — both driven by reasoning-loop timeouts, not arithmetic errors
- **Wins for FP8**: **MMLU-thinking +6.7 pp** and **MMLU no-think +3.3 pp** — FP8 actually scores higher on knowledge MCQ (within sampling noise at n=30, but consistently across both modes)
- **Worst case**: on the trivial MATH problem "sum `3/10 + 3/1000` to the nearest thousandth" FP8 enters an **infinite repetition loop** and never produces `\boxed{0.303}` within 8 192 tokens; BF16 solves it in 1 416 tokens.
- **Reasoning efficiency**: FP8 spends on average **+8% more tokens** than BF16 (1 195 vs 1 105). On logical-reasoning curated prompts FP8 is actually **-46%** shorter than BF16.
- **Max-token cap reached**: FP8 hit `finish_reason=length` on **20 / 270** requests (vs **19** for BF16 — essentially identical).

## Methodology

### Generation parameters

- **temperature = 0** (greedy / deterministic mode)
- **seed = 42** (fixed for reproducibility)
- **top_p = 1.0**
- **`enable_thinking`** is toggled via `chat_template_kwargs` (supported by sglang for Gemma-3/4 chat templates).

### Categories and thinking mode

Each category uses a thinking mode chosen based on the task type:

| Category | thinking | max_tokens | Description |
|---|---|---|---|
| `mmlu` | off | 512 | Knowledge MCQ · 4-choice · EN |
| `mmlu_thinking` | on | 4 096 | Knowledge MCQ · 4-choice · EN |
| `arc` | off | 512 | Science MCQ · EN |
| `arc_thinking` | on | 4 096 | Science MCQ · EN |
| `gsm8k` | on | 8 192 | Grade-school math · EN |
| `math` | on | 8 192 | Competition math · EN |
| `humaneval` | off | 4 096 | Python code gen · EN |
| `ifeval` | off | 4 096 | Instruction following · EN |
| `curated_logical` | on | 4 096 | Syllogism / puzzle · RU+EN |
| `curated_knowledge` | off | 1 024 | Factual recall · RU+EN |
| `curated_instr_follow` | off | 4 096 | Format constraints · RU+EN |
| `curated_creative` | off | 4 096 | Story / poem · RU+EN |
| `curated_translation` | off | 1 024 | RU↔EN translation |
| `curated_code` | off | 4 096 | Code bugfix / explain · RU+EN |
| `curated_multi_reason` | on | 4 096 | Math / logic in RU |
| `curated_safety` | off | 4 096 | Helpful on benign edge · RU+EN |

> For MCQ benchmarks (MMLU, ARC) **both** variants were run: `*_thinking` (thinking=on) and the base (thinking=off), to cover both the "fast" and "reasoning" modes.

### Datasets

Standard datasets were fetched via the HF `datasets` API, with 30 samples per dataset and a fixed sampling seed (`numpy.default_rng(42)`).

| Dataset | Config / split | Size | Used for |
|---|---|---|---|
| `cais/mmlu` | `all` / `test` | 30 | Knowledge MCQ (57 subjects) |
| `allenai/ai2_arc` | `ARC-Challenge` / `test` | 30 | Science MCQ |
| `openai/gsm8k` | `main` / `test` | 30 | Grade-school math |
| `EleutherAI/hendrycks_math` | 7 subjects / `test` (mixed) | 30 | Competition math |
| `openai/openai_humaneval` | `test` | 30 | Python code generation |
| `google/IFEval` | `train` | 30 | Instruction following |

**Curated collection** — 30 hand-crafted prompts across 8 sub-categories: logical reasoning (5), knowledge QA (5), instruction following (5), creative writing (3), translation (4), code explain/fix (3), multilingual reasoning (3), safety/edge (2).

### Evaluators

| Category | Approach |
|---|---|
| MMLU / ARC | Parse option letter (A-E) via regex ("answer is X", line-prefix "A.", etc.), compare to gold index. |
| GSM8K | Extract number from `Answer: N` or the last number in the answer. |
| MATH | Extract `\boxed{...}`, normalize LaTeX + numeric comparison with gold solution. |
| HumanEval | Extract Python code from markdown fences, sandboxed exec with official tests (timeout 20s). |
| IFEval | Subset of rule verifiers implemented (`length_constraints:*`, `keywords:*`, `detectable_format:number_bullet_lists`, `punctuation:no_comma`, `startend:end_checker`, `change_case:*`, `language:response_language`). Not implemented: `combination:*`, `detectable_format:*` (other than bullets), `keywords:letter_frequency`, `length_constraints:nth_paragraph_first_word`. Tasks with only unimplemented rules are excluded from the denominator (skipped). |
| Curated | Programmatic checks: valid JSON, bullet/word count, absence of forbidden words, line count, paragraph length, exact-text match, contains-phrase match, translation language check, etc. |

## Results by category

| Category | thinking | n | BF16 | FP8 | Δ | BF16-only wins | FP8-only wins | both pass | both fail | BF16 hit max | FP8 hit max | avg tokens BF16 | avg tokens FP8 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU (no-think) | off | 30 | 83.3% (25/30) | 86.7% (26/30) | **+3.3 pp** | 1 | 2 | 24 | 3 | 2 | 2 | 282 | 280 |
| MMLU (thinking) | on | 30 | 83.3% (25/30) | 90.0% (27/30) | **+6.7 pp** | 0 | 2 | 25 | 3 | 1 | 1 | 1 225 | 1 299 |
| ARC-Challenge (no-think) | off | 30 | 93.3% (28/30) | 93.3% (28/30) | **+0.0 pp** | 0 | 0 | 28 | 2 | 0 | 0 | 161 | 181 |
| ARC-Challenge (thinking) | on | 30 | 100.0% (30/30) | 100.0% (30/30) | **+0.0 pp** | 0 | 0 | 30 | 0 | 0 | 0 | 638 | 615 |
| GSM8K | on | 30 | 83.3% (25/30) | 76.7% (23/30) | **-6.7 pp** | 2 | 0 | 23 | 5 | 5 | 7 | 2 417 | 3 065 |
| MATH | on | 30 | 43.3% (13/30) | 36.7% (11/30) | **-6.7 pp** | 4 | 2 | 9 | 15 | 10 | 10 | 4 574 | 4 771 |
| HumanEval | off | 30 | 96.7% (29/30) | 96.7% (29/30) | **+0.0 pp** | 0 | 0 | 29 | 1 | 0 | 0 | 77 | 78 |
| IFEval (subset) | off | 30 | 56.7% (17/30) | 56.7% (17/30) | **+0.0 pp** | 0 | 0 | 17 | 4 | 0 | 0 | 293 | 291 |
| Curated · logical | on | 5 | 80.0% (4/5) | 100.0% (5/5) | **+20.0 pp** | 0 | 1 | 4 | 0 | 1 | 0 | 1 407 | 760 |
| Curated · knowledge | off | 5 | 100.0% (5/5) | 100.0% (5/5) | **+0.0 pp** | 0 | 0 | 5 | 0 | 0 | 0 | 3 | 3 |
| Curated · instr-follow | off | 5 | 100.0% (5/5) | 100.0% (5/5) | **+0.0 pp** | 0 | 0 | 5 | 0 | 0 | 0 | 27 | 25 |
| Curated · creative | off | 3 | 100.0% (3/3) | 100.0% (3/3) | **+0.0 pp** | 0 | 0 | 3 | 0 | 0 | 0 | 92 | 91 |
| Curated · translation | off | 4 | 75.0% (3/4) | 75.0% (3/4) | **+0.0 pp** | 0 | 0 | 3 | 1 | 0 | 0 | 80 | 113 |
| Curated · code | off | 3 | 100.0% (3/3) | 100.0% (3/3) | **+0.0 pp** | 0 | 0 | 3 | 0 | 0 | 0 | 11 | 11 |
| Curated · multi-reason | on | 3 | 100.0% (3/3) | 100.0% (3/3) | **+0.0 pp** | 0 | 0 | 3 | 0 | 0 | 0 | 189 | 187 |
| Curated · safety | off | 2 | 100.0% (2/2) | 100.0% (2/2) | **+0.0 pp** | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 2 |
| **OVERALL** | — | **270** | **81.5% (220/270)** | **81.5% (220/270)** | **0.00 pp** | **7** | **7** | **213** | **34** | **19** | **20** | **1 105** | **1 195** |

### Token usage (reasoning + completion)

FP8 spends slightly more tokens than BF16 on average. On curated logical reasoning FP8 is actually **46% more token-efficient** than BF16:

| Category | avg tokens BF16 | avg tokens FP8 | Δ tokens | Δ % |
|---|---:|---:|---:|---:|
| MMLU (no-think) | 282 | 280 | -2 | -0.8% |
| MMLU (thinking) | 1 225 | 1 299 | +74 | +6.0% |
| ARC-Challenge (no-think) | 161 | 181 | +20 | +12.6% |
| ARC-Challenge (thinking) | 638 | 615 | -23 | -3.6% |
| GSM8K | 2 417 | 3 065 | +649 | +26.8% |
| MATH | 4 574 | 4 771 | +197 | +4.3% |
| HumanEval | 77 | 78 | +1 | +1.6% |
| IFEval (subset) | 293 | 291 | -1 | -0.4% |
| Curated · logical | 1 407 | 760 | -647 | **-46.0%** |
| Curated · knowledge | 3 | 3 | 0 | +0.0% |
| Curated · instr-follow | 27 | 25 | -2 | -7.3% |
| Curated · creative | 92 | 91 | -1 | -1.1% |
| Curated · translation | 80 | 113 | +32 | +40.1% |
| Curated · code | 11 | 11 | 0 | +0.0% |
| Curated · multi-reason | 189 | 187 | -1 | -0.7% |
| Curated · safety | 2 | 2 | 0 | +0.0% |

> Token takeaway: FP8 is **roughly as token-efficient as BF16** overall (+8%). The only categories with a noticeable overhead are GSM8K (+27%, mostly from the same loop cases that hit max_tokens) and curated translation (+40%, on 4 samples — too small to be meaningful).

## Worst-case reasoning loops

FP8 produces a small number of reasoning-loop failures on math tasks, in roughly the same number of cases as BF16 itself. The most striking example:

**`math#15`** (problem: `What is the sum 3/10 + 3/1000, in decimal form to the nearest thousandth?`)

- **BF16** solves it in 1 416 tokens, finishes with `\boxed{0.303}`.
- **FP8** runs for 8 192 tokens (hits `finish_reason=length`), repeating the **same four-line block** indefinitely:
  ```
  *  The sum is 0.303.
  *  The decimal form is 0.303.
  *  The thousandths place is the third decimal place.
  *  0.303 is already at the thousandths place.
  ```
  ... (repeats until the token budget is exhausted; never emits `\boxed{...}`)

**`gsm8k#5`** (Valerie's salary problem):

- **BF16** finishes in 854 tokens with the correct answer.
- **FP8** runs for 8 192 tokens and never produces `Answer: N`.

These loops are concentrated on long-reasoning math workloads. FP8 hits the max-token cap on **20/270** requests, vs **19/270** for BF16 — essentially identical.

## Conclusions

1. **Overall accuracy**: **zero measurable gap** between FP8 and BF16 at n=270 (both 81.5%). The win/loss distribution is **perfectly symmetric** (7 BF16-only wins, 7 FP8-only wins).

2. **Standard benchmarks with deep CoT** (MMLU-thinking, GSM8K, MATH): the only measurable gaps are -6.7 pp on GSM8K and -6.7 pp on MATH, both dominated by reasoning-loop timeouts, not arithmetic errors. On MMLU FP8 slightly outperforms BF16 (+3.3 pp no-think, +6.7 pp thinking) — within sampling noise at n=30 but consistent across both modes.

3. **Standard benchmarks without deep CoT** (ARC, HumanEval, IFEval): identical quality (0.00 pp on every category).

4. **Reasoning stability**: FP8 hits the max-token cap on **20/270** requests, vs **19/270** for BF16 — essentially identical. The remaining FP8 timeouts are concentrated on the same hard MATH problems where BF16 also often runs out of budget.

5. **Simple tasks** (knowledge QA, translation, multilingual reasoning, code-explain, safety, creative): identical quality. FP8 quantization **does not damage** the model's "fast" modes.

6. **Instruction following (IFEval)**: no measurable difference at n=30, but the implemented rule subset is limited and does not include the hardest categories (`combination:*`, `detectable_format:*` beyond bullets). A larger sample would be needed to draw a confident conclusion here.

7. **Token efficiency**: FP8 produces **+8% more tokens on average** (1 195 vs 1 105) — close to BF16. On curated logical-reasoning prompts FP8 is actually **46% more efficient** than BF16.