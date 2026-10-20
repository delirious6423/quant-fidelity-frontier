---
base_model:
- google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
license: apache-2.0
language:
- en
- ru
tags:
- nvfp4
- quantization
- blackwell
---

# Gemma-4-26B-A4B NVFP4 — Quality Benchmark vs BF16

## TL;DR

- **Overall accuracy**: BF16 **82.6%** vs NVFP4 **78.1%** (Δ = -4.44 pp)
- **Win/Tie/Loss** (across 270 task pairs): BF16-only wins = **15**, NVFP4-only wins = **3**, both pass = **208**, both fail = **35**, evaluator skipped = **9** (IFEval rules not implemented)
- **Most noticeable degradation**: **GSM8K -23.3 pp** (grade-school math) — NVFP4 frequently enters reasoning loops and burns the whole 8 192 token budget without producing an `Answer: N`.
- **Worst case**: on a trivial MATH problem ("sum 3/10 + 3/1000 to the nearest thousandth") NVFP4 enters an **infinite repetition loop** and never produces `\boxed{0.303}` within 8 192 tokens; BF16 solves the same problem in 1 327 tokens.
- **Reasoning efficiency**: NVFP4 spends on average **+88% more tokens** on GSM8K and **+40% more** on MMLU-thinking than BF16 — quantization makes the chain-of-thought more "diffuse" and prone to repetition.
- **Max-token cap reached**: NVFP4 hit `finish_reason=length` on **33 / 270** requests (vs 15 for BF16). Most of these are on math tasks where the model loops.
- **On standard benchmarks without deep CoT** (ARC, HumanEval, IFEval, knowledge QA, translation, code-explain, multilingual reasoning, safety): the difference is within 0–1 samples out of 30 — not significant.

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

| Category | thinking | n | BF16 | NVFP4 | Δ | BF16-only wins | NVFP4-only wins | both pass | both fail | BF16 hit max | NVFP4 hit max | avg tokens BF16 | avg tokens NVFP4 |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MMLU (no-think) | off | 30 | 83.3% (25/30) | 76.7% (23/30) | **-6.7 pp** | 3 | 1 | 22 | 4 | 1 | 5 | 277 | 302 |
| MMLU (thinking) | on | 30 | 90.0% (27/30) | 83.3% (25/30) | **-6.7 pp** | 3 | 1 | 24 | 2 | 1 | 5 | 1 151 | 1 614 |
| ARC-Challenge (no-think) | off | 30 | 93.3% (28/30) | 93.3% (28/30) | **+0.0 pp** | 0 | 0 | 28 | 2 | 0 | 0 | 166 | 192 |
| ARC-Challenge (thinking) | on | 30 | 100.0% (30/30) | 100.0% (30/30) | **+0.0 pp** | 0 | 0 | 30 | 0 | 0 | 0 | 626 | 687 |
| GSM8K | on | 30 | 83.3% (25/30) | 60.0% (18/30) | **-23.3 pp** | 7 | 0 | 18 | 5 | 5 | 12 | 2 305 | 4 338 |
| MATH | on | 30 | 46.7% (14/30) | 46.7% (14/30) | **+0.0 pp** | 1 | 1 | 13 | 15 | 7 | 10 | 4 465 | 5 086 |
| HumanEval | off | 30 | 96.7% (29/30) | 96.7% (29/30) | **+0.0 pp** | 0 | 0 | 29 | 1 | 0 | 0 | 77 | 101 |
| IFEval (subset) | off | 30 | 56.7% (17/30) | 56.7% (17/30) | **+0.0 pp** | 0 | 0 | 17 | 4 | 0 | 0 | 288 | 291 |
| Curated · logical | on | 5 | 80.0% (4/5) | 80.0% (4/5) | **+0.0 pp** | 0 | 0 | 4 | 1 | 1 | 1 | 1 330 | 1 568 |
| Curated · knowledge | off | 5 | 100.0% (5/5) | 100.0% (5/5) | **+0.0 pp** | 0 | 0 | 5 | 0 | 0 | 0 | 3 | 3 |
| Curated · instr-follow | off | 5 | 100.0% (5/5) | 100.0% (5/5) | **+0.0 pp** | 0 | 0 | 5 | 0 | 0 | 0 | 27 | 29 |
| Curated · creative | off | 3 | 100.0% (3/3) | 100.0% (3/3) | **+0.0 pp** | 0 | 0 | 3 | 0 | 0 | 0 | 92 | 99 |
| Curated · translation | off | 4 | 75.0% (3/4) | 50.0% (2/4) | **-25.0 pp** | 1 | 0 | 2 | 1 | 0 | 0 | 80 | 190 |
| Curated · code | off | 3 | 100.0% (3/3) | 100.0% (3/3) | **+0.0 pp** | 0 | 0 | 3 | 0 | 0 | 0 | 11 | 12 |
| Curated · multi-reason | on | 3 | 100.0% (3/3) | 100.0% (3/3) | **+0.0 pp** | 0 | 0 | 3 | 0 | 0 | 0 | 193 | 194 |
| Curated · safety | off | 2 | 100.0% (2/2) | 100.0% (2/2) | **+0.0 pp** | 0 | 0 | 2 | 0 | 0 | 0 | 2 | 2 |
| **OVERALL** | — | **270** | **82.6% (223/270)** | **78.1% (211/270)** | **-4.44 pp** | **15** | **3** | **208** | **35** | **15** | **33** | **1 069** | **1 437** |

### Token usage (reasoning + completion)

NVFP4 on average spends **more tokens than BF16 on every category with significant CoT**, and roughly the same on short-format tasks:

| Category | avg tokens BF16 | avg tokens NVFP4 | Δ tokens | Δ % |
|---|---:|---:|---:|---:|
| MMLU (no-think) | 277 | 302 | +25 | +9.0% |
| MMLU (thinking) | 1 151 | 1 614 | +463 | +40.2% |
| ARC-Challenge (no-think) | 166 | 192 | +26 | +15.9% |
| ARC-Challenge (thinking) | 626 | 687 | +60 | +9.6% |
| GSM8K | 2 305 | 4 338 | +2 033 | **+88.2%** |
| MATH | 4 465 | 5 086 | +621 | +13.9% |
| HumanEval | 77 | 101 | +24 | +31.7% |
| IFEval (subset) | 288 | 291 | +3 | +1.0% |
| Curated · logical | 1 330 | 1 568 | +239 | +18.0% |
| Curated · knowledge | 3 | 3 | 0 | +0.0% |
| Curated · instr-follow | 27 | 29 | +2 | +6.6% |
| Curated · creative | 92 | 99 | +8 | +8.4% |
| Curated · translation | 80 | 190 | +111 | +139.6% |
| Curated · code | 11 | 12 | +1 | +8.8% |
| Curated · multi-reason | 193 | 194 | 0 | +0.2% |
| Curated · safety | 2 | 2 | 0 | +0.0% |

> Token takeaway: NVFP4 is **less efficient** in reasoning (especially math/logical), with the largest gap on GSM8K (+88%). On MATH and GSM8K this inefficiency frequently turns into **repetition loops** that exhaust the 8 192-token budget (see "Worst-case loops" below).

## Worst-case reasoning loops (NVFP4 only)

**Example 1 — `math#15`** (problem: `What is the sum 3/10 + 3/1000, in decimal form to the nearest thousandth?`)

- **BF16** solves it in 1 327 tokens: outputs `0.3 + 0.003 = 0.303` and finishes with `\boxed{0.303}`.
- **NVFP4** runs for 8 192 tokens (hits `finish_reason=length`), repeating the **same six-line block** dozens of times without ever emitting `\boxed{...}`:
  ```
  * 3/10 = 0.3
  * 3/1000 = 0.003
  * 0.3 + 0.003 = 0.303
  * The decimal form is 0.303.
  * The thousandths place is the third decimal place.
  * 0.303 is already at the thousandths place.
  ```
  ... (this exact block repeats until the token budget is exhausted)

**Example 2 — `gsm8k#1`** (problem: *John picks 4 bananas on Wednesday, 6 on Thursday, triple that on Friday. How many bananas?*)

- **BF16** finishes in 429 tokens with `Answer: 22`.
- **NVFP4** runs for 8 192 tokens, looping on:
  ```
  * Wednesday: 4
  * Thursday: 6
  * Friday: 3 * 4 = 12
  * Total: 4 + 6 + 12 = 22.
  ```
  ... (repeats until token budget exhausted; never writes `Answer: N`)

This looping behaviour is responsible for most of the GSM8K accuracy gap: 12 / 30 NVFP4 GSM8K requests hit the 8 192-token cap, vs only 5 / 30 for BF16 — and in 7 of those cases BF16 completed successfully in 500–2 800 tokens.

# sglang workaround for vision_tower

**TL;DR:** vLLM nightly runs this model out of the box. For sglang, a small monkey-patch is needed until official support lands.

## Problem

sglang crashes on the first multimodal request:

RuntimeError: shape '1, 3456' is invalid for input of size 8709120
  at compressed_tensors_w4a4_nvfp4.py:169 → out.view(*output_shape)

## Root cause

In `config.json` → `quantization_config.ignore`, the vision_tower layers are listed under their **unfused** names (`q_proj.linear`, `k_proj.linear`, `v_proj.linear`, `gate_proj.linear`, `up_proj.linear`, `down_proj.linear`). However, sglang's `gemma4_vision` implementation fuses them into `qkv_proj` / `gate_up_proj`.

As a result, the ignore-filter in `CompressedTensorsConfig.get_quant_method` **does not match** the fused layer names — an empty NVFP4 scheme gets created for them (no `weight_packed` / `weight_scale` in the checkpoint, since vision_tower is stored as bf16), and `out.view(*output_shape)` with `[1, w_n]` crashes on a 3D input `[1, seq, hidden]`.

## Patch

Two monkey-patches (no sglang source edits):

1. **`CompressedTensorsConfig.get_quant_method`** — return `UnquantizedLinearMethod()` for any layer whose `prefix` contains `vision_tower.`.
2. **`CompressedTensorsW4A4Fp4.apply_weights`** — correct reshape for arbitrary input dimensions: `out_shape = list(x.shape[:-1]) + [w_n]` instead of the hardcoded `[x.shape[0], w_n]`.

## Usage

Drop `gemma4.py` next to your model and launch sglang through it (the patch is applied in-process before the server starts):

```python
#!/usr/bin/env python
import sys
import sglang
import sglang.srt.layers.quantization.compressed_tensors.compressed_tensors as ct_mod
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a4_nvfp4 import (
    CompressedTensorsW4A4Fp4 as _Nvfp4,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    enable_flashinfer_fp4_gemm, fp4_gemm, fp4_quantize,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

# Patch 1: force UnquantizedLinearMethod for vision_tower layers
_orig_get = ct_mod.CompressedTensorsConfig.get_quant_method
def _patched_get(self, layer, prefix):
    if "vision_tower." in prefix:
        return UnquantizedLinearMethod()
    return _orig_get(self, layer, prefix)
ct_mod.CompressedTensorsConfig.get_quant_method = _patched_get

# Patch 2: handle 3D inputs in apply_weights
def _safe_apply(self, layer, x, bias=None):
    out_dtype = x.dtype
    w_n, _ = layer.weight_packed.shape
    out_shape = list(x.shape[:-1]) + [w_n]
    x_fp4, x_bs = fp4_quantize(x, layer.input_global_scale)
    w = layer.weight_packed.T if enable_flashinfer_fp4_gemm else layer.weight_packed
    w_bs = layer.weight_scale.T if enable_flashinfer_fp4_gemm else layer.weight_scale
    out = fp4_gemm(x_fp4, w, x_bs, w_bs, layer.alpha, out_dtype, w_n)
    if bias is not None:
        out = out + bias
    return out.view(*out_shape)
_Nvfp4.apply_weights = _safe_apply

# Launch sglang serve
from sglang.launch_server import run_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.plugins import load_plugins

if __name__ == "__main__":
    load_plugins()
    server_args = prepare_server_args(sys.argv[1:])
    run_server(server_args)
```

Launch:
```python
python gemma4.py \
  --model-path /models/gemma-4-26B-A4B-it-NVFP4 \
  --served-model-name gemma-4-26B-A4B-it \
  --host 0.0.0.0 --port 8000 \
  --trust-remote-code
```