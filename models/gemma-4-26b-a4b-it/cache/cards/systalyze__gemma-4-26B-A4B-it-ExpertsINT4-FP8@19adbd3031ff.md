---
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - gemma
  - gemma4
  - moe
  - gptq
  - quantization
  - compressed-tensors
  - mixed-precision
  - w4a16
  - fp8
  - vllm
---

# gemma-4-26B-A4B-it-ExpertsINT4-FP8

## Model Overview

This is a mixed-precision quantization of
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) optimized for
inference. The 128 MoE expert weights per layer are quantized to INT4 with GPTQ (group size 64);
non-expert transformer linear layers use FP8 dynamic quantization.

- **Model Architecture:** Gemma4ForConditionalGeneration
  - **Input:** Text / Image
  - **Output:** Text
- **Model Optimizations:**
  - **Weight quantization:** INT4 (GPTQ, group size 64) for MoE expert weights; FP8 for all other linear layers
  - **Activation quantization:** FP8 (dynamic, per-token) for non-expert linear layers; none for experts (W4A16)
  - **Unquantized:** router, embeddings, `lm_head`, vision tower
- **Intended Use Cases:** same as the base model — multimodal chat, instruction following, reasoning, tool use
- **Release Date:** 2026-09-03
- **Version:** 1.0
- **Base model developers:** Google DeepMind
- **Quantization and release:** [Systalyze](https://systalyze.com)

The base model is a 30-layer / 128-expert MoE. Expert weights dominate the parameter count, but
each token touches only a few of them, so experts are stored in INT4 while the always-resident
path (attention, dense MLP, router, embeddings, vision tower) stays at FP8 or higher. The
checkpoint is stored in the `compressed-tensors` format and loads directly in vLLM.

| | Precision | On-disk |
|---|---|---|
| Base | BF16 | 49 GB |
| FP8 dynamic | FP8 | 27 GB |
| **ExpertsINT4** | **experts INT4 / rest FP8** | **16 GB** |

**3.1× smaller than BF16**, and small enough to serve from a single 24 GB card's worth
of weights (KV cache and activations are extra).

## Models compared

Every number below was measured here, on the same vLLM stack, against these checkpoints of the
same base model. The BF16 base appears in the accuracy tables only, the AWQ checkpoint in the
throughput tables only:

| Label in tables | Checkpoint | Scheme |
|---|---|---|
| Base BF16 | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) | Unquantized reference |
| **ExpertsINT4** | [`systalyze/gemma-4-26B-A4B-it-ExpertsINT4-FP8`](https://huggingface.co/systalyze/gemma-4-26B-A4B-it-ExpertsINT4-FP8) | MoE expert weights INT4 via GPTQ (W4A16, group size 64); non-expert transformer linear layers FP8 dynamic; router, embeddings, `lm_head`, and vision tower unquantized |
| FP8-dynamic | [`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic) | Transformer-block linear operators FP8 dynamic; vision tower, embeddings, `lm_head`, and MoE router kept in original precision |
| W4A16 (AWQ) | [`cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit) | Published AWQ W4A16 checkpoint; its quantization config excludes selected modules |

## Throughput

Measured with `aiperf` against vLLM on 8×H200, ISL ≈ 6,883 / OSL ≈ 322,
`completion_frac = 1.0` on every cell. The three checkpoints marked above use the same harness, same box,
same request stream.

`out_tok_per_gpu_s` — higher is better:

| Concurrency | FP8-dynamic | **ExpertsINT4** | W4A16 (AWQ) | ExpertsINT4 vs FP8 |
|---:|---:|---:|---:|---:|
| 1 | 186.3 | **209.4** | 193.9 | **+12.4%** |
| 4 | 513.2 | **721.9** | 681.3 | **+40.7%** |
| 8 | 839.3 | **1295.5** | 1232.9 | **+54.4%** |
| 16 | **1132.0** | 1044.7 | 909.4 | −7.7% |
| 32 | **1489.7** | 1269.7 | 1095.9 | −14.8% |

Latency at the same points (p50 TTFT / p50 ITL, ms):

| Concurrency | FP8-dynamic | **ExpertsINT4** | W4A16 (AWQ) |
|---:|---:|---:|---:|
| 1 | 158.2 / 4.92 | 207.0 / **4.16** | 242.6 / 4.45 |
| 4 | 166.8 / 6.15 | **59.5** / **5.16** | 61.4 / 5.43 |
| 8 | 183.8 / 6.54 | **61.6** / **5.43** | 62.6 / 5.72 |
| 16 | **209.8** / 8.35 | 284.8 / **6.75** | 329.7 / 7.10 |
| 32 | **259.0** / 12.18 | 318.5 / **10.79** | 411.5 / 11.10 |

The win is concurrency-dependent. Mixed precision leads in the memory-bound regime
(concurrency 1–8, peaking at **+54%** over FP8 at concurrency 8) and trails FP8 once
the workload becomes compute-bound (concurrency 16–32, down to **−15%**). If you serve
at high concurrency, plain FP8 is the better choice. ExpertsINT4 beats the all-INT4 W4A16 checkpoint at every
concurrency point.

## Accuracy

All three checkpoints were evaluated in the same session, on the same stack and the same
hardware, over **full test sets** (no subsampling). Higher is better.

Setup: `lm_eval` 0.4.13 against vLLM 0.27.1 (OpenAI-compatible API), one H200 per checkpoint,
identical chat template for all three arms, `--apply_chat_template`, greedy decoding
(`temperature=0`), `--seed 1234`, **`enable_thinking=False`**. Thinking-mode results are
reported separately below.

| Category | Benchmark | Base BF16 | FP8-dynamic | ExpertsINT4 |
|---|---|---:|---:|---:|
| Instruction following | IFEval (0-shot, prompt-level strict) | 89.28 | 88.91 | 89.83 |
|  | IFEval (0-shot, inst-level strict) | 92.81 | 92.57 | 93.17 |
| Reasoning and knowledge | MMLU-Pro (5-shot CoT) | 82.06 | 82.09 | 81.30 |
|  | GPQA Diamond (0-shot CoT) | 72.22 | 68.69 | 66.67 |
|  | BIG-Bench Hard (3-shot CoT) | 94.21 | 94.06 | 94.06 |
| Math | GSM8K Platinum (0-shot CoT) | 90.57 | 90.65 | 90.65 |
|  | MATH (4-shot CoT, answer-content scoring) | 94.56 | 94.28 | 93.82 |
|  | AIME 2025 (0-shot, pass@1) | 50.00 | 56.67 | 56.67 |
| Coding | HumanEval (0-shot, pass@1) | 98.17 | 98.17 | 95.73 |
|  | MBPP+ (0-shot, pass@1) | 96.30 | 96.30 | 96.30 |
| Reading comprehension | DROP (0-shot, F1) | 62.19 | 61.66 | 61.33 |
| Multilingual | MGSM, native CoT (11 languages) | 72.22 | 72.25 | 72.11 |
|  | MGSM, English CoT (11 languages) | 72.51 | 72.55 | 73.35 |
|  | MMLU-ProX lite, 7 languages pooled (5-shot) | 76.17 | 76.04 | 75.05 |

Sanity check on the harness: our measured **base** scores land within about 1.4 points of the
figures Red Hat AI publishes for this same base model (IFEval 89.28 here, MMLU-Pro 82.06,
GPQA Diamond 72.22), so the numbers above are comparable to the wider literature even though
every value in the table was measured here.

### MMLU-ProX lite, per language (5-shot)

| Language | Base BF16 | FP8-dynamic | ExpertsINT4 |
|---|---:|---:|---:|
| German | 71.43 | 71.60 | 71.94 |
| Spanish | 79.59 | 79.59 | 77.55 |
| French | 79.59 | 79.25 | 79.42 |
| Japanese | 75.51 | 75.68 | 75.51 |
| Chinese | 78.40 | 78.06 | 76.70 |
| Hindi | 75.34 | 75.68 | 73.13 |
| Arabic | 73.30 | 72.45 | 71.09 |

### Context retrieval robustness (RULER, 0-shot, 4K / 8K)

Measured separately, ExpertsINT4 against FP8-dynamic only, on 1xH100 with vLLM 0.28.0 and
`lm_eval` 0.4.12, `limit=200` per cell.

| Context | Task | ExpertsINT4 | FP8-dynamic | Delta |
|---|---|---:|---:|---:|
| 4096 | ruler_cwe | 0.5085 | 0.4395 | +0.0690 |
| 4096 | ruler_fwe | 0.8867 | 0.9017 | -0.0150 |
| 4096 | ruler_qa_squad | 0.7600 | 0.7750 | -0.0150 |
| 8192 | ruler_cwe | 0.4770 | 0.4410 | +0.0360 |
| 8192 | ruler_fwe | 0.9467 | 0.9200 | +0.0267 |
| 8192 | ruler_qa_squad | 0.7850 | 0.7683 | +0.0167 |

Results move in both directions relative to FP8, with no consistent degradation across the
evaluated 4K and 8K cells. Per-cell standard error is roughly +-0.035. These results evaluate
only 4K and 8K contexts and do not evaluate the model's full supported context window.

### Thinking mode

Gemma 4 can also be served with thinking enabled. Measured that way the three checkpoints agree to
within about a point of each other, and accuracy is clearly higher than with thinking off. One
behaviour drives everything else: on some questions the model never closes its thought channel,
exhausts the token budget and returns no answer, which scores zero. Raising the cap from 16k to 32k
tokens does not remove it and the unquantized BF16 base is affected as much as the quantized
checkpoints, so each cell gives accuracy over the questions that produced an answer, with the share
that produced none beside it. Budgets were 8k-30k output tokens per task.

| Benchmark | Base BF16 | FP8-dynamic | ExpertsINT4 |
|---|---:|---:|---:|
| IFEval (0-shot, inst-level strict) | 96.93 (23.7% none) | 96.49 (28.1% none) | 96.29 (26.1% none) |
| MMLU-Pro (5-shot CoT) | 92.61 (41.9% none) | 93.41 (44.5% none) | 92.11 (39.1% none) |
| GPQA Diamond (0-shot CoT) | 88.66 (51.0% none) | 87.00 (49.5% none) | 86.24 (44.9% none) |
| BIG-Bench Hard (3-shot CoT) | 96.38 (31.3% none) | 96.70 (34.3% none) | 96.22 (30.9% none) |
| GSM8K Platinum (0-shot CoT) | 90.54 (1.2% none) | 90.67 (1.6% none) | 90.13 (1.9% none) |
| AIME 2025 (0-shot, pass@1) | 100.00 (30.0% none) | 100.00 (40.0% none) | 100.00 (46.7% none) |
| HumanEval (0-shot, pass@1) | 100.00 (18.3% none) | 100.00 (27.4% none) | 100.00 (28.0% none) |
| MBPP+ (0-shot, pass@1) | 98.88 (5.6% none) | 99.16 (5.8% none) | 98.88 (5.3% none) |
| MGSM, native CoT (11 languages) | 74.72 (2.3% none) | 75.01 (2.5% none) | 74.87 (2.5% none) |
| MMLU-ProX lite, 7 languages pooled | 82.94 (19.8% none) | 83.48 (21.2% none) | 83.79 (22.7% none) |

### Scope

Except where the thinking-mode section says otherwise, every number on this card was produced with
**thinking disabled**, which is the mode we would recommend for throughput-sensitive serving. The
throughput and RULER measurements are thinking-disabled only.

## Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="systalyze/gemma-4-26B-A4B-it-ExpertsINT4-FP8", max_model_len=12288)
out = llm.generate(
    ["Explain mixture-of-experts routing in two sentences."],
    SamplingParams(temperature=1.0, top_p=0.95, top_k=64, max_tokens=256),
)
print(out[0].outputs[0].text)
```

Sampling defaults follow the base model's `generation_config.json`
(`temperature=1.0`, `top_p=0.95`, `top_k=64`).

## License

[Apache 2.0](https://ai.google.dev/gemma/docs/gemma_4_license), inherited from
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).
