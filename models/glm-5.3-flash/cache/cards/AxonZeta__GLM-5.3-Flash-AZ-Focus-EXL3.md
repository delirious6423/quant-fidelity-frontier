---
license: mit
library_name: transformers
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- glm
- reasoning
- efficient-reasoning
- exl3
pipeline_tag: text-generation
---

# GLM-5.3-Flash-AZ-Focus (EXL3)

**GLM-5.3-Flash with the second-guessing turned down: 40-50% less thinking at max effort, with reasoning, maths and code
accuracy level with or above the original.**

GLM-5.3-Flash thinks carefully, but a lot of that thinking is hesitation: *"Wait…"*, *"Let me double-check…"*,
re-deriving answers it already had. We optimised the model's weights against that habit, while leaving alone the
checking it does as an agent (running tests, inspecting output before it submits). AZ-Focus
still reasons step by step. It just loops less.

- **About 40% less thinking at max effort** on knowledge and reasoning benchmarks, with accuracy level or slightly up.
- **Maths and science up:** AIME 2025 81.7% vs 75.0%†, GPQA Diamond 82.3% vs 80.8%†.
- **Agentic work close to the original:** SWE-bench Verified 92 vs 95 (first 100 tasks, two runs each); τ²-bench
  within about a point†.
- **Weights only.** The edit lives in the weights; no adapter, runtime patch or extra code.

This repository is a 4-bit EXL3 quantization (routed experts at 4 bpw, everything else BF16), built by AxonZeta
directly from Z.AI's BF16 release. An FP8 version is at
[AxonZeta/GLM-5.3-Flash-AZ-Focus-FP8](https://huggingface.co/AxonZeta/GLM-5.3-Flash-AZ-Focus-FP8).

Made by [AxonZeta](https://axonzeta.ai), an independent AI lab in Manchester, UK.

## Results (max effort)

| Benchmark | GLM-5.3-Flash | AZ-Focus | Thinking tokens |
|---|---|---|---|
| MMLU-Pro (500 questions) | 86.2% · 2,114 tokens | 86.8% · 1,289 tokens | 39% less |
| BIG-Bench Hard (540 tasks) | 83.7% · 634 | 84.1% · 326 | 49% less |
| AIME 2025 (30, 2 runs)† | 75.0% · 16.5k | 81.7% · 11.2k | 32% less |
| GPQA Diamond (198)† | 80.8% | 82.3% | ~20% less |
| HumanEval+ & MBPP+ (EvalPlus)† | 85.8% | 86.9% | ~40% less |
| SWE-bench Verified, first 100 (mini-SWE-agent, two runs) | 95 / 95 | 93 / 91 | |
| τ²-bench airline / retail (two runs)† | 0.89 / 0.919 | 0.88 / 0.905 | |

Thinking on, Z.AI's recommended sampling, same serving setup for both models. Single runs on a few hundred questions
move by about a point either way. None of these benchmarks was used to build the model.

† Measured on an earlier build of the same edit.

## Effort setting

AZ-Focus works best at **max** or **high** effort, where most of the hesitation is. At low effort GLM-5.3-Flash
already thinks very little, and the edit costs accuracy without saving much (BBH 80.0% vs 83.5%, MMLU-Pro 83.0% vs
84.2%). Use the original if you run at low effort.

## Usage

Same architecture, tokenizer and chat template as GLM-5.3-Flash (the template is Z.AI's, revision a6c167b, which keeps
image and video input working in engines that check for it). The routed experts are EXL3 4 bpw with the `mcg`
codebook; load it with an engine that supports GLM-5.3-Flash's EXL3 layout. The difference shows in thinking mode;
with thinking off, output length is unchanged.

## Limitations

- Agentic coding is about 3 points below the original on SWE-bench Verified (two runs of 100 tasks). If long agent
  runs are your main use, compare both on your own tasks.
- Low effort: see above.
- Other languages, very long contexts and multi-hour agent sessions haven't been measured separately yet.

## License

MIT, as for the original model. Based on [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by Z.AI
(Copyright (c) 2026 Z.AI Co., Ltd; see `LICENSE` and `NOTICE`). `azfocus_provenance.json` lists the source of every
file.

Questions, or want to build on this with us? hello@axonzeta.ai
