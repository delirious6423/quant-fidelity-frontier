---
license: other
license_name: webai-non-commercial-license-ver.-1.0
license_link: https://huggingface.co/webAI-Official/Air-26B-A4B-gguf/blob/main/LICENSE.md
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
language:
 - en
tags:
 - gemma4
 - mixture-of-experts
 - gguf
 - llama.cpp
 - quantized
---

# Air-26B-A4B

A **13.47 GB, 4-bit GGUF** of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
for llama.cpp, built so the **128K-token context fits in 16 GB** of memory. Nothing was trained: the weights are
Google's, quantized with webAI's proprietary Air method. Models on the webAI app will let you create your own Air
version of a model.

**On our held-out evaluation text, Its predictions stayed 2.6× closer to the original BF16 model than Google's own 4-bit release in our internal evaluations**, at a quarter of BF16's size and about twice its speed on everyday prompts. 

How each number was measured, and on what hardware, is under [Testing conditions](#testing-conditions-and-reproducibility).

### At a glance

| | BF16 (original) | **Air-26B-A4B** |
|---|---:|---:|
| **Distance from BF16** (KL divergence, lower is closer; Google's 4-bit: 0.176) | 0 | **0.068** |
| File size | 50.5 GB | **13.5 GB** |
| Memory at a 128K context | — | **15.97 GB** |
| Average score, 6 benchmarks | 79.3 | **78.3** |
| wIL Track A gate | 0.508 | **0.502** |
| wIL Track B, all 6,156 items | 0.812 | **0.812** |
| Generation speed, one request | 28.7 tok/s | **60.6 tok/s** |
| Time to first token, 4K-token prompt | 2.45 s | **1.28 s** |

<sub>Scores: one sampled run each, llama.cpp on one H100 (BF16 with a 16-bit KV cache, Air with an 8-bit one).
KL divergence: one held-out text in 2,048-token windows. Speed: llama-bench on one DGX Spark, one request at a time.
Memory: llama.cpp's buffers on a 24 GB MacBook Pro. Results on other hardware or settings may differ; see
[Testing conditions](#testing-conditions-and-reproducibility).</sub>

### Quick start

```bash
llama-server -hf webAI-Official/Air-26B-A4B-gguf --jinja -c 131072 -fa on -ctk q8_0 -ctv q8_0 --temp 1.0 --top-p 0.95 --top-k 64
```

Then open http://localhost:8080. More options under [Usage](#usage).

**Contents:** [Fidelity](#fidelity--how-close-the-predictions-are-to-bf16) · [Highlights](#highlights) ·
[Model details](#model-details) · [Benchmarks](#benchmark-suites--thinking-on) ·
[wIL Track A](#track-a--in-domain-formal-logic-wil) · [wIL Track B](#track-b--held-out-benchmarks-wil) ·
[Size, memory and speed](#size-memory-and-speed) ·
[Testing conditions](#testing-conditions-and-reproducibility) · [Usage](#usage) · [Limitations](#limitations) ·
[Evaluation protocol](#evaluation-protocol) · [License](#license-and-attribution)

![Air-26B-A4B vs BF16 by benchmark](benchmarks.png)

## Fidelity — how close the predictions are to BF16

![KL divergence from BF16](figures/fidelity.png)

| Build | Size (GB) | KL divergence (mean) | 99th percentile | Same top token as BF16 |
|---|---:|---:|---:|---:|
| **Air-26B-A4B** | 13.47 | **0.068** | 1.67 | 95.8% |
| Google QAT Q4_0 (official) | 14.44 | 0.176 | 4.06 | 91.8% |
| Plain Q4_K_M ¶ | 16.80 | 0.176 | 4.40 | 92.3% |
| *ggml-org Q8_0 (8-bit, for reference)* | 26.86 | 0.021 | 0.36 | 98.1% |

KL divergence measures how far a build's next-token probabilities are from BF16's, here on held-out text covering
code, tool calls, knowledge, instruction following, math and general writing; lower is closer. Google's QAT build was
trained to differ slightly from BF16, so benchmarks are the fairer test for it.

¶ Measured in a separate run against a fresh BF16 reference, where Air scored 0.069.

## Highlights

* **2.6× closer to BF16 than Google's official 4-bit release** on our held-out text: KL divergence 0.068
  against 0.176 (a plain Q4_K_M, 3.3 GB larger, also scores 0.176).
* **A quarter of the size:** 13.5 GB against 50.5 GB; the full 128K context fits in 16 GB
  (15.97 GB measured).
* **Within about a point of BF16:** an average of 78.3 against 79.3 over 6 benchmarks;
  knowledge, math and code keep 0.994 of BF16's score.
* **Level with BF16 on wIL:** Track A gate 0.502 against 0.508; Track B
  0.812 against 0.812.
* **About twice as fast on a DGX Spark:** 2.1× faster generation and a 1.9× sooner first token on a 4K-token prompt;
  at the full 256K, still 1.4× faster
  generation and 1.3× faster prompt processing.

It is **not lossless**. Instruction following is the main shortfall (IFBench keeps
0.945 of BF16's score), and three wIL Track B datasets score 1.7 to 3.7 points lower. See
[Limitations](#limitations).

## Model Details

| Property | Value |
|---|---|
| Model ID | `webAI-Official/Air-26B-A4B-gguf` |
| Base model | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| Architecture | Gemma 4 mixture of experts: 25.2B parameters, about 4B active per token |
| Input / output | Text / text |
| Context window | 262,144 tokens (256K), as the base model. Memory measured at 128K; speed measured up to 256K. |
| File | `Air-26B-A4B.gguf`, 13.47 GB, 4.27 bits per weight |
| Method | webAI's proprietary Air method (quantization only; no training) |
| Reasoning | Gemma 4's thinking mode, on by default |
| License | webAI Non-Commercial License ver. 1.0 |

## Results

All comparisons are against **BF16**, Google's original weights, on the same items with the same settings.
Differences come with 95% paired intervals; "no clear change" means the interval includes zero.

### Benchmark suites — thinking on

| Benchmark | Track | BF16 | Air-26B-A4B | Retention (95% CI) |
|---|---|---:|---:|---|
| LiveCodeBench v6 | Code | 75.8 | 74.6 | 0.985 (0.956–1.016) |
| AIME 2026 | Math | 85.8 | 86.0 | 1.002 (0.985–1.023) |
| GPQA Diamond | Knowledge | 81.1 | 80.7 | 0.995 (0.966–1.026) |
| MMLU-Pro (2,000 of 12,032 questions) | Knowledge | 82.8 | 82.1 | 0.991 (0.977–1.005) |
| IFEval | Instruction following | 91.0 | 90.4 | 0.993 (0.976–1.010) |
| IFBench | Instruction following | 59.3 | 56.1 | 0.945 (0.906–0.986) |

* **MMLU-Pro** is partial: 2,000 of 12,032 questions, sampled evenly across subjects.
* **IFEval** (541 prompts × 2 answers) and **IFBench** (300 × 4) are complete; a fifth of their prompts were also used during development.

![Score kept against BF16](figures/retention.png)

Retention is Air's score divided by BF16's.

### Track A — in-domain formal logic (wIL)

| Lane | BF16 | Air-26B-A4B | Difference (95% CI) | Verdict |
|---|---:|---:|---|---|
| Entailment label, accuracy | 0.575 | 0.625 | +0.050 (−0.020 to +0.120) | no clear change |
| MCQ answer, strict accuracy | 0.020 | 0.010 | −0.010 (−0.035 to +0.015) | no clear change |
| MCQ answer, loose match | 0.740 | 0.685 | −0.055 (−0.120 to +0.005) | no clear change |
| Procedural, strict accuracy | 0.010 | 0.045 | +0.035 (+0.010 to +0.065) | better |
| Procedural, loose match | 0.235 | 0.220 | −0.015 (−0.040 to +0.005) | no clear change |
| Semantic parse, token F1 | 0.482 | 0.479 | −0.003 (−0.013 to +0.007) | no clear change |
| Rule induction, derivation score | 0.624 | 0.736 | +0.112 (+0.046 to +0.180) | better |
| Lean formalization, token F1 | 0.399 | 0.420 | +0.021 (−0.004 to +0.045) | no clear change |
| Lean formalize and prove, token F1 | 0.310 | 0.313 | +0.003 (−0.019 to +0.025) | no clear change |
| FOL translation, exact match | 0.005 | 0.015 | +0.010 (+0.000 to +0.025) | no clear change |
| LM corpus perplexity ↓ | 14.272 | 14.066 | — | — |
| Math corpus perplexity ↓ | 60.173 | 61.358 | — | — |
| **Gate** | **0.508** | **0.502** | −0.006 (−0.030 to +0.018) | no clear change |
| Mean answer length (tokens) | 3,193 | 2,288 | — | — |

The gate is wIL's headline score in this version of the harness: the mean of the entailment, MCQ and procedural
lanes (with loose-match credit) and semantic-parse token F1. Air is level with BF16 on it, and the two lanes where
the models differ both favour Air. Strict scores are near zero for both because the base model does not use wIL's
exact output formats. These numbers are not comparable with the TwIL-LM3-Pro card, which uses a different gate.

### Track B — held-out benchmarks (wIL)

| Dataset | BF16 | Air-26B-A4B | Difference (95% CI) | Verdict |
|---|---:|---:|---|---|
| GSM8K | 0.973 | 0.970 | −0.003 (−0.010 to +0.000) | no clear change |
| SVAMP | 0.957 | 0.950 | −0.007 (−0.017 to +0.000) | no clear change |
| GSM-Symbolic | 0.990 | 0.997 | +0.007 (−0.003 to +0.020) | no clear change |
| ARC-Challenge (CoT) | 0.983 | 0.973 | −0.010 (−0.023 to +0.000) | no clear change |
| LogicBench | 0.827 | 0.863 | +0.037 (+0.007 to +0.070) | better |
| StrategyQA | 0.780 | 0.810 | +0.030 (+0.000 to +0.060) | no clear change |
| DROP | 0.793 | 0.803 | +0.010 (−0.007 to +0.027) | no clear change |
| CommonsenseQA | 0.863 | 0.837 | −0.027 (−0.053 to −0.003) | worse |
| MuSR | 0.642 | 0.651 | +0.008 (−0.015 to +0.032) | no clear change |
| MMLU-Redux | 0.953 | 0.953 | +0.000 (−0.020 to +0.020) | no clear change |
| IFEval | 0.880 | 0.863 | −0.017 (−0.043 to +0.010) | no clear change |
| RuDaS-OOD | 0.166 | 0.134 | −0.032 (−0.051 to −0.012) | worse |
| BBH logic | 0.995 | 0.998 | +0.003 (−0.001 to +0.007) | no clear change |
| MATH-500 | 0.900 | 0.883 | −0.017 (−0.033 to −0.003) | worse |
| **Macro, 10 CoT datasets** | **0.876** | **0.881** | — | — |
| **Macro, all 14** | **0.836** | **0.835** | — | — |
| Pooled, all items (n = 6,156) | 0.812 | 0.812 | +0.000 (−0.004 to +0.005) | no clear change |

Air is level with BF16 on both macros and the pooled score. LogicBench scores higher and
CommonsenseQA, RuDaS-OOD and MATH-500 lower, by 1.7 to 3.7 points; with fourteen datasets, about one difference this size is
expected by chance. Track B runs with thinking off.

### Size, memory and speed

![Size and memory](figures/size_memory.png)

Memory at a 128K context on a 24 GB MacBook Pro (llama.cpp, flash attention, 8-bit KV cache): weights
12.53 GiB, KV cache 1.48 GiB, compute 0.86 GiB,
**15.97 GB in total**.

**One request at a time**, measured with llama-bench on one DGX Spark with the same settings for both models (time to
first token is prompt length ÷ prompt-processing speed):

![Time to first token and generation speed](figures/speed.png)

| Test | BF16 (tok/s) | Air-26B-A4B (tok/s) | Speedup | Time to first token |
|---|---:|---:|---:|---:|
| Prompt processing, 512 tokens | 1,665.5 | 3,158.7 | ×1.90 | 0.31 s → 0.16 s |
| Prompt processing, 4,096 tokens | 1,674.9 | 3,199.0 | ×1.91 | 2.45 s → 1.28 s |
| Prompt processing, 32,768 tokens | 1,555.3 | 2,713.9 | ×1.74 | 21.1 s → 12.1 s |
| Prompt processing, 65,536 tokens | 1,424.2 | 2,341.2 | ×1.64 | 46.0 s → 28.0 s |
| Prompt processing, 131,072 tokens | 1,215.8 | 1,791.2 | ×1.47 | 107.8 s → 73.2 s |
| Prompt processing, 262,144 tokens | 934.1 | 1,236.2 | ×1.32 | 4.7 min → 3.5 min |
| Generation, empty context | 28.7 | 60.6 | ×2.11 | — |
| Generation, after 32,768 tokens | 25.3 | 48.9 | ×1.93 | — |
| Generation, after 65,536 tokens | 23.5 | 42.2 | ×1.80 | — |
| Generation, after 131,072 tokens | 20.7 | 34.0 | ×1.65 | — |
| Generation, after 262,016 tokens | 16.6 | 23.7 | ×1.42 | — |

![Speed against context length](figures/speed_context.png)

**Under load on one H100** (output tokens per second, whole GPU) †:

| Suites | BF16 | Air-26B-A4B | Ratio |
|---|---:|---:|---:|
| AIME / GPQA / LiveCodeBench | 410 | 625 | ×1.52 |
| IFEval / IFBench | 538 | 897 | ×1.67 |

† During the benchmark runs. Air's smaller file fit two servers on the GPU against BF16's one, so this is output per
GPU, not speed per request.

## Testing Conditions

The results above were measured under the following conditions:

* **Quality scores** (benchmarks, wIL) come from one sampled run per model, without repeats, so there is no run-to-run noise estimate.
* **Air's benchmark scores include its 8-bit KV cache**; BF16 used a 16-bit one. Running Air with a 16-bit cache was not benchmarked.
* **Speed is hardware-specific.** The speedups were measured on one DGX Spark (NVIDIA GB10). Time to first token excludes server and network overhead.
* **Memory** (15.97 GB at 128K) is llama.cpp's allocated buffers on one MacBook Pro configuration (Metal backend, one slot, 8-bit KV cache); it is not peak process memory and excludes the operating system and other apps. A 16-bit KV cache, more parallel slots or another backend need more memory.
* **KL divergence** was measured on one held-out text with short (2,048-token) windows; other text or longer contexts may give a different ratio.
* **wIL** is webAI's internal benchmark using vLLM with Air's weights expanded to BF16 format, not the GGUF in llama.cpp.
* **Under-load throughput** (1.5–1.7× output per GPU) comes from the benchmark runs on an 80 GB H100, where Air fitted two servers and BF16 one.
* **Not measured:** memory at 256K, long-context answer quality, tool calling, and speed on devices with 16–24 GB.

## Usage

```bash
llama-server -hf webAI-Official/Air-26B-A4B-gguf --jinja -c 131072 -np 1 -fa on -ctk q8_0 -ctv q8_0 \
  --temp 1.0 --top-p 0.95 --top-k 64
# or download the file first, then run it locally:
hf download webAI-Official/Air-26B-A4B-gguf Air-26B-A4B.gguf --local-dir .
llama-server -m Air-26B-A4B.gguf --jinja -c 131072 -np 1 -fa on -ctk q8_0 -ctv q8_0 \
  --temp 1.0 --top-p 0.95 --top-k 64
```

* `--jinja` applies Gemma 4's own chat template, which is built into the file.
* Thinking is on by default. To turn it off, pass `--reasoning off` (older llama.cpp builds:
  `--chat-template-kwargs '{"enable_thinking": false}'`).
* The sampling flags are Google's recommended defaults.
* Allow long answers: thinking answers are long, and a short limit costs far more accuracy than the quantization does.

## Limitations

* **One run each**, not pre-registered; no repeat-run noise floor yet.
* **Instruction following** is the main shortfall: IFBench keeps 0.945 of BF16's score.
* **Tool calling was not tested**, so no claim can be made about Air's tool-calling ability.
* **Long-context quality was not measured.** The model runs at 256K, but answer quality at long contexts was not
  tested on this build.
* **Answer caps:** 30% of BF16's IFBench answers and 10% of Air's
  (16% of BF16's) Track A answers reached the answer cap and score zero, so
  those scores partly reflect answer length.
* **Partial benchmarks:** MMLU-Pro uses a 2,000-question sample; LiveCodeBench v6 uses
  131 problems from February to April 2025. Scores are not comparable with public leaderboards.
* **Text only:** the base model's image input is not included.

## Evaluation protocol

- **Benchmark suites:** llama.cpp 448147d on one H100; thinking on; temperature 1.0, top-p 0.95, top-k 64; shared
  per-request seeds; answer caps of 32,768 tokens (AIME, GPQA, LiveCodeBench), 16,384 (MMLU-Pro) and 8,192 (IFEval,
  IFBench). BF16 ran with a 16-bit KV cache on one server, Air with an 8-bit KV cache on two servers sharing the GPU
  (24–32 requests in parallel per server). Scored with webAI's evaluation harness: LiveCodeBench graded in a sandbox,
  IFEval and IFBench with their published verifiers.
- **Speed:** llama-bench (llama.cpp 448147d, CUDA) on one DGX Spark (NVIDIA GB10, 128 GB); all layers on the GPU,
  flash attention, 16-bit KV cache for both, batch 2,048 / micro-batch 512, 128 generated tokens, three runs per test
  (two at 64K and above).
- **Memory:** llama-server (llama.cpp 448147d, Metal) on a MacBook Pro M5 Pro with 24 GB; 131,072-token context, one
  slot, flash attention, 8-bit KV cache; the figure is llama.cpp's allocated buffers.
- **wIL Track A:** vLLM 0.19.1 on one H100; 200 rows per objective; greedy; 16,384-token context.
- **wIL Track B:** vLLM 0.19.1 on a DGX Spark; up to 300 examples per task; greedy; 16,384-token context; thinking off.
- **Fidelity:** llama.cpp KL divergence against the BF16 GGUF on held-out text.
- vLLM cannot load this GGUF, so for the wIL runs its quantized weights were expanded to BF16 format with the
  same values; the scores reflect the quantized model.

## License and attribution

Released under the **webAI Non-Commercial License ver. 1.0** (`LICENSE.md`).

The base model, [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it), is Copyright
Google LLC and licensed under the [Apache License 2.0](https://ai.google.dev/gemma/docs/gemma_4_license), whose text
is kept in `apache-2.0-LICENSE.txt`. This repository holds a modified (quantized) version of it; all credit for the
base model goes to Google DeepMind. Gemma is a trademark of Google LLC.
