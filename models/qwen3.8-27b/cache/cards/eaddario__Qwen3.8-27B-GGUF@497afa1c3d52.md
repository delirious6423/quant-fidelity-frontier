---
base_model:
- Qwen/Qwen3.8-27B
datasets:
- eaddario/imatrix-calibration
language:
- en
license:
- apache-2.0
pipeline_tag: image-text-to-text
tags:
- gguf
- quant
- target_bpw
- experimental
---

# Experimental global target bits‑per‑weight quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
Using **non-standard** (forked) [LLaMA C++][llm] release [b10588][llm-rel] for quantization.

Original model: [Qwen/Qwen3.8-27B][mdl]

From the original model creators:

> Following the widespread community adoption of the Qwen3.5 and Qwen3.6 series, we are pleased to introduce Qwen3.8, the most capable generation in the Qwen open-model family to date.
>
> Built on the architectural foundation of Qwen3.5, Qwen3.8 delivers substantial gains across coding, professional work, research, and long-horizon agentic tasks. Qwen3.8-27B brings these advances to a compact, deployment-friendly dense model: a native vision-language model that understands images and videos, with flexible thinking control, designed to carry complex, multi-step tasks through to completion with greater reliability.
>
> ## Qwen3.8 Highlights
>
> Qwen3.8-27B features the following enhancements:
> - **Core Capabilities**: Comprehensive improvements across coding, professional work, research, and long-horizon agentic tasks.
> - **Agent Execution**: Stronger autonomous planning and better handling of environment feedback, leading to more reliable end-to-end task completion.
> - **Downstream Compatibility**: Broader support for popular harnesses and development tools, making it easier to integrate into your existing stack.
> - **Flexible Thinking Control**: Thinking mode is on by default and can be disabled per request; reasoning depth can be tuned with `reasoning_effort`, and reasoning context from historical messages is retained via `preserve_thinking`.
> - **Vision-Language Understanding**: Native support for image and video understanding, from STEM diagrams and documents to hour-scale videos.

---

# ⚠️ PLEASE READ THIS BEFORE USING THESE EXPERIMENTAL VERSIONS! ⚠️
An area of personal interest is finding ways to optimize the inference performance of LLMs when deployed in resource-constrained environments like commodity hardware, desktops, laptops, mobiles, edge devices, etc. There are many approaches to accomplish this, including architecture simplification and knowledge distillation, but my focus has been primarily on quantization and pruning.

The method to produce these experimental versions involves using a custom version of [`llama-imatrix`][imx] to generate an imatrix that includes tensor statistics, and a custom version of [`llama-quantize`][qtz], which computes a per-tensor quantization error, to automatically select the lowest error quantization recipe that achieves a global target bits‑per‑weight (bpw). More details on the implementation and test results [here][bpw]

There are two pull requests ([#14891][imtx-pr] & [#15550][qtz-pr]) to merge these changes back into the core llama.cpp project. This may or may not ever happen so, until then, the modified versions will be available on [GitHub][gh].

For testing and comparison, I use models produced by [Bartowski][btk] (see credits below) and [Unsloth][ust] ([Daniel and Michael Han][ust-ai] do some really interesting stuff!) but when they don't provide versions of the required model, tests and comparisons are against standard quantization obtained by simply running `llama-quantize` with no further optimizations.

All experimental versions were generated using an appropriate imatrix created from datasets available at [eaddario/imatrix-calibration][ical]. In `llama.cpp`, an imatrix is a calibration file derived from running representative text through the model and collecting activation statistics. It is used to weight quantization error so that error in more “important” directions (as estimated from activations) is penalized more heavily.

The process to generate these models is roughly as follows:
1. Convert the original model's [safetensors][sfts] to [GGUF][ggf] F16
2. Estimate the [Perplexity][ppl] score for the F16 model (baseline) using the [wikitext-2-raw-v1][wki-dat] dataset, and save the [logits][lgt]
3. Generate an [imatrix][imx-dat] from the most appropriate [calibration dataset][ical]
4. Quantize the baseline model targeting a bpw average (e.g. `llama-quantize --target-bpw 4.5678 --state-file --imatrix imatrix.gguf baseline-model-F16.gguf 12`)
5. Calculate Perplexity, KL Divergence, ARC (Easy+Challenge), GPQA-Diamond, HellaSwag, MMLU-Redux, Truthful QA and WinoGrande scores for each quantized model
6. Keep version with the best 𝜌PPL and μKLD scores
7. Repeat until all desired quants are created

### Misconceptions about BF16 to F16 Conversion
A common concern when converting BFloat16 ([BF16][bf16]) models to Float16 (F16) is the potential for accuracy loss. Specifically:
- Weight Clipping (Overflow): Clipping, or overflow, is often feared but only occurs if a model's weights exceed the range of ±65,503. This is a relatively rare issue in practice.
- Subnormal Zeroing (Underflow): A more frequent occurrence is underflow, where weights smaller than approximately 5.96x10⁻⁸ are converted to zero.

Crucially, when the F16 model is subsequently used for quantization, the resulting degradation in metrics like Perplexity ([PPL][ppl]) or Kullback–Leibler Divergence ([KLD][kld]) is minimal. Any variations are typically restricted to the hundreds or thousandths decimal places compared to the BF16 model.

However, considering that weight clipping presents a more substantial risk to model integrity, every BF16 base model undergoes validation prior to the conversion process. Consequently, no models hosted in this repository exhibit performance degradation due to overflow clipping.

While BF16 offers precision benefits, performance remains a key factor.
- Conversion Speed: Tests, such as timing `convert_hf_to_gguf.py`, show a notable performance difference, with conversion to BF16 being 15–30% slower than to F16.
- Inference Speed: A less pronounced but still present difference (3–6%) is observed during inference. Although native BF support has been introduced by many chip manufacturers, the slower performance **may** stem from the entire software and hardware stack (firmware, libraries, etc.) not being fully optimized yet.

The choice to prioritize F16 over BF16 is driven by a focus on maximizing performance in specific deployment environments. My primary objective is not large-scale quantization production, a domain where others like [Bartowski][btk] and [Unsloth][ust] excel at, but rather optimizing inference performance for resource-constrained environments. Since BF16 support is not yet widespread in areas like mobile, edge, and embedded devices, using F16 ensures broader compatibility and easier optimization for these use cases.

# Advantages and disadvantages of the global target bits‑per‑weight quantization process
### Advantages
1. **Target arbitrary size models**
    - When specifying `--target-bpw 4.5678` for instance, the algorithm will produce a model (nearly) exactly of that size, which is very useful for maximizing VRAM usage. In a system with 24GB VRAM and a 70B model, standard quants might produce a 16.8GB file (too small, quality left on table) or a 24.1GB file (won't fit). This approach can generate a 23.85GB file to utilize the hardware fully.

2. **Data-driven mixed precision often can improve quality at fixed size**
    - Instead of using hardcoded heuristics (e.g. make `attn_v` Q5_K for a 70B model), that may be sub‑optimal for a given architecture or size, the quantization mix is determined by the actual error sensitivity of the specific model's weights. This, in practice, often yields a better quality/size trade-off, especially in aggressive quantization scenarios (1.5 to 3.5 bpw), or for unusual architectures.

    - **Please note**: `llama.cpp`’s heuristics have been tuned across many models and are highly optimized; although the target bpw method produces better quality often (>75% based on tests with 130 models from 11 different families), it can also lose in surprising cases.

3. **Allows better like-for-like comparisons between models and families**
    - Standard `llama.cpp` quantization uses hardcoded rules like: *"use Q4_K_M, except bump some tensors up/down, except fall back if incompatible, except keep some tensors unquantized..."* and for that reason, two different models quantized with the same Q4_K_M type can end up with very different bpw (e.g. 4.75 and 4.30).

    - All things being equal, the performance of a model is usually proportional to its overall bpw size; models with a higher bpw tend to perform better than lower bpw models. Since model A has simply been given more bits, it will typically perform better (lower perplexity, better eval scores, etc.) even if the underlying quantization method is identical. That makes comparing the performance not a controlled experiment, because the comparison is between models with different effective compression ratios.

    - `--target-bpw` tries to address that by making the experiment more controlled: each model gets quantized to land on (approximately) the same global byte budget, so that the models' performance differences are more attributable to architecture/training differences, quantization error behaviour at the same compression ratio, optimizer’s allocation decisions, etc.

### Disadvantages
1. **Quantization process is significantly slower than standard**
    - This approach can take 5x-10x longer as it quantizes a sample of most tensors into 15 different formats, dequantizes them back to floats, computes error diffs, and selects the best size/error option that fits the global bpw budget.

    - However, the `--state-file` option will save/use the above-mentioned computations so that future quantizations, for the same model, can be generated at normal speed. It also allows to interrupt the computation process and resume it at a later time.

2. **The optimization target is only a proxy for the model's performance quality**
    - The process minimizes a per-tensor estimated error computed from sampled rows, not actual perplexity or divergence of output distributions (a future version may address this). Since errors interact nonlinearly across layers, there are no guarantees it will select the best possible quantization recipe subject to the bpw size constraint.

3. **An imatrix with activations data is required for best results**
    - Activation data is required to compute the bias factor (i.e. the systematic error projected onto activation directions). If the imatrix file does not contain activation data, the `--target-bpw` option will refuse to run.

---

# Models
### Bits per weight, size, perplexity and KL Divergence scores
| Model                                       |     BPW | Size (GB) |                μPPL |   𝜌PPL |               μKLD |    Same Top-P |
| ------------------------------------------- | ------: | --------: | ------------------: | -----: | -----------------: | ------------: |
| [Qwen3.8-27B-F16](./Qwen3.8-27B-F16.gguf)   | 16.0016 |      51.0 |  6.669665 ±0.042605 |   100% |                N/A |           N/A |
| [Qwen3.8-27B-Q1_L](./Qwen3.8-27B-Q1_L.gguf) |  1.7500 |       5.6 | 15.076645 ±0.103470 | 80.83% | 0.931233 ±0.002887 | 60.773 ±0.127 |
| [Qwen3.8-27B-Q2_K](./Qwen3.8-27B-Q2_K.gguf) |  2.5000 |       8.0 |  8.213575 ±0.052813 | 93.16% | 0.288783 ±0.001496 | 77.247 ±0.109 |
| [Qwen3.8-27B-Q3_K](./Qwen3.8-27B-Q3_K.gguf) |  3.5000 |      11.0 |  6.960375 ±0.043746 | 98.13% | 0.082257 ±0.000671 | 87.799 ±0.085 |
| [Qwen3.8-27B-Q4_K](./Qwen3.8-27B-Q4_K.gguf) |  4.5000 |      14.0 |  6.736913 ±0.043056 | 99.48% | 0.022733 ±0.000324 | 93.505 ±0.064 |
| [Qwen3.8-27B-Q5_K](./Qwen3.8-27B-Q5_K.gguf) |  5.5000 |      18.0 |  6.696776 ±0.042814 | 99.75% | 0.009866 ±0.000216 | 95.745 ±0.052 |
| [Qwen3.8-27B-Q6_K](./Qwen3.8-27B-Q6_K.gguf) |  6.5000 |      21.0 |  6.661399 ±0.042549 | 99.89% | 0.004195 ±0.000167 | 97.225 ±0.043 |
| [Qwen3.8-27B-Q7_K](./Qwen3.8-27B-Q7_K.gguf) |  7.5000 |      24.0 |  6.678988 ±0.042743 | 99.94% | 0.002050 ±0.000086 | 97.913 ±0.037 |
| [Qwen3.8-27B-Q8_0](./Qwen3.8-27B-Q8_0.gguf) |  8.5000 |      27.0 |  6.671738 ±0.042653 | 99.96% | 0.001442 ±0.000061 | 98.134 ±0.035 |

### ARC, GPQA-Diamond, HellaSwag, MMLU-Redux, Truthful QA, and WinoGrande scores
Scores generated using [llama-perplexity][ppl] with 750 tasks per test, and a context size of 1024 tokens.

For the test data used in the generation of these scores, follow the appropriate links: [ARC Challenge, Truthful QA][tst-dat], [GPQA-Diamond][gpqa-dat], [HellaSwag][hsw-tst], [MMLU-Redux][mrdx], [WinoGrande][wng-tst]

| Model                                       |   ARC Challenge |    GPQA-Diamond | HellaSwag |      MMLU-Redox |     Truthful QA |      WinoGrande | Avg Score |
| ------------------------------------------- | --------------: | --------------: | --------: | --------------: | --------------: | --------------: | --------: |
| [Qwen3.8-27B-Q1_L](./Qwen3.8-27B-Q1_L.gguf) | 52.5333 ±1.8246 | 23.7374 ±3.0314 |     62.53 | 54.8000 ±1.8185 | 30.0000 ±1.6744 | 63.2000 ±1.7621 |     47.80 |
| [Qwen3.8-27B-Q2_K](./Qwen3.8-27B-Q2_K.gguf) | 66.2667 ±1.7276 | 30.3030 ±3.2743 |     76.00 | 78.1333 ±1.5103 | 32.1333 ±1.7063 | 71.8667 ±1.6430 |     59.12 |
| [Qwen3.8-27B-Q3_K](./Qwen3.8-27B-Q3_K.gguf) | 62.9333 ±1.7648 | 34.3434 ±3.3832 |     81.20 | 82.6667 ±1.3831 | 34.6667 ±1.7389 | 74.8000 ±1.5864 |     61.77 |
| [Qwen3.8-27B-Q4_K](./Qwen3.8-27B-Q4_K.gguf) | 63.4667 ±1.7594 | 32.3232 ±3.3323 |     82.00 | 81.3333 ±1.4237 | 34.4000 ±1.7358 | 75.4667 ±1.5722 |     61.50 |
| [Qwen3.8-27B-Q5_K](./Qwen3.8-27B-Q5_K.gguf) | 64.0000 ±1.7539 | 31.3131 ±3.3042 |     82.40 | 82.0000 ±1.4038 | 32.9333 ±1.7172 | 76.1333 ±1.5576 |     61.46 |
| [Qwen3.8-27B-Q6_K](./Qwen3.8-27B-Q6_K.gguf) | 63.7333 ±1.7567 | 33.8384 ±3.3711 |     82.40 | 81.8667 ±1.4078 | 34.0000 ±1.7309 | 76.0000 ±1.5605 |     61.97 |
| [Qwen3.8-27B-Q7_K](./Qwen3.8-27B-Q7_K.gguf) | 64.2667 ±1.7510 | 32.8283 ±3.3457 |     82.27 | 82.2667 ±1.3956 | 33.7333 ±1.7276 | 76.0000 ±1.5605 |     61.89 |
| [Qwen3.8-27B-Q8_0](./Qwen3.8-27B-Q8_0.gguf) | 64.0000 ±1.7539 | 32.3232 ±3.3323 |     82.40 | 82.2667 ±1.3956 | 33.4667 ±1.7242 | 75.7333 ±1.5664 |     61.70 |

### Tokens per second benchmarks
Scores generated using [llama-bench][bch]. Standard (`llama-quantize` with no optimization) Q4_K_M quantization included for comparison.

| model                                       |      size |  params | backend  | threads |          test |            t/s |
| ------------------------------------------- | --------: | ------: | -------- | ------: | ------------: | -------------: |
| [Qwen3.8-27B-Q1_L](./Qwen3.8-27B-Q1_L.gguf) |  5.57 GiB | 27.32 B | MTL,BLAS |      12 |         pp512 |  244.02 ± 5.88 |
| [Qwen3.8-27B-Q1_L](./Qwen3.8-27B-Q1_L.gguf) |  5.57 GiB | 27.32 B | MTL,BLAS |      12 |         tg128 |   23.23 ± 0.20 |
| [Qwen3.8-27B-Q1_L](./Qwen3.8-27B-Q1_L.gguf) |  5.57 GiB | 27.32 B | MTL,BLAS |      12 | pp1024+tg1024 |   39.63 ± 1.09 |
| [Qwen3.8-27B-Q2_K](./Qwen3.8-27B-Q2_K.gguf) |  7.95 GiB | 27.32 B | MTL,BLAS |      12 |         pp512 | 227.42 ± 23.91 |
| [Qwen3.8-27B-Q2_K](./Qwen3.8-27B-Q2_K.gguf) |  7.95 GiB | 27.32 B | MTL,BLAS |      12 |         tg128 |   22.35 ± 0.69 |
| [Qwen3.8-27B-Q2_K](./Qwen3.8-27B-Q2_K.gguf) |  7.95 GiB | 27.32 B | MTL,BLAS |      12 | pp1024+tg1024 |   38.80 ± 0.13 |
| [Qwen3.8-27B-Q3_K](./Qwen3.8-27B-Q3_K.gguf) | 11.13 GiB | 27.32 B | MTL,BLAS |      12 |         pp512 | 224.23 ± 25.53 |
| [Qwen3.8-27B-Q3_K](./Qwen3.8-27B-Q3_K.gguf) | 11.13 GiB | 27.32 B | MTL,BLAS |      12 |         tg128 |   20.14 ± 0.62 |
| [Qwen3.8-27B-Q3_K](./Qwen3.8-27B-Q3_K.gguf) | 11.13 GiB | 27.32 B | MTL,BLAS |      12 | pp1024+tg1024 |   36.12 ± 0.25 |
| [Qwen3.8-27B-Q4_K](./Qwen3.8-27B-Q4_K.gguf) | 14.31 GiB | 27.32 B | MTL,BLAS |      12 |         pp512 |  240.25 ± 6.54 |
| [Qwen3.8-27B-Q4_K](./Qwen3.8-27B-Q4_K.gguf) | 14.31 GiB | 27.32 B | MTL,BLAS |      12 |         tg128 |   18.96 ± 1.44 |
| [Qwen3.8-27B-Q4_K](./Qwen3.8-27B-Q4_K.gguf) | 14.31 GiB | 27.32 B | MTL,BLAS |      12 | pp1024+tg1024 |   34.74 ± 0.44 |
| [Qwen3.8-27B-Q5_K](./Qwen3.8-27B-Q5_K.gguf) | 17.49 GiB | 27.32 B | MTL,BLAS |      12 |         pp512 |  183.56 ± 3.60 |
| [Qwen3.8-27B-Q5_K](./Qwen3.8-27B-Q5_K.gguf) | 17.49 GiB | 27.32 B | MTL,BLAS |      12 |         tg128 |   17.82 ± 1.41 |
| [Qwen3.8-27B-Q5_K](./Qwen3.8-27B-Q5_K.gguf) | 17.49 GiB | 27.32 B | MTL,BLAS |      12 | pp1024+tg1024 |   30.83 ± 0.38 |
| [Qwen3.8-27B-Q6_K](./Qwen3.8-27B-Q6_K.gguf) | 20.67 GiB | 27.32 B | MTL,BLAS |      12 |         pp512 |  188.80 ± 0.57 |
| [Qwen3.8-27B-Q6_K](./Qwen3.8-27B-Q6_K.gguf) | 20.67 GiB | 27.32 B | MTL,BLAS |      12 |         tg128 |   17.36 ± 0.79 |
| [Qwen3.8-27B-Q6_K](./Qwen3.8-27B-Q6_K.gguf) | 20.67 GiB | 27.32 B | MTL,BLAS |      12 | pp1024+tg1024 |   29.67 ± 0.58 |
| [Qwen3.8-27B-Q7_K](./Qwen3.8-27B-Q7_K.gguf) | 23.85 GiB | 27.32 B | MTL,BLAS |      12 |         pp512 |  196.65 ± 0.02 |
| [Qwen3.8-27B-Q7_K](./Qwen3.8-27B-Q7_K.gguf) | 23.85 GiB | 27.32 B | MTL,BLAS |      12 |         tg128 |   16.49 ± 0.42 |
| [Qwen3.8-27B-Q7_K](./Qwen3.8-27B-Q7_K.gguf) | 23.85 GiB | 27.32 B | MTL,BLAS |      12 | pp1024+tg1024 |   29.49 ± 0.37 |
| [Qwen3.8-27B-Q8_0](./Qwen3.8-27B-Q8_0.gguf) | 27.03 GiB | 27.32 B | MTL,BLAS |      12 |         pp512 |  206.16 ± 4.47 |
| [Qwen3.8-27B-Q8_0](./Qwen3.8-27B-Q8_0.gguf) | 27.03 GiB | 27.32 B | MTL,BLAS |      12 |         tg128 |   15.38 ± 0.14 |
| [Qwen3.8-27B-Q8_0](./Qwen3.8-27B-Q8_0.gguf) | 27.03 GiB | 27.32 B | MTL,BLAS |      12 | pp1024+tg1024 |   28.17 ± 0.27 |

# Metrics used
**[Perplexity][ppx]:** one of the key metrics used in NLP evaluation. It measures the quality of a language model by evaluating how well it predicts the next token given a particular sequence of words. A PPL of **1** indicates an exact match between predicted and actual, whereas values greater than one indicate a degree of "surprise" the generated token differs from the expected.

**[Kullback–Leibler (KL) Divergence][kld]:** a statistical measure of how much a probability distribution differs from another. When quantizing models (or altering the original tensors in any way for that matter), the closest we can preserve the weights' probability distribution to the original model the better, thus the closest to **0** the better.

**[AI2 Reasoning Challenge (ARC)][arc]:** a benchmark to evaluate the ability of AI models to answer complex science questions that require logical reasoning beyond pattern matching.

**[GPQA-Diamond][gpqa]:** a challenging dataset of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry.

**[HellaSwag][hsw]:** the Harder Endings, Longer contexts, and Low-shot Activities for Situations With Adversarial Generations (bit of a mouthful!) is a benchmark designed to test commonsense natural language inference. It requires the model to predict the most likely ending of a sentence.

**[MMLU][mmlu]:** the Massive Multitask Language Understanding evaluates LLMs’ general knowledge and problem-solving abilities across 57 subjects, including elementary mathematics, US history, computer science, and law.

**[Truthful QA][tqa]:** evaluates how well LLMs generate truthful responses to questions. It identifies whether AI models can avoid generating false or misleading information, particularly in areas where human knowledge is prone to misconceptions.

**[Winogrande][wng]:** based on the [Winograd Schema Challenge][wng-chl], is a natural language understanding task requiring models to resolve ambiguities in sentences involving pronoun references.

## Credits
[LLaMa C++][llm] has a large and vibrant community of [contributors][llm-ctt] (~1,600 last time I checked) that actively maintain and extend its functionality, adding new models and architectures almost as fast as they appear. Considering the breakneck speed at which the AI/ML field is advancing, this alone is a remarkable feat!

While I'm grateful to all contributors, I want to recognise three in particular:
* [Colin Kealty][btk] (Bartowski), for the many contributions and for being one of the best sources of high quality quantized models available on Hugging Face
* [Georgi Gerganov][ggg] for his amazing work with **llama.cpp** and the **ggml/gguf** libraries
* [Iwan Kawrakow][ikk] for being one of the key authors behind the many quantization algorithms and the imatrix functionality.

[arc]:      https://llm-stats.com/benchmarks/ai2-reasoning-challenge-(arc)
[base]:     https://huggingface.co/Qwen/Qwen3.8-27B
[b-q4km]:   https://huggingface.co/bartowski
[bch]:      https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench
[bf16]:     https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
[bpw]:      https://github.com/ggml-org/llama.cpp/discussions/18531
[btk]:      https://huggingface.co/bartowski
[ggf]:      https://huggingface.co/docs/hub/en/gguf
[ggg]:      https://github.com/ggerganov
[gh]:       https://github.com/EAddario/llama.cpp/tree/master
[gpqa]:     https://arxiv.org/abs/2311.12022
[gpqa-dat]: https://huggingface.co/datasets/eaddario/benchmark
[hsw-tst]:  https://github.com/klosax/hellaswag_text_data
[hsw]:      https://rowanzellers.com/hellaswag
[ical]:     https://huggingface.co/datasets/eaddario/imatrix-calibration
[ikk]:      https://github.com/ikawrakow
[imtx-pr]:  https://github.com/ggml-org/llama.cpp/pull/14891
[imx-dat]:  https://huggingface.co/eaddario/Qwen3.8-27B-GGUF/tree/main/imatrix
[imx]:      https://github.com/EAddario/llama.cpp/tree/imatrix
[kld]:      https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
[lgt]:      https://huggingface.co/eaddario/Qwen3.8-27B-GGUF/tree/main/logits
[llm-ctt]:  https://github.com/ggml-org/llama.cpp/graphs/contributors
[llm-rel]:  https://github.com/ggml-org/llama.cpp/releases#release-b10588
[llm]:      https://github.com/ggerganov/llama.cpp
[mdl]:      https://huggingface.co/Qwen/Qwen3.8-27B
[mmlu]:     https://en.wikipedia.org/wiki/MMLU
[mrdx]:     https://huggingface.co/datasets/Green-Sky/mmlu-redux-2.0-for-llama.cpp
[ppl]:      https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity
[ppx]:      https://huggingface.co/docs/transformers/en/perplexity
[qtz-pr]:   https://github.com/ggml-org/llama.cpp/pull/15550
[qtz]:      https://github.com/EAddario/llama.cpp/tree/quantize
[sfts]:     https://huggingface.co/docs/safetensors/en/index
[tqa]:      https://github.com/sylinrl/TruthfulQA
[tst-dat]:  https://huggingface.co/datasets/ikawrakow/validation-datasets-for-llama.cpp/tree/main
[u-q4km]:   https://huggingface.co/unsloth
[ust-ai]:   https://unsloth.ai
[ust]:      https://huggingface.co/unsloth
[wki-dat]:  https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1
[wng-chl]:  https://cdn.aaai.org/ocs/4492/4492-21843-1-PB.pdf
[wng-tst]:  https://huggingface.co/datasets/ikawrakow/winogrande-eval-for-llama.cpp/tree/main
[wng]:      https://winogrande.allenai.org
