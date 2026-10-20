---
base_model:
- google/gemma-4-26B-A4B-it
datasets:
- eaddario/imatrix-calibration
language:
- en
license:
- apache-2.0
pipeline_tag: any-to-any
tags:
- gguf
- quant
- experimental
- target_bpw
---

# Experimental global target bits‑per‑weight quantization of [google/gemma-4-26B-A4B-it][base]
Using **non-standard** (forked) [LLaMA C++][llm] release [b8838][llm-rel] for quantization.

Original model: [google/gemma-4-26B-A4B-it][mdl]

From the original model creators:

> <div align="center">
>  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png>
> </div>
>
> <p align="center">
>     <a href="https://huggingface.co/collections/google/gemma-4" target="_blank">Hugging Face</a> |
>     <a href="https://github.com/google-gemma" target="_blank">GitHub</a> |
>     <a href="https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/" target="_blank">Launch Blog</a> |
>     <a href="https://ai.google.dev/gemma/docs/core" target="_blank">Documentation</a>
>     <br>
>     <b>License</b>: <a href="https://ai.google.dev/gemma/docs/gemma_4_license" target="_blank">Apache 2.0</a> | <b>Authors</b>: <a href="https://deepmind.google/models/gemma/" target="_blank">Google DeepMind</a>
> </p>
>
> Gemma is a family of open models built by Google DeepMind. Gemma 4 models are multimodal, handling text and image input (with audio supported on small models) and generating text output. This release includes open-weights models in both pre-trained and instruction-tuned variants. Gemma 4 features a context window of up to 256K tokens and maintains multilingual support in over 140 languages. 
>
> Featuring both Dense and Mixture-of-Experts (MoE) architectures, Gemma 4 is well-suited for tasks like text generation, coding, and reasoning. The models are available in four distinct sizes: **E2B**, **E4B**, **26B A4B**, and **31B**. Their diverse sizes make them deployable in environments ranging from high-end phones to laptops and servers, democratizing access to state-of-the-art AI.
>
> Gemma 4 introduces key **capability and architectural advancements**:
>
> * **Reasoning** – All models in the family are designed as highly capable reasoners, with configurable thinking modes.
>
> * **Extended Multimodalities** – Processes Text, Image with variable aspect ratio and resolution support (all models), Video, and Audio (featured natively on the E2B and E4B models).
>
> * **Diverse & Efficient Architectures** – Offers Dense and Mixture-of-Experts (MoE) variants of different sizes for scalable deployment.
>
> * **Optimized for On-Device** – Smaller models are specifically designed for efficient local execution on laptops and mobile devices.
>
> * **Increased Context Window** – The small models feature a 128K context window, while the medium models support 256K.
>
> * **Enhanced Coding & Agentic Capabilities** – Achieves notable improvements in coding benchmarks alongside native function-calling support, powering highly capable autonomous agents.
>
> * **Native System Prompt Support** – Gemma 4 introduces native support for the `system` role, enabling more structured and controllable conversations.

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
5. Calculate Perplexity, KL Divergence, ARC (Easy+Challenge), HellaSwag, MMLU, Truthful QA and WinoGrande scores for each quantized model
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
| Model                                                     |     BPW | Size (GB) |                     μPPL |   𝜌PPL |               μKLD |    Same Top-P |
| --------------------------------------------------------- | ------: | --------: | -----------------------: | -----: | -----------------: | ------------: |
| [gemma-4-26B-A4B-it-F16](./gemma-4-26B-A4B-it-F16.gguf)   | 16.0073 |      47.0 |  6617.791728 ±107.205431 |   100% |                N/A |           N/A |
| [gemma-4-26B-A4B-it-Q2_K](./gemma-4-26B-A4B-it-Q2_K.gguf) |  2.5145 |       7.4 | 17569.623360 ±285.143912 | 40.59% | 9.423727 ±0.015560 | 10.100 ±0.079 |
| [gemma-4-26B-A4B-it-Q3_K](./gemma-4-26B-A4B-it-Q3_K.gguf) |  3.5000 |      10.0 |  8598.460723 ±162.541607 | 80.01% | 3.945886 ±0.012077 | 41.573 ±0.129 |
| [gemma-4-26B-A4B-it-Q4_K](./gemma-4-26B-A4B-it-Q4_K.gguf) |  4.5000 |      13.0 | 19441.390960 ±385.285232 | 87.72% | 2.067631 ±0.008670 | 55.808 ±0.129 |
| [gemma-4-26B-A4B-it-Q5_K](./gemma-4-26B-A4B-it-Q5_K.gguf) |  5.5000 |      16.0 | 16520.065420 ±329.228992 | 90.37% | 1.337499 ±0.007116 | 65.841 ±0.124 |
| [gemma-4-26B-A4B-it-Q6_K](./gemma-4-26B-A4B-it-Q6_K.gguf) |  6.5000 |      19.0 | 17532.470669 ±349.360366 | 91.90% | 1.015299 ±0.006077 | 70.259 ±0.119 |
| [gemma-4-26B-A4B-it-Q7_K](./gemma-4-26B-A4B-it-Q7_K.gguf) |  7.4997 |      22.0 | 17147.252339 ±340.529496 | 93.49% | 0.636821 ±0.004695 | 77.572 ±0.109 |
| [gemma-4-26B-A4B-it-Q8_0](./gemma-4-26B-A4B-it-Q8_0.gguf) |  8.4996 |      25.0 | 17152.384071 ±340.978253 | 93.69% | 0.580789 ±0.004459 | 78.799 ±0.107 |

### ARC, HellaSwag, MMLU, Truthful QA and WinoGrande scores
Scores generated using [llama-perplexity][ppl] with 750 tasks per test, and a context size of 768 tokens.

For the test data used in the generation of these scores, follow the appropriate links: [HellaSwag][hsw-tst], [ARC, Truthful QA][tst-dat], [MMLU][mrdx] and [WinoGrande][wng-tst]

| Model                                                     |             ARC | HellaSwag |            MMLU |     Truthful QA |      WinoGrande | Avg Score |
| --------------------------------------------------------- | --------------: | --------: | --------------: | --------------: | --------------: | --------: |
| [gemma-4-26B-A4B-it-Q2_K](./gemma-4-26B-A4B-it-Q2_K.gguf) | 35.4667 ±1.7481 |     36.13 | 26.0000 ±1.6027 | 29.4667 ±1.6658 | 48.9333 ±1.8265 |     35.20 |
| [gemma-4-26B-A4B-it-Q3_K](./gemma-4-26B-A4B-it-Q3_K.gguf) | 44.1333 ±1.8143 |     54.40 | 47.6000 ±1.8249 | 32.2667 ±1.7082 | 53.7333 ±1.8219 |     46.43 |
| [gemma-4-26B-A4B-it-Q4_K](./gemma-4-26B-A4B-it-Q4_K.gguf) | 42.0000 ±1.8034 |     51.07 | 49.6000 ±1.8269 | 32.0000 ±1.7045 | 56.0000 ±1.8138 |     46.13 |
| [gemma-4-26B-A4B-it-Q5_K](./gemma-4-26B-A4B-it-Q5_K.gguf) | 42.2667 ±1.8050 |     53.47 | 54.0000 ±1.8211 | 31.7333 ±1.7007 | 56.1333 ±1.8132 |     47.52 |
| [gemma-4-26B-A4B-it-Q6_K](./gemma-4-26B-A4B-it-Q6_K.gguf) | 41.6000 ±1.8010 |     52.13 | 54.4000 ±1.8199 | 32.5333 ±1.7119 | 54.4000 ±1.8199 |     47.01 |
| [gemma-4-26B-A4B-it-Q7_K](./gemma-4-26B-A4B-it-Q7_K.gguf) | 41.3333 ±1.7993 |     52.53 | 53.3333 ±1.8229 | 32.1333 ±1.7063 | 55.8667 ±1.8143 |     47.04 |
| [gemma-4-26B-A4B-it-Q8_0](./gemma-4-26B-A4B-it-Q8_0.gguf) | 42.0000 ±1.8034 |     52.27 | 52.8000 ±1.8241 | 32.5333 ±1.7119 | 56.6667 ±1.8106 |     47.25 |

### Tokens per second benchmarks
Scores generated using [llama-bench][bch]. Standard (`llama-quantize` with no optimization) Q4_K_M quantization included for comparison.

| model                                                     |      size |  params | backend  | threads |          test |             t/s |
| --------------------------------------------------------- | --------: | ------: | -------- | ------: | ------------: | --------------: |
| [gemma-4-26B-A4B-it-Q2_K](./gemma-4-26B-A4B-it-Q2_K.gguf) |  7.39 GiB | 25.23 B | BLAS,MTL |      12 |         pp512 |  1401.19 ±20.33 |
| [gemma-4-26B-A4B-it-Q2_K](./gemma-4-26B-A4B-it-Q2_K.gguf) |  7.39 GiB | 25.23 B | BLAS,MTL |      12 |         tg128 |    96.27  ±1.08 |
| [gemma-4-26B-A4B-it-Q2_K](./gemma-4-26B-A4B-it-Q2_K.gguf) |  7.39 GiB | 25.23 B | BLAS,MTL |      12 | pp1024+tg1024 |   171.06  ±3.07 |
| [gemma-4-26B-A4B-it-Q3_K](./gemma-4-26B-A4B-it-Q3_K.gguf) | 10.28 GiB | 25.23 B | BLAS,MTL |      12 |         pp512 |  1422.88  ±5.82 |
| [gemma-4-26B-A4B-it-Q3_K](./gemma-4-26B-A4B-it-Q3_K.gguf) | 10.28 GiB | 25.23 B | BLAS,MTL |      12 |         tg128 |    99.25  ±0.85 |
| [gemma-4-26B-A4B-it-Q3_K](./gemma-4-26B-A4B-it-Q3_K.gguf) | 10.28 GiB | 25.23 B | BLAS,MTL |      12 | pp1024+tg1024 |   156.23  ±1.91 |
| [gemma-4-26B-A4B-it-Q4_K](./gemma-4-26B-A4B-it-Q4_K.gguf) | 13.22 GiB | 25.23 B | BLAS,MTL |      12 |         pp512 |  1385.18  ±7.25 |
| [gemma-4-26B-A4B-it-Q4_K](./gemma-4-26B-A4B-it-Q4_K.gguf) | 13.22 GiB | 25.23 B | BLAS,MTL |      12 |         tg128 |    94.55  ±1.08 |
| [gemma-4-26B-A4B-it-Q4_K](./gemma-4-26B-A4B-it-Q4_K.gguf) | 13.22 GiB | 25.23 B | BLAS,MTL |      12 | pp1024+tg1024 |   156.71  ±0.33 |
| [gemma-4-26B-A4B-it-Q5_K](./gemma-4-26B-A4B-it-Q5_K.gguf) | 16.16 GiB | 25.23 B | BLAS,MTL |      12 |         pp512 |  1321.55  ±5.68 |
| [gemma-4-26B-A4B-it-Q5_K](./gemma-4-26B-A4B-it-Q5_K.gguf) | 16.16 GiB | 25.23 B | BLAS,MTL |      12 |         tg128 |    81.25  ±0.24 |
| [gemma-4-26B-A4B-it-Q5_K](./gemma-4-26B-A4B-it-Q5_K.gguf) | 16.16 GiB | 25.23 B | BLAS,MTL |      12 | pp1024+tg1024 |   142.95  ±3.71 |
| [gemma-4-26B-A4B-it-Q6_K](./gemma-4-26B-A4B-it-Q6_K.gguf) | 19.09 GiB | 25.23 B | BLAS,MTL |      12 |         pp512 |  1328.48  ±8.00 |
| [gemma-4-26B-A4B-it-Q6_K](./gemma-4-26B-A4B-it-Q6_K.gguf) | 19.09 GiB | 25.23 B | BLAS,MTL |      12 |         tg128 |    82.00  ±0.14 |
| [gemma-4-26B-A4B-it-Q6_K](./gemma-4-26B-A4B-it-Q6_K.gguf) | 19.09 GiB | 25.23 B | BLAS,MTL |      12 | pp1024+tg1024 |   136.02  ±3.90 |
| [gemma-4-26B-A4B-it-Q7_K](./gemma-4-26B-A4B-it-Q7_K.gguf) | 22.03 GiB | 25.23 B | BLAS,MTL |      12 |         pp512 |  1414.49  ±5.13 |
| [gemma-4-26B-A4B-it-Q7_K](./gemma-4-26B-A4B-it-Q7_K.gguf) | 22.03 GiB | 25.23 B | BLAS,MTL |      12 |         tg128 |    75.55  ±0.16 |
| [gemma-4-26B-A4B-it-Q7_K](./gemma-4-26B-A4B-it-Q7_K.gguf) | 22.03 GiB | 25.23 B | BLAS,MTL |      12 | pp1024+tg1024 |   131.36  ±0.31 |
| [gemma-4-26B-A4B-it-Q8_0](./gemma-4-26B-A4B-it-Q8_0.gguf) | 24.97 GiB | 25.23 B | BLAS,MTL |      12 |         pp512 |  1435.65  ±2.43 |
| [gemma-4-26B-A4B-it-Q8_0](./gemma-4-26B-A4B-it-Q8_0.gguf) | 24.97 GiB | 25.23 B | BLAS,MTL |      12 |         tg128 |    73.95  ±0.23 |
| [gemma-4-26B-A4B-it-Q8_0](./gemma-4-26B-A4B-it-Q8_0.gguf) | 24.97 GiB | 25.23 B | BLAS,MTL |      12 | pp1024+tg1024 |   128.04  ±0.23 |

# Metrics used
**[Perplexity][ppx]:** one of the key metrics used in NLP evaluation. It measures the quality of a language model by evaluating how well it predicts the next token given a particular sequence of words. A PPL of **1** indicates an exact match between predicted and actual, whereas values greater than one indicate a degree of "surprise" the generated token differs from the expected.

**[Kullback–Leibler (KL) Divergence][kld]:** a statistical measure of how much a probability distribution differs from another. When quantizing models (or altering the original tensors in any way for that matter), the closest we can preserve the weights' probability distribution to the original model the better, thus the closest to **0** the better.

**[AI2 Reasoning Challenge (ARC)][arc]:** a benchmark to evaluate the ability of AI models to answer complex science questions that require logical reasoning beyond pattern matching.

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
[base]:     https://huggingface.co/google/gemma-4-26B-A4B-it
[b-q4km]:   https://huggingface.co/bartowski
[bch]:      https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench
[bf16]:     https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
[bpw]:      https://github.com/ggml-org/llama.cpp/discussions/18531
[btk]:      https://huggingface.co/bartowski
[ggf]:      https://huggingface.co/docs/hub/en/gguf
[ggg]:      https://github.com/ggerganov
[gh]:       https://github.com/EAddario/llama.cpp/tree/master
[hsw-tst]:  https://github.com/klosax/hellaswag_text_data
[hsw]:      https://rowanzellers.com/hellaswag
[ical]:     https://huggingface.co/datasets/eaddario/imatrix-calibration
[ikk]:      https://github.com/ikawrakow
[imtx-pr]:  https://github.com/ggml-org/llama.cpp/pull/14891
[imx-dat]:  https://huggingface.co/eaddario/gemma-4-26B-A4B-it-GGUF/tree/main/imatrix
[imx]:      https://github.com/EAddario/llama.cpp/tree/imatrix
[kld]:      https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
[lgt]:      https://huggingface.co/eaddario/gemma-4-26B-A4B-it-GGUF/tree/main/logits
[llm-ctt]:  https://github.com/ggml-org/llama.cpp/graphs/contributors
[llm-rel]:  https://github.com/ggml-org/llama.cpp/releases/tag/b8838
[llm]:      https://github.com/ggerganov/llama.cpp
[mdl]:      https://huggingface.co/google/gemma-4-26B-A4B-it
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
