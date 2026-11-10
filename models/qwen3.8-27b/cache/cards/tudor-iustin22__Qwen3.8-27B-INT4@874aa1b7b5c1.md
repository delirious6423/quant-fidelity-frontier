---
language:
- en
- zh
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- int4
- vllm
- compressed-tensors
- bluebird-llm
- intel-gaudi
- intel-arc
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
provider: Qwen
name: tudor-iustin22/Qwen3.8-27B-INT4
readme: https://huggingface.co/tudor-iustin22/Qwen3.8-27B-INT4/blob/main/README.md
---

<div align="center">

# Qwen3.8-27B-INT4

**SOTA INT4 weight quantization of Qwen3.8-27B, built for Intel Arc and Intel Gaudi 3 accelerators**

[![X](https://img.shields.io/badge/Follow-@tudoriustin__-000000?logo=x&logoColor=white)](https://x.com/tudoriustin_)
[![Bluebird-LLM](https://img.shields.io/badge/Method-Bluebird--LLM-2563eb)](https://www.bluebirdllm.com)
[![Base model](https://img.shields.io/badge/Base-Qwen3.8--27B-6d28d9)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](https://choosealicense.com/licenses/apache-2.0/)

</div>

> **Follow [@tudoriustin_](https://x.com/tudoriustin_) on X** for more SOTA quantizations and updates from Project Bluebird. Project website: [www.bluebirdllm.com](https://www.bluebirdllm.com)

## About the Model

Qwen3.8-27B-INT4 is a 4-bit weight quantization of Qwen3.8-27B, made for Intel Gaudi 3 and Intel Arc accelerators. It cuts weight storage and accelerator memory requirements by roughly 75% relative to 16-bit precision while staying close to the BF16 baseline on instruction following, knowledge, mathematical reasoning and scientific question answering. It was quantized with [Bluebird-LLM](https://www.bluebirdllm.com) on Intel Gaudi 3. Other GPUs are supported through any compatible vLLM build.

## Model Details

| | |
| --- | --- |
| **Base model** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Architecture** | Qwen3_5 |
| **Modalities** | Text / Image in, Text out |
| **Weight quantization** | INT4 (W4A16) |
| **Activation quantization** | BF16 |
| **KV cache** | FP8 |
| **Format** | compressed-tensors (optimised INT4 format) |
| **Target hardware** | Intel Arc and Intel Gaudi 3 accelerators |
| **Quantization hardware** | Intel Gaudi 3 |
| **Developers** | Tudor Iustin, Project Bluebird |
| **Version** | 1.0 |
| **License** | Apache 2.0 |

## Method

**Scope.** Only the weights of the linear operators within transformer blocks are quantized. Activations are kept in their original precision, and the KV cache is stored in FP8. This reduces the bits per parameter from 16 to 4, cutting disk size and GPU memory requirements by approximately 75%.

**Framework.** [Bluebird-LLM](https://www.bluebirdllm.com) is a heavily optimised training and quantization framework for Intel hardware, based on [LLM Compressor](https://github.com/vllm-project/llm-compressor).

**Procedure.** Quantization was performed with Bluebird-LLM on Intel Gaudi 3. We combine AWQ smoothing with the W4A16 GPTQ scheme, calibrated on samples from open-perfectblend, and export the result in compressed-tensors format.

## Evaluation

### Setup

We evaluate on IFEval, MMLU-Pro, MATH-500, GPQA Diamond, GSM8K Platinum and AIME 2025 using [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness). Models are served with vLLM through its OpenAI-compatible API on Intel Gaudi 3. The BF16 and INT4 models are evaluated under the same pipeline.

### Results

| Benchmark | Qwen3.8-27B (BF16) | Qwen3.8-27B-INT4 |
| --- | :---: | :---: |
| IFEval (0-shot, prompt-level strict) | **92.24%** | 91.93% |
| MMLU-Pro (exact-match) | **84.46%** | 83.45% |
| MATH-500 (pass@1) | **83.73%** | 83.33% |
| GPQA Diamond (pass@1) | **89.23%** | 87.88% |
| GSM8K Platinum (strict-match) | 95.75% | **96.77%** |
| AIME 2025 (pass@1) | **95.42%** | 94.17% |

Bold marks the higher score in each row.

## Quickstart

### vLLM

```bash
vllm serve tudor-iustin22/Qwen3.8-27B-INT4 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml
```

## Acknowledgements

This work builds on [LLM Compressor](https://github.com/vllm-project/llm-compressor), the foundation of Bluebird-LLM, [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team, [vLLM](https://github.com/vllm-project/vllm) for inference, and [lm-evaluation-harness](https://github.com/neuralmagic/lm-evaluation-harness) for evaluation.

## Citation

```bibtex
@misc{iustin2026qwen38int4,
  title  = {Qwen3.8-27B-INT4: INT4 Weight Quantization with Bluebird-LLM on Intel Gaudi 3},
  author = {Tudor Iustin},
  year   = {2026},
  url    = {https://huggingface.co/tudor-iustin22/Qwen3.8-27B-INT4}
}
```

---

<div align="center">

**More SOTA quants coming. Follow [@tudoriustin_](https://x.com/tudoriustin_) on X.**

</div>
