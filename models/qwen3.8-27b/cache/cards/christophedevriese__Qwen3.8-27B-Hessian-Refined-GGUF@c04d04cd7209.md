---
language:
- en
library_name: llama.cpp
pipeline_tag: text-generation
tags:
- gguf
- quantized
- macaq
- hessian
- post-quantization
- qwen
- unsloth
base_model:
- Qwen/Qwen3.8-27B
---

# Qwen3.8-27B — Hessian-refined Unsloth Q4_K_XL GGUF

Author and investigator: Christophe Devriese.

AI assistance: AI tools supported the refinement implementation, experimental investigation, analysis, and preparation of the accompanying technical report. Results are measurements from actual experiments, not independent validation by AI.

This is an experimental post-quantization refinement of Unsloth's Qwen3.8-27B `UD-Q4_K_XL` GGUF. It is not a new BF16-to-GGUF quantization: the starting point is Unsloth's existing quantized codes. BF16 weights and activation Hessians from the original model guide updates to those codes, preserving the original GGUF quantization formats and scales.

## Method

For each linear layer, with teacher weights `W`, decoded quantized weights `Q` and calibration activations `X`, form the Hessian approximation `H = XᵀX` and refine quantized codes to reduce

`L(Q) = Tr[(Q − W) H (Q − W)ᵀ]`.

Supported GGUF code formats are refined using compatible scalar-code flips or vector/product-codebook moves, then encoded back into GGUF. This experiment used 65,536 calibration tokens, 128-token activation chunks, 64 transformer layers, and refined 379 tensors; the output head remained unchanged.

## Model details

| Property | Value |
|---|---|
| Base | Qwen3.8-27B |
| Starting quant | Unsloth UD-Q4_K_XL |
| Format | GGUF |
| Layers | 64 |
| Parameters | approximately 27.3B |
| Context capability of base model | 262,144 tokens (engine/resource limits may reduce this) |
| Calibration | 65,536 tokens |
| Refined tensors | 379 |
| Checkpoint size | 17,659,407,552 bytes (16.45 GiB) |
| Difference from original file | +288 bytes |

## Evaluation

Both students were evaluated against a BF16 GGUF with the same llama.cpp implementation.

| Metric | Unsloth original | Hessian-refinement |
|---|---:|---:|
| Short-context KL(BF16 || GGUF), 960 held-out positions, lower better | 0.00456710 | 0.00396283 |
| WikiText-2 validation perplexity, 132 × 2,048-token chunks, lower better | 6.1009 | 6.1143 |
| HumanEval 32-task subset, pass count | 31/32 | 31/32 |
| LiveCodeBench 24 AtCoder-task subset, pass count | 15/24 | 16/24 |

Interpretation: The Hessian-guided modification lowered short-context BF16-relative KL divergence by 13.23% at effectively constant checkpoint size. It did not improve full-validation perplexity (0.22% higher) and showed no reliable coding-quality improvement (one additional pass in a very small coding subset). These results are preliminary, and should not be represented as a general 13% improvement in model quality.

The coding tests were custom subsets, not official leaderboard scores; the LiveCodeBench subset used up to 16 withheld cases per problem, a fixed 2,048-generation-token cap, and temperature-zero Python generation. Results can be sensitive to selection, decoding, timeout, and output checker rules.

## Compatibility

Text-generation inference was tested with CUDA-enabled llama.cpp and with Open WebUI using the OpenAI-compatible llama.cpp API. This is not evidence of cross-engine equivalence or a supported deployment service.

## Attribution and acknowledgments

- Research author: Christophe Devriese.
- AI support: assistance with the refinement code implementation, investigation, analysis, and technical writing.
- The experiment remains the responsibility of the named author; AI assistance is not independent verification of the findings.

- Original model: [Qwen](https://huggingface.co/Qwen)
- Starting quantization: [Unsloth](https://huggingface.co/unsloth)
- Refinement approach: MACaQ-style Hessian-guided code refinement adapted to GGUF post-quantization updates.

Technical report: Hessian-guided refinement of an Unsloth Qwen3.8-27B GGUF (Christophe Devriese, 12 November 2026).

This model redistributes adapted weights from the original model and its Unsloth GGUF derivative. Review the upstream model licenses and redistribution requirements before further commercial redistribution. This research artifact is provided without any claim that it improves all downstream tasks.
