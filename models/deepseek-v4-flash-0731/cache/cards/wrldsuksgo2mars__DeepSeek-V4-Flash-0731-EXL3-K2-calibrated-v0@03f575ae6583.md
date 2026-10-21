---
license: mit
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
tags:
  - deepseek-v4
  - exl3
  - quantized
---

# DeepSeek V4 Flash 0731 EXL3 K2 calibrated v0

This is a self-contained EXL3 K2 (2 bpw, MCG codebook) quantization of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). The routed expert `gate_proj`, `up_proj`, and `down_proj` weights, including the integrated dSpark blocks, are quantized; attention, routing, shared-expert, embedding, output-head, and other coordinator weights retain their source representation.

## Calibration

The quantization used 1,421 prompts totaling approximately 1.08 million tokens. The corpus combined English- and Chinese-focused multilingual Wikipedia material, open-source code review and transformation prompts, and reasoning prompts drawn from mathematics, science, computing, philosophy, economics, law, and politics. Code examples emphasized Python, C++, CUDA, C, and Rust while retaining broader language coverage.

Quantization was performed layer by layer with quantized-prefix replay, so each later block was calibrated against activations produced through the already-quantized prefix. Natural router selections supplied expert activations; sparsely routed experts were supplemented from lower-ranked router candidates up to a 1,024-row target, with an identity-Hessian residual only for any remaining shortfall. Hessians were accumulated as FP32 activation Gram matrices, symmetrized in FP64, and used by the EXL3 trellis quantizer with `sigma_reg = 0.025`.

No standardized quality benchmark is claimed for this release. As with any low-bit quantization, behavior can differ from the source model and should be evaluated for the intended workload.

## Serving on one DGX Spark

The [validated vLLM recipe](https://github.com/tpurtell/deepseek-v4-flash-0731-exl3-k2-spark) builds a pinned native arm64/SM121 image and serves this standard, unsliced checkpoint with its own full K2 dSpark drafter. At `gpu_memory_utilization=0.85`, its qualified default accepts a 1,000,000-token request and exposes a measured 1,183,301-token KV pool. Reproducible build instructions, the compact-cache runtime patches, and raw correctness/performance measurements are included in the recipe.

## Model

The checkpoint follows the source DeepSeek V4 Flash architecture: 43 blocks, 4,096 hidden size, 256 routed experts with top-6 routing, one shared expert, integrated dSpark speculative blocks, and a configured maximum position length of 1,048,576 tokens.

An inference runtime must support DeepSeek V4 and EXL3 with the MCG codebook. The files use the standard Hugging Face sharded checkpoint layout and include the EXL3 tensor-storage mapping needed to load the quantized expert projections.

Use the tokenizer and prompt formatting described by the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). The original model and this quantized checkpoint are distributed under the included MIT license.
