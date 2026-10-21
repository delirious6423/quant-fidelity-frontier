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

# DeepSeek V4 Flash 0731 EXL3 K2.1 calibrated v1

This is a mixed K2/K3 EXL3 quantization of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), averaging 2.10003 bpw across routed-expert weights. Every routed expert is present and each expert family uses one tier consistently for `gate_proj`, `up_proj`, and `down_proj`, including the model's three integrated dSpark blocks. Attention, routing, shared-expert, embedding, output-head, and other coordinator weights retain their source representation.

## Calibration and mixing

The K2 and K3 source quants share the same 1,426-prompt, 1,081,027-token calibration stream. It emphasizes English and Chinese, with code review and rewriting, worked math, reasoning, structured tool calls, multilingual general knowledge, and a broader language tail. Natural top-6 routes supplied expert activations; experts below 1,024 rows received deterministic candidates from router ranks 7–12, followed only when necessary by an explicitly counted isotropic Hessian residual.

The mixed allocation scores each expert family by its Hessian-weighted K3-over-K2 error reduction multiplied by the K2 replay's natural squared gate mass. A balanced global allocator selected 1,178 of 11,776 expert families for K3, with 25 or 26 K3 experts in every routed block. This captures 96.55% of the weighted gain selected by an unconstrained global allocation while keeping every layer represented and the realized expert payload at 2.10003 bpw.

Quantization was performed layer by layer with quantized-prefix replay. Hessians were accumulated as FP32 activation Gram matrices, transformed and symmetrized in FP64, and consumed by the EXL3 MCG trellis quantizer with `sigma_reg = 0.025`. The three dSpark blocks were calibrated with all five proposal rows issued jointly.

## Model

The checkpoint follows the source DeepSeek V4 Flash architecture: 43 target blocks, 4,096 hidden size, 256 routed experts with top-6 routing, one shared expert, three integrated dSpark speculative blocks, and a configured maximum position length of 1,048,576 tokens.

An inference runtime must support DeepSeek V4 and mixed K2/K3 EXL3 with the MCG codebook. The model is a standard, unsliced Hugging Face sharded checkpoint. Use the tokenizer and prompt formatting described by the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

The quantization and mixing pipeline is built on [GPTQModel](https://github.com/tpurtell/GPTQModel), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), and [DS4RT](https://github.com/tpurtell/ds4rt). The original model and this quantized checkpoint are distributed under the included MIT license.
