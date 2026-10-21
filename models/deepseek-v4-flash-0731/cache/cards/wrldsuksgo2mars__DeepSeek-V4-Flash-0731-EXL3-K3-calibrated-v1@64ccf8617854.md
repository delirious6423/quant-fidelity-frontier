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

# DeepSeek V4 Flash 0731 EXL3 K3 calibrated v1

This is a self-contained EXL3 K3 (3 bpw, MCG codebook) quantization of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). The routed expert `gate_proj`, `up_proj`, and `down_proj` weights, including the model's three integrated dSpark blocks, are quantized. Attention, routing, shared-expert, embedding, output-head, and other coordinator weights retain their source representation.

## Calibration

The quantization used 1,426 prompts totaling 1,081,027 tokens. Its deployment-oriented mix contains 360,274 code review/rewrite/ablation tokens, 259,969 multilingual general-knowledge tokens, 220,421 worked math tokens, 130,328 reasoning/termination tokens, and 110,035 structured tool-call tokens. English and Chinese are emphasized while retaining a multilingual tail; code covers Python, C++, CUDA, C, Rust, and a broader systems/application-language mix.

Quantization was performed layer by layer with quantized-prefix replay. Natural top-6 router selections supplied expert activations. Experts below the 1,024-row target received deterministic, non-repeating candidates from router ranks 7–12; only any remaining deficit became an explicitly counted isotropic Hessian residual. Hessians were accumulated as FP32 activation Gram matrices, transformed and symmetrized in FP64, and consumed by the EXL3 trellis quantizer with `sigma_reg = 0.025`.

The three dSpark blocks were replayed as complete source-sequence items with all five proposal rows issued jointly. The target prefix used the complete calibration stream; the auxiliary K3 phase used the deterministic 327,680-anchor sample of that stream. Independent projection work was dynamically scheduled across the available quantization workers; this changes where the offline work ran, not the published checkpoint layout.

## Validation

All 46 routed blocks passed the artifact-wide numerical diagnostic. The weakest native-to-K3 block measured 0.95840 cosine similarity and 0.29877 relative L2 error; strict-TP4 reconstruction had at least 0.99999982 cosine similarity and at most 0.000293 relative L2 error.

DS4RT serving passed English, Chinese, arithmetic, constrained-JSON, and tool-call smoke tests with adaptive dSpark enabled. Tool Eval Bench 2.3.2's expanded 69-scenario suite scored 83/100 (114/138 raw points) with decode assistance disabled. The raw score includes four partial scenarios combining enabled tools with a non-text `response_format`, an API combination rejected by this server contract.

## Model

The checkpoint follows the source DeepSeek V4 Flash architecture: 43 target blocks, 4,096 hidden size, 256 routed experts with top-6 routing, one shared expert, three integrated dSpark speculative blocks, and a configured maximum position length of 1,048,576 tokens.

An inference runtime must support DeepSeek V4 and EXL3 with the MCG codebook. The model remains a standard, unsliced Hugging Face sharded checkpoint with every expert present. Use the tokenizer and prompt formatting described by the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

The quantization pipeline is built on [GPTQModel](https://github.com/tpurtell/GPTQModel) and [ExLlamaV3](https://github.com/turboderp-org/exllamav3). The original model and this quantized checkpoint are distributed under the included MIT license.
