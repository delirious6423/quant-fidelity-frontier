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

# DeepSeek V4 Flash 0731 EXL3 K2 calibrated v1

This is a self-contained EXL3 K2 (2 bpw, MCG codebook) quantization of [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731). The routed expert `gate_proj`, `up_proj`, and `down_proj` weights, including the model's three integrated dSpark blocks, are quantized. Attention, routing, shared-expert, embedding, output-head, and other coordinator weights retain their source representation.

## Calibration

The quantization used 1,426 prompts totaling 1,081,027 tokens. Its deployment-oriented mix contains 360,274 code review/rewrite/ablation tokens, 259,969 multilingual general-knowledge tokens, 220,421 worked math tokens, 130,328 reasoning/termination tokens, and 110,035 structured tool-call tokens. English and Chinese are emphasized while retaining a multilingual tail; code covers Python, C++, CUDA, C, Rust, and a broader systems/application-language mix. Worked math draws from NuminaMath-CoT and Chinese CMATH material, while structured output uses packed xLAM-style schemas, queries, and expected calls.

Quantization was performed layer by layer with quantized-prefix replay, so each later block was calibrated against activations produced through the already-quantized prefix. Natural top-6 router selections supplied expert activations. Experts below the 1,024-row target received deterministic, non-repeating candidates from router ranks 7–12; only any remaining deficit became an explicitly counted isotropic Hessian residual. Hessians were accumulated as FP32 activation Gram matrices, transformed and symmetrized in FP64, and consumed by the EXL3 trellis quantizer with `sigma_reg = 0.025`.

The three dSpark blocks were replayed as complete source-sequence items with all five proposal rows issued jointly. Independent projection work was dynamically scheduled across two Blackwell RTX GPUs and four DGX Sparks; this changes where the offline work ran, not the published checkpoint layout.

## Serve it on one DGX Spark

The [DGX Spark recipe](https://github.com/tpurtell/ds4-mia-exl3-k2-1spark)
starts this checkpoint on one Spark by default, with the native dSpark draft,
NVFP4 DS-MLA cache, and a one-million-token request ceiling. Its prebuilt
runtime is `ghcr.io/tpurtell/ds4-mia-exl3-k2-1spark:latest`; clone the recipe,
copy `.env.example` to `.env`, then run `./launch.sh --nodes 1 --model k2`.
The same recipe can run K2 or the native checkpoint across two Sparks for a
matched comparison.

## Validation

All 46 routed blocks passed the numerical publication gate. On a source-disjoint 88-prompt / 67,437-token held-out split, the weakest target block had 0.91196 cosine similarity and 0.41132 relative L2 error against the native experts. The three dSpark blocks passed their separate RMS-isotropic proxy gate, and strict-TP4 reconstruction across the complete model had at least 0.99999988 cosine similarity and at most 0.000296 relative L2 error.

In a matched DS4RT run with adaptive dSpark, v1 accepted 3,645 of 6,230 proposed draft tokens (58.51%) and produced 99.66 weighted decode tok/s. The v0 control accepted 3,570 of 6,175 (57.81%) at 98.34 tok/s. Tool Eval Bench 2.3.2's expanded 69-scenario suite scored v1 at 86/100 with decode assistance disabled and enabled; the matched v0 control scored 88/100 in both modes. No request or backend errors occurred in the scored runs.

## Model

The checkpoint follows the source DeepSeek V4 Flash architecture: 43 target blocks, 4,096 hidden size, 256 routed experts with top-6 routing, one shared expert, three integrated dSpark speculative blocks, and a configured maximum position length of 1,048,576 tokens.

An inference runtime must support DeepSeek V4 and EXL3 with the MCG codebook. The model remains a standard, unsliced Hugging Face sharded checkpoint with every expert present. Use the tokenizer and prompt formatting described by the [original model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

The quantization pipeline is built on [GPTQModel](https://github.com/tpurtell/GPTQModel) and [ExLlamaV3](https://github.com/turboderp-org/exllamav3). The original model and this quantized checkpoint are distributed under the included MIT license.
