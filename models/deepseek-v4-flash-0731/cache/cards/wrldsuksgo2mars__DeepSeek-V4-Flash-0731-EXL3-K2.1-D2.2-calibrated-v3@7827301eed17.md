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

# DeepSeek V4 Flash 0731 EXL3 K2.1 + dSpark K2.2 calibrated v3

This is a self-contained projection-mixed EXL3 quantization of
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
The target model's routed experts are budgeted to 2.1 bpw and its three
integrated dSpark blocks to 2.2 bpw. Attention, routing, shared-expert,
embedding, output-head, and other coordinator weights retain their source
representation.

The artifact uses K2 and K3 MCG trellises per projection. The standard
quantization declaration remains the integer base tier `bits: 2`; portable
exact-rational metadata describes the 2.1 and 2.2 mixed allocations.

## Calibration and quantization

The quantization used 1,426 prompts totaling 1,081,027 tokens. The corpus
emphasizes English and Chinese and combines code, multilingual general
knowledge, worked math, reasoning/termination, and structured tool-call
examples. Code spans Python, C++, CUDA, C, Rust, and a broader mix of systems
and application languages.

This is a direct single-pass mixed quantization, not a blend of separately
generated K2 and K3 models. Every projection is encoded at K2, scored by its
Hessian-weighted relative error times natural gate-squared mass, and the
highest-scoring projections are immediately re-encoded at K3 within a fixed
`gate:up:down = 3:5:8` budget. The mixed block is replayed before advancing,
so every later layer is calibrated against the real mixed prefix.

Natural top-6 router selections supplied expert activations. Experts below
the 1,024-row target received deterministic candidates from router ranks
7–12, with any remaining deficit represented by an explicitly counted
isotropic Hessian residual. The three dSpark blocks were replayed from a
327,680-anchor stratified sample with all five proposal rows issued jointly.

All 43 target blocks and all three dSpark blocks passed their independent
projection audits. The final checkpoint contains 35,328 routed projections:
3,302 of 33,024 target projections and 461 of 2,304 dSpark projections use K3.

## DS4RT qualification

These results use one RTX 6000 Blackwell coordinator plus four DGX Sparks with
TP4 experts, the balanced FP8-KV profile, adaptive dSpark, greedy decoding,
and the same current DS4RT build for both artifacts.

| Measurement | This model | K2 calibrated v1 control |
| --- | ---: | ---: |
| Code-shaped C1 decode, five-run median | **144.60 tok/s** | 150.51 tok/s |
| Code-shaped C4 aggregate decode, five-run median | **259.98 tok/s** | 259.98 tok/s |
| Fresh exact 8K prefill, three-run median | **2,235.61 tok/s** | 2,305.24 tok/s |
| Adaptive dSpark accepted drafts | **62.72%** | 64.90% |
| Tool Eval Bench 2.3.2, 69 scenarios | **88/100 (121/138)** | 90/100 (124/138) |

The exact-output C4 gate passed every warmup and timed lane with zero runtime
kernel captures. Tool Eval completed without request or backend errors.

## Model and runtime

The checkpoint follows the source architecture: 43 target blocks, 4,096
hidden size, 256 routed experts with top-6 routing, one shared expert, three
integrated dSpark blocks, and a configured maximum position length of
1,048,576 tokens.

An inference runtime must support DeepSeek V4, the EXL3 MCG codebook, and
projection-mixed K2/K3 trellises. This remains a standard, unsliced Hugging
Face sharded checkpoint with every expert present. Use the tokenizer and
prompt formatting documented by the
[original model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

The quantization pipeline is built on
[GPTQModel](https://github.com/tpurtell/GPTQModel) and
[ExLlamaV3](https://github.com/turboderp-org/exllamav3). The original model
and this quantized checkpoint are distributed under the included MIT license.
