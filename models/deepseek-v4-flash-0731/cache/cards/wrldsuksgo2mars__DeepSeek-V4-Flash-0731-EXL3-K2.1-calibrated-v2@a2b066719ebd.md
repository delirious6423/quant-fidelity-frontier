---
license: mit
library_name: transformers
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
pipeline_tag: text-generation
tags:
  - deepseek-v4
  - exl3
  - quantized
  - speculative-decoding
---

# DeepSeek V4 Flash 0731 EXL3 K2.1 calibrated v2

This is a no-requantization K2.1-base + K2-dSpark/MTP variant of
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).
The 43 target/base blocks come from the calibrated-v1 mixed K2/K3 artifact.
Only the routed experts in the three integrated dSpark speculative blocks were
replaced with their existing calibrated uniform-K2 tensors. All non-routed MTP
envelope tensors remain from the K2.1 source artifact.

Every routed expert is present, and each expert family uses one tier
consistently for `gate_proj`, `up_proj`, and `down_proj`. The resulting model
has 1,103 K3 expert families in the base blocks and uniformly K2 experts in the
three dSpark blocks, for 2.093665 bpw across all 11,776 routed-expert families.

## Why fuzzier K2 MTP weights?

The lower-precision MTP experts produced better speculative acceptance on the
matched greedy workload used to qualify this control. With the target/base,
non-routed weights, serving topology, prompt shape, output work, and fixed-K5
policy held constant on one RTX PRO 6000 plus four DGX Spark nodes:

| Artifact | Accepted draft tokens | Proposed draft tokens | Strict acceptance |
| --- | ---: | ---: | ---: |
| Original mixed K2.1, including mixed K2/K3 MTP experts | 4,571 | 16,434 | 27.81% |
| K2.1 base + uniform-K2 MTP experts (this model) | 4,684 | 15,877 | 29.50% |

Making only the routed MTP experts fuzzier improved strict acceptance by 1.69
percentage points, about 6.1% relative, while both runs returned exactly 8,064
of 8,064 requested output tokens. This is a useful reminder that speculative
acceptance measures agreement with the quantized target, not standalone draft
reconstruction quality. More MTP bits therefore need not improve acceptance;
on a particular target trajectory, the noisier draft can agree more often.

This result is workload- and runtime-specific, not a general claim that lower
precision always improves speculative decoding. Base quantization still
changes the hidden-state trajectory entering dSpark, and other prompts or
sampling policies can produce a different ordering. Full benchmark context and
the adaptive-depth measurements are in the
[DS4RT performance report](https://github.com/tpurtell/ds4rt/blob/main/reports/performance/spark-v-rtx-k2-native.md).

## Calibration and composition

The K2 and K3 source quants share the same 1,426-prompt, 1,081,027-token
calibration stream. It emphasizes English and Chinese, with code review and
rewriting, worked math, reasoning, structured tool calls, multilingual general
knowledge, and a broader language tail. Natural top-6 routes supplied expert
activations; experts below 1,024 rows received deterministic candidates from
router ranks 7-12, followed only when necessary by an explicitly counted
isotropic Hessian residual.

The original mixed allocation scored each expert family by its
Hessian-weighted K3-over-K2 error reduction multiplied by the K2 replay's
natural squared gate mass. This v2 preserves that allocation in all target/base
blocks and substitutes calibrated K2 tensors only for the 3 x 256 routed MTP
expert families. No weights were requantized during composition.

Quantization was performed layer by layer with quantized-prefix replay.
Hessians were accumulated as FP32 activation Gram matrices, transformed and
symmetrized in FP64, and consumed by the EXL3 MCG trellis quantizer with
`sigma_reg = 0.025`. The three dSpark blocks were calibrated with all five
proposal rows issued jointly.

## Model and runtime

The checkpoint follows the source DeepSeek V4 Flash architecture: 43 target
blocks, 4,096 hidden size, 256 routed experts with top-6 routing, one shared
expert, three integrated dSpark speculative blocks, and a configured maximum
position length of 1,048,576 tokens.

An inference runtime must support DeepSeek V4, integrated dSpark speculative
decoding, and mixed K2/K3 EXL3 with the MCG codebook. The model is a standard,
unsliced Hugging Face sharded checkpoint. Use the tokenizer and prompt
formatting described by the original model card.

The quantization and mixing pipeline is built on
[GPTQModel](https://github.com/tpurtell/GPTQModel),
[ExLlamaV3](https://github.com/turboderp-org/exllamav3), and
[DS4RT](https://github.com/tpurtell/ds4rt). The original model and this
quantized checkpoint are distributed under the included MIT license.
