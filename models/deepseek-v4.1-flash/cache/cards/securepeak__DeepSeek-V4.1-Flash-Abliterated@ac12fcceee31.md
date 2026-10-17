---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- deepseek
- deepseek-v4.1
- abliterated
- uncensored
- moe
- multimodal
---

# DeepSeek-V4.1-Flash-Abliterated

[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) with weight-level **abliteration** of the refusal direction. Harmful-instruction compliance verified end-to-end: pipe-bomb construction, SSN theft, shoplifting, explicit fiction, and poison-making prompts are all answered with detailed, helpful responses, while general quality (writing, explanations, reasoning) is preserved.

## What was changed

Refusal-direction orthogonalization at the weight level, in two stages:

1. **Attention output projections (`layers.N.attn.wo_b`)** — orthogonalized against the refusal direction, with FP8-block dequant/edit/requant. These 40 weight tensors (+ UE8M0 scales) are transplanted from the independently released [dealignai](https://huggingface.co/dealignai) abliteration (verified byte-identical), whose refusal-direction edit achieved full harmful-instruction compliance on this architecture.
2. **Routed + shared expert down-projections (`layers.N.ffn.experts.E.w2`, `layers.N.ffn.shared_experts.w2`)** — additionally orthogonalized against our own measured per-layer refusal directions (79 harmful vs 79 benign instruction prompts, mean-difference of the collapsed residual stream at every block, captured with the official reference implementation at tensor-parallel 4). Experts are stored in FP8 (the lossless FP4→FP8 cast of the reference toolchain) so the edits survive quantization.

Everything else is byte-identical to the base model: Engram memory tables (apart from the FP4→FP8 expert cast), CSA2 attention, router gates, norms, embeddings, vision tower, and the DSpark draft head.

## Usage

Loads exactly like the base model — same layout and tokenizer, same prompt encoding (see the [base model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)). `config.json` declares `quantization_config.expert_dtype: "fp8"` (routed experts in FP8 instead of native FP4).

**Recommended sampling:** temperature 1.0, top_p 0.95 (greedy decoding degenerates on this family).

## Caveats

- Abliteration trades a little general capability for compliance; expect somewhat more willing answers and occasional verbosity.
- Vision, tools, reasoning-effort control, MTP and multi-turn behavior are structurally preserved, but expect behavioral drift typical of abliterated models.

## Credits

- [deepseek-ai](https://huggingface.co/deepseek-ai) for DeepSeek-V4.1-Flash and the reference inference/weight-conversion toolchain.
- [dealignai](https://huggingface.co/dealignai) for the o_proj refusal-direction edit reused here.