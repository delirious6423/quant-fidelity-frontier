---
library_name: speculators
base_model:
- inference-optimization/Qwen3.8-27B-DSpark-FP8-DYNAMIC
- Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: text-generation
tags:
- speculative-decoding
- dspark
- qwen3
- fp8
- dummy
---

# Qwen3.8-27B-DSpark-FP8-DYNAMIC-30L

**Dummy 30-layer FP8_DYNAMIC DSpark drafter for size/latency experiments. Not a trained checkpoint.**

Generated from the 5-layer base drafter [inference-optimization/Qwen3.8-27B-DSpark-FP8-DYNAMIC](https://huggingface.co/inference-optimization/Qwen3.8-27B-DSpark-FP8-DYNAMIC) by cycling its 5 layers (new layer `i` = base layer `i mod 5`, including each layer's `weight_scale` tensors) and setting `transformer_layer_config.num_hidden_layers = 30` (and `layer_types` accordingly). The `quantization_config` is unchanged from the base. Shared tensors (`fc`, norms, markov/confidence heads) are copied as-is.

Do not use for acceptance-rate or quality evaluation — layers beyond the first 5 are weight duplicates. Pair with [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) as the verifier, as in the base model.
