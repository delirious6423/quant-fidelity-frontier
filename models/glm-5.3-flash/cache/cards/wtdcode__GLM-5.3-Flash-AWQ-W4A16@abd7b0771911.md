---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
---

# GLM-5.3-Flash-AWQ-W4A16

Quantized version of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).

If you come here with older cards like A100, A6000 or 3090, consider using [our vLLM fork](https://github.com/wtdcode/vllm-backport) which has served billions of tokens for frontier models with older GPUs.

## What is quantized

Weight-only INT4 (symmetric, group size 128) via AWQ, stored in the compressed-tensors `pack-quantized` format. Calibrated from the official FP8 release (dequantized to BF16 first).

**Quantized (INT4 W4A16):**
- Routed MoE experts in the 42 MoE decoder layers (layers 3-44): `mlp.experts.{0..287}.{gate_proj, up_proj, down_proj}` (~312B of the 321B parameters)

**Kept in BF16 (not quantized):**
- Token embedding (`embed_tokens`) and `lm_head`
- Linear attention / KDA (`self_attn.{q,k,v,o}_proj`, forget gate) and DSA/MLA attention (`q_a/q_b/kv_a/kv_b_proj`, indexer)
- Hyper-connections (`attn_hc.*`, `ffn_hc.*`)
- MoE router (`mlp.gate`) and shared experts (`mlp.shared_experts.*`)
- Dense MLPs of the first 3 layers
- Vision encoder (`model.visual.*`)
- NextN/MTP layer (`layers.45.*`, dequantized from the FP8 source to BF16)
