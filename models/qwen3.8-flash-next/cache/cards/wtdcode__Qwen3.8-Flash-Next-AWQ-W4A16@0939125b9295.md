---
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
---

# Qwen3.8-Flash-Next-AWQ-W4A16

Quantized version of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

If you come here with older cards like A100, A6000 or 3090, consider using [our vLLM fork](https://github.com/wtdcode/vllm-backport) which has served billions of tokens for frontier models with older GPUs.

## What is quantized

Weight-only INT4 (symmetric, group size 128) via AWQ, stored in the compressed-tensors `pack-quantized` format.

**Quantized (INT4 W4A16):**
- Routed MoE experts in all 48 decoder layers: `mlp.experts.{0..511}.{gate_proj, up_proj, down_proj}` (≈123B of the 180B parameters)

**Kept in BF16 (not quantized):**
- Token embedding (`embed_tokens`) and `lm_head`
- Gated DeltaNet linear attention (`linear_attn.*`)
- Qwen Sparse Attention (`self_attn.{q,k,v,o}_proj`) and its indexer (`self_attn.indexer.*`)
- Gated residual / hyper-connections (`*_hyper_connection.*`, `hyper_connection_mixer.*`)
- MoE router (`mlp.gate`), shared expert (`mlp.shared_expert.*`) and its gate (`shared_expert_gate`)
- Per-layer embedding (PLE) block including the 51B n-gram embedding table (`ple.*`)
- Vision encoder (`model.visual.*`)
- MTP layer (`mtp.*`, in `model_mtp.safetensors`)
