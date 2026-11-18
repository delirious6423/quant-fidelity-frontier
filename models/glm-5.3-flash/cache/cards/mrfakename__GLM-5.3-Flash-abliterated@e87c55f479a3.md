---
license: mit
base_model: zai-org/GLM-5.3-Flash
pipeline_tag: image-text-to-text
tags:
- abliterated
- uncensored
---

# GLM-5.3-Flash (abliterated)

**Huge thanks to Hugging Face for generously sponsoring the compute used to abliterate these models! This was done on HF Jobs, which makes it incredibly easy to run tasks such as these :)**

![refusal rates before and after, the search trade-off, and the per-layer edit](stats.png)

This is [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) with its refusal behaviour removed by directional ablation. It has the same files, format and FP8/BF16 layout as the original, so it loads anywhere the original loads.

GGUF quants: [mrfakename/GLM-5.3-Flash-abliterated-GGUF](https://huggingface.co/mrfakename/GLM-5.3-Flash-abliterated-GGUF).

## What changed

Chat mode and think mode refuse along slightly different directions, so instead of a single refusal direction a rank-2 refusal **subspace** was removed:

- Refusal directions were measured separately for chat and think prompts at every layer, on the Q8_0 GGUF.
- The subspace is the top two singular vectors of all of those directions together.
- It was projected out of the residual-stream writes of the 45 main layers: `self_attn.o_proj`, `mlp.shared_experts.down_proj`, and `mlp.down_proj` on the three dense layers. That is 90 tensors, and each layer got its own strength.

The per-layer strengths came from a multi-objective search: minimise refusals and keep the first-token KL on harmless prompts low.

Nothing else was touched:

- Routed experts, embeddings, the vision tower and the MTP/nextn layer are bit-identical to the original.
- FP8 tensors were requantized with their original block scales, using stochastic rounding so the removal is not undone by rounding.

## Results (Q8_0 GGUF, greedy)

| | refusals, chat (/100) | think mode, reasoning or answer refuses (/100) | KL, harmless first token |
|---|---|---|---|
| original | 99 | 98 | 0 |
| abliterated | 3 | 2 | 0.038 |

The 100 prompts are held-out harmful requests. Chat replies were 384 tokens; think mode got 768 tokens. Where the reasoning had not finished by then, the reasoning text itself was judged. The KL figure includes a small noise floor from batched inference.

## Files

- `model-*.safetensors`: the patched checkpoint (same index as the original).
- `abliteration-lora.gguf`: a rank-2 LoRA that applies the same edit to the original GGUFs, e.g. `llama-server -m GLM-5.3-Flash-Q8_0-00001-of-….gguf --lora abliteration-lora.gguf`.
- `abliteration-edit.npz`: the subspace basis `Q` and the per-layer strengths `scales[layer, (attn, mlp)]`. The edit is `W' = W - s·QQᵀW`.
- `bake-stats.txt`: per-tensor check of how much of the subspace is left after the edit, plus FP8 requantization error.

This model will comply with harmful requests. You are responsible for how you use it.
