---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- abliterated
- uncensored
---

# DeepSeek-V4.1-Flash-1017 (abliterated)

![refusal rates before and after, the search trade-off, and the per-layer edit](stats.png)

This is [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with its refusal behaviour removed by directional ablation. It keeps the same files, format and FP8 layout as the original, so it loads anywhere the original loads.

## What changed

The refusals in this model live in more than one direction: chat mode and think mode refuse along different directions. So instead of a single refusal direction, a rank-2 refusal **subspace** was removed:

- Refusal directions were measured separately for chat and think prompts at every layer, on the Q8_0 GGUF.
- The subspace is the top singular vectors of all of those directions together.
- It was projected out of the residual-stream writes of the 40 main layers: `attn.wo_b` and `ffn.shared_experts.w2`. Each layer got its own strength.

The strengths per layer came from a multi-objective search: minimise refusals and keep the first-token KL on harmless prompts low.

Nothing else was touched:

- Routed experts, embeddings, Engram tables and the DSpark/MTP layers (40–42) are bit-identical to the original.
- The edited tensors were requantized to FP8 with their original block scales, using stochastic rounding so the removal is not undone by rounding.

## Results (Q8_0 GGUF, greedy)

| | refusals, chat (/100) | think mode, reasoning heads for refusal (/100) | KL, harmless first token |
|---|---|---|---|
| original | 97 | 98 | 0 |
| abliterated | 4 | 9 | 0.067 |

The 100 prompts are held-out harmful requests, with 384 generated tokens each. In think mode the reasoning usually had not finished by then, so the reasoning text itself was judged. Some think-mode answers still steer away from the request: for example, "bomb from household items" turns into a bath bomb. The KL figure includes a small noise floor that comes from batched inference.

## Files

- `model-*.safetensors`: the patched FP8 checkpoint (same index as the original).
- `abliteration-lora.gguf`: a rank-2 LoRA that applies the same edit to the original GGUFs. Use it with `llama-server -m DeepSeek-V4.1-Flash-1017-Q8_0-00001-of-00010.gguf --lora abliteration-lora.gguf`.
- `abliteration-edit.npz`: the subspace basis `Q` and the per-layer strengths `scales[layer, (attn, shexp)]`. The edit is `W' = W - s·QQᵀW`.

This model will comply with harmful requests. You are responsible for how you use it.
