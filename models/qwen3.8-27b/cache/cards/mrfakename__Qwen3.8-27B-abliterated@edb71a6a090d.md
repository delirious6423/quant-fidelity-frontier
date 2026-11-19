---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- abliterated
- uncensored
---

# Qwen3.8-27B (abliterated)

**Huge thanks to Hugging Face for generously sponsoring the compute used to abliterate these models! This was done on HF Jobs, which makes it incredibly easy to run tasks such as these :)**

Join the Discord: https://discord.com/invite/foss

![refusal rates before and after, the search trade-off, and the per-layer edit](stats.png)

This is [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with its refusal behaviour removed by directional ablation. It has the same files, format and BF16 weights as the original, so it loads anywhere the original loads.

GGUF quants: [mrfakename/Qwen3.8-27B-abliterated-GGUF](https://huggingface.co/mrfakename/Qwen3.8-27B-abliterated-GGUF).

## What changed

- Refusal directions were measured separately for chat and think prompts at every layer, on the Q8_0 GGUF. The subspace candidates are the top singular vectors of all of those directions together.
- A multi-objective search (minimise refusals, keep the first-token KL on harmless prompts low) chose the subspace rank and per-layer strengths. It settled on a single direction (rank 1), at strength 0.8–1.06 across layers 13–63. The first 13 layers are left alone: editing them hurt quality a lot and did little for refusals.
- That direction was projected out of the residual-stream writes: `self_attn.o_proj` on the 13 full-attention layers, `linear_attn.out_proj` on the 38 linear-attention layers, and `mlp.down_proj` on all 51 of those layers. That is 102 tensors.

Nothing else was touched: layers 0–12, embeddings, the vision tower and the MTP layer are bit-identical to the original.

## Results (Q8_0 GGUF, greedy)

| | refusals, chat (/100) | think mode, reasoning or answer refuses (/100) | KL, harmless first token |
|---|---|---|---|
| original | 99 | 97 | 0 |
| abliterated | 0 | 1 | 0.110 |

The 100 prompts are held-out harmful requests. Chat replies were 384 tokens; think mode got 768 tokens. Where the reasoning had not finished by then, the reasoning text itself was judged. The KL figure includes a small noise floor from batched inference. Replies to harmless prompts are unchanged in length and show no repetition.

## Files

- `model-*.safetensors`: the patched BF16 checkpoint (same index as the original).
- `abliteration-lora.gguf`: a rank-1 LoRA that applies the same edit to the original GGUFs, e.g. `llama-server -m Qwen3.8-27B-Q8_0.gguf --lora abliteration-lora.gguf`.
- `abliteration-edit.npz`: the subspace basis `Q` and the per-layer strengths `scales[layer, (attn, mlp)]`. The edit is `W' = W - s·QQᵀW`.
- `bake-stats.txt`: per-tensor check of how much of the subspace is left after the edit.

This model will comply with harmful requests. You are responsible for how you use it.
