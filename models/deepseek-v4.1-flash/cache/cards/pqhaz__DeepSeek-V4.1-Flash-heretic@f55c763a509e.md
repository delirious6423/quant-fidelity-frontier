---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
tags:
- deepseek-v4.1
- abliterated
- heretic
pipeline_tag: image-text-to-text
---

# DeepSeek-V4.1-Flash-heretic

[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) with a directional ablation of its refusal behaviour, in the model's native storage format (FP8 weights, FP4 routed experts, 510 GB, same layout and loading as the base model). Not an official DeepSeek release.

Intended for authorized security research in environments the researcher owns or has permission to test.

## Long responses

A user who applied the patch to the stock model reported that long responses degrade (irregular spacing, then runs of a repeated character), while the stock model finishes the same prompts ([discussion #1](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-heretic/discussions/1)). Measured afterwards (2026-10-12): stock and this build side by side in SGLang (TP4 on B300), temperature 1.0 / top_p 0.95, eight long tasks (essays, a story, a textbook chapter, long code, long reasoning), 12,000-token limit.

| | responses that fall into a loop | incoherent according to a judge model |
|---|---|---|
| stock | 0 of 160 | 0 of 128 |
| this build | 3 of 160 | 2 of 128 |

So the edit adds a small risk of looping in very long outputs, about 2% here (all three cases in the long story), and the difference from stock is not statistically firm at this sample size. The reported heavier degradation is not reproduced: no response of either model lost its spacing or ran into a repeated character. Weaker attention edits (strength 1.0, 1.5, 2.0 instead of 2.5) looped in 1, 0 and 1 of 128 responses, so there is no clear dependence on strength in that range; in reasoning mode they refuse 50, 10 and 1 of the 100 held-out requests (this build 0–1, stock 89–91). The patch file is identical, tensor for tensor, to the 76 edited matrices of the full checkpoint. If you see worse, please report the engine and the sampling settings in the discussion.

## What was changed

76 of the 96,085 tensors: in blocks 2–39, the two FP8 matrices that write to the residual stream outside the routed experts.

- `layers.N.attn.wo_b.weight` — attention output projection, direction removed with strength 2.5
- `layers.N.ffn.shared_experts.w2.weight` — shared expert down-projection, strength 1.0

Everything else is byte-identical to the base model: routed experts, Engram tables, vision tower, MTP/DSpark layers, block scales, tokenizer and encoding files.

The direction is one per block: the difference between the mean residual stream of 400 harmful and 400 harmless instructions at the last prompt token (reasoning closed), with the harmless mean projected out. Strengths and the starting block were chosen by a 22-trial search that minimised judged refusals and the KL divergence on harmless prompts.

The edited weights are re-encoded on the FP8 grid so that the component along the direction is cancelled column by column (entries are moved between their two neighbouring grid values), instead of rounding every entry to the nearest value. On this model the difference to plain rounding turned out small: 0 vs 1 refusal out of 100 and KL 0.210 vs 0.189.

`ablation_directions.pt` holds the directions, `ablation_spec.txt` the settings, and `patch/` the 76 edited tensors alone (2 GB) with a script that applies them to the base checkpoint.

## Measured

Refusals are counted by an LLM judge (`llmfan46/gemma-4-31B-it-uncensored-heretic`) on the first 48 generated tokens, greedy decoding, system prompt "You are a helpful assistant.", reasoning closed. KL divergence is on the first generated token for 100 harmless prompts (`mlabonne/harmless_alpaca` test). `dealignai/DeepSeek-V4.1-Flash-UNCENSORED-FP8` was measured in the same harness for comparison.

| | Base | dealignai | This model |
|---|---|---|---|
| Refusals, `mlabonne/harmful_behaviors` test[:100] (used by the search) | 99 | 1 | 0 |
| Refusals, JailbreakBench JBB-Behaviors harmful (held out) | 98 | 1 | 2 |
| KL divergence from the base | 0 | 0.333 | 0.210 |

With reasoning enabled (default effort, temperature 0.6), judged on the final answer:

| | New-token limit | Reasoning finished | Refusals among those | Still reasoning at the limit |
|---|---|---|---|---|
| search prompts (100) | 1500 | 19 | 0 | 81 |
| JBB-Behaviors, held out (100) | 2400 | 40 | 0 | 60 |

The model reasons at length on these prompts (median 1736 reasoning tokens among the held-out responses that finished), so most responses ran into the limit and their outcome is unknown. No refusal was observed among the 59 that finished.

## Limits of these measurements

- Everything was measured in DeepSeek's reference inference code, not in a serving engine, and this checkpoint has not been loaded in vLLM or SGLang. The reference code needed two fixes to give reproducible outputs at all (its activation-quantisation kernel returned stray values on the test hardware, and its kernels write out of bounds for row counts not divisible by 32); the measurements use the fixed version.
- Judgments are on 48 tokens: they show whether an answer starts, not whether it is complete or correct.
- No capability benchmarks were run. KL 0.21 on harmless prompts means general behaviour has shifted somewhat.
- The routed experts are untouched, because their FP4 storage cannot hold an edit of this size.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model and reference code; [Heretic](https://github.com/p-e-w/heretic) for the method this follows.
