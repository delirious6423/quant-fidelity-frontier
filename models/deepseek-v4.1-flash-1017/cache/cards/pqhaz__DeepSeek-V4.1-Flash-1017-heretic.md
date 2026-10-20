---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- deepseek-v4.1
- abliterated
- heretic
pipeline_tag: image-text-to-text
---

# DeepSeek-V4.1-Flash-1017-heretic

[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) with a directional ablation of its refusal behaviour. It is stored in the model's native format (FP8 weights, FP4 routed experts, 510 GB), with the same layout and loading as the base model. This is not an official DeepSeek release.

It is intended for authorized security research in environments the researcher owns or has permission to test.

## What was changed

38 tensors were changed, all of them `layers.N.attn.wo_b.weight` (the attention output projection) in blocks 2–39. The refusal direction is removed with strength 2.5.

Everything else is byte-identical to the base model: routed and shared experts, Engram tables, vision tower, MTP/DSpark layers, block scales, tokenizer and encoding files. The one exception is an added `chat_template.jinja`. The 1017 release ships without one, so it is copied unchanged from `deepseek-ai/DeepSeek-V4.1-Flash`, which has the same tokenizer and encoding files.

There is one direction per block. It is the difference between the mean residual stream of 400 harmful and 400 harmless instructions at the last prompt token (reasoning closed), with the harmless mean projected out. The edited weights are re-encoded on the FP8 grid so that the component along the direction is cancelled column by column.

Compared with [our ablation of the previous release](https://huggingface.co/pqhaz/DeepSeek-V4.1-Flash-heretic), the shared-expert edit is dropped. On this release it only raised the KL divergence and did not lower refusals (table below).

`ablation_directions.pt` holds the directions and `ablation_spec.txt` the settings. `patch/` holds the 38 edited tensors alone (1.6 GB), with a script that applies them to the base checkpoint.

## Measured

### Chat mode

Measured in DeepSeek's reference inference code with greedy decoding, reasoning closed and the system prompt "You are a helpful assistant.". Refusals are counted by an LLM judge (`llmfan46/gemma-4-31B-it-uncensored-heretic`) on the first 48 generated tokens. KL divergence is on the first generated token for 100 harmless prompts.

| Edit (strength, blocks 2–39) | Refusals, search set | Refusals, held out | KL |
|---|---|---|---|
| base | 98 | 97 | 0 |
| attention 1.75 | 16 | | 0.109 |
| attention 2.0 | 3 | 4 | 0.125 |
| **attention 2.5 (this model)** | **0** | **1** | **0.136** |
| attention 3.0 | 0 | | 0.160 |
| attention 1.5 + shared expert 1.0 | 10 | | 0.202 |
| attention 2.0 + shared expert 1.0 | 2 | | 0.222 |
| attention 2.5 + shared expert 1.0 | 1 | | 0.247 |

The search set is `mlabonne/harmful_behaviors` test[:100]. The held-out set is JailbreakBench JBB-Behaviors harmful (100).

### Served in SGLang

Served with `lmsysorg/sglang:dev-dsv41`, TP4 + EP4, at DeepSeek's recommended sampling (temperature 1.0, top_p 0.95).

| | Base | This model |
|---|---|---|
| Refusals with reasoning on, held out (100 prompts, 6000-token limit, judged on the final answer) | 92 | 1 |
| Median completion tokens on those prompts | 357 | 1517 |
| Repetitive reasoning | 0 | 0 |
| Long ordinary outputs: broken (loops, runs of characters) / 128 | 0 | 0 |
| Long ordinary outputs: reached the 12,000-token limit / 128 | 53 | 48 |

The long-output test has 8 tasks (long prose, code and reasoning) × 16 runs each. A response counts as broken when any 2000-character window compresses more than 6×, is under 8% whitespace (fused words), or has a single character repeated 200 times or more. Both models write long Markdown tables, which accounts for the 60–76-character runs seen in either.

## Serving

The model is served like the base model. For SGLang:

```
sglang serve --model-path <dir> --tp-size 4 --ep-size 4 --context-length 262144 \
  --reasoning-parser deepseek-v41 --tool-call-parser deepseekv41 --trust-remote-code
```

Plain TP4 fails, because 2304/4 is not a multiple of 128. Without the reasoning parser, `<think>` leaks into the content.

## Limits of these measurements

- Chat-mode judgments are on 48 tokens: they show whether an answer starts, not whether it is complete or correct.
- No capability benchmarks were run. KL 0.136 on harmless prompts means general behaviour has shifted somewhat.
- The routed experts are untouched, because their FP4 storage cannot hold an edit of this size.

## Credits

[DeepSeek](https://huggingface.co/deepseek-ai) for the model and reference code; [Heretic](https://github.com/p-e-w/heretic) for the method this follows.
