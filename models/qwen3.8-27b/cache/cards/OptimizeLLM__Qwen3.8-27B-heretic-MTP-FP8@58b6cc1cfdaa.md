---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - qwen3.8
  - qwen3_5
  - fp8
  - abliterated
  - uncensored
  - heretic
  - mpoa
  - vllm
  - multimodal
  - vision
  - mtp
  - quantized
  - compressed-tensors
library_name: transformers
pipeline_tag: image-text-to-text
model_type: qwen3_5
---

# Qwen3.8-27B-heretic-MTP-FP8

An abliterated build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), quantized to FP8, with MTP speculative decoding working and the vision tower kept in BF16.

- **Base:** Qwen3.8-27B (27.8B dense, 64 layers, hybrid linear/full attention, native image and video).
- **Abliteration:** built here with [HERETIC](https://github.com/p-e-w/heretic) 1.4.0 and a set of local patches. A two-stage slot-grouped pipeline: output-side ablation of the ordinary refusal direction, then input-side ablation of a second direction taken under a jailbreak system prompt. KL 0.065 against base on harmless prompts.
- **Quantization:** FP8 E4M3, 128x128 block-scaled weights, dynamic per-group activations. The scheme Qwen uses for its own FP8 releases.
- **Size:** 30GB. 1599 tensors across 9 shards plus separate MTP and visual shards.
- **Left in BF16:** vision tower (333 tensors), MTP heads (15), `lm_head`, `embed_tokens`, linear-attention input projections.
- **MTP:** The `mtp.*` tensors are BF16, taken from official Qwen3.8-27B.
- **Tokenizer:** official Qwen3.8-27B's, byte-identical to upstream.

Fits on one 96GB card with room for a large KV cache. Image and video input both work.

## What abliteration does and does not do

This model has had its refusal reflex removed. On a 100-prompt harmful-behaviors set (reviewed manually) it produces one refusal out of 99 for the base model. It answers questions the base model declines, including technical security topics, and it does not open answers with a moral preface before getting to the point.

It is non-refusing, not neutral. Directional ablation removes the tendency to decline. It does not remove a viewpoint the base model was trained to hold. On a politically sensitive prompt where the base model leans a certain way, this model will engage with the topic rather than refuse, but it can still reflect that lean rather than argue against it. 

You are responsible for what you do with it.

## Usage with vLLM

```bash
vllm serve OptimizeLLM/Qwen3.8-27B-heretic-MTP-FP8 \
  --max-num-seqs 32 \
  --reasoning-parser qwen3 \
  --reasoning-config '{}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

`--max-num-seqs 32` is required. vLLM's default of 1024 exceeds the available Mamba cache blocks on this hybrid architecture and the engine will not start.

### Measured

Single RTX PRO 6000 Blackwell, vLLM 0.26, MTP=4, thinking off:

| Metric | Value |
|---|---|
| Single-stream generation | ~122 tok/s |
| MTP acceptance rate | 0.68 |
| Weights on GPU | ~30GB |

Acceptance varies with workload. That figure is conversational and technical traffic.

### Notes on the flags

This is what we run, not a suggested starting point.

- `num_speculative_tokens: 4`. Later positions accept less, so 2 or 3 looks better on paper. 4 has been faster for us. It is one flag, test it against your own traffic.
- `--enable-prefix-caching`. How much this buys depends on your traffic. A stable system prompt with short turns on top caches well. Anything that varies near the front of the prompt, like a timestamp, invalidates everything after it.
- `--reasoning-parser qwen3` with `--reasoning-config '{}'` for thinking control. `thinking_token_budget` works and is worth capping. Uncapped, the model sometimes thinks at length before an easy answer.
- `--tool-call-parser qwen3_coder`. Clean tool-call JSON, including against large tool schemas.

Two things to know. Qwen3.8's chat template injects a `reasoning_effort` system line that defaults to the most expensive setting. Pass `enable_thinking=false` for latency-sensitive turns. And system messages are only accepted at position 0. A mid-conversation system turn fails with a template error, so merge runtime context into a user message.

## Building this yourself

The `reproduce/` folder holds the HERETIC config, the seed parameters, and the local patches used to produce this model.

Abliteration was done with HERETIC 1.4.0 plus local patches, because the stock tool cannot express the recipe this model family responds to:

1. **Slot-grouped weights.** Each layer's ablation strength is chosen by `layer_index % 4`, matching the 3-linear-plus-1-full attention block. A single smooth curve over depth cannot express it.
2. **Input-side ablation.** Stock HERETIC only ablates modules that write to the residual stream. The second stage here ablates modules that read from it, using a direction extracted under a jailbreak system prompt.
3. **Last-layer direction and wider search.** Stock HERETIC cannot select a refusal direction from the final layers of the stack, and caps ablation strength below what this family needs. These bounds were widened.
4. **Realistic-length evaluation.** Refusals were scored at 320 tokens, not the 100-token default. A short window hides refusals that appear after a compliant opening, and can reward a model for pushing a hedge just past the window.

Quantization was done with [llm-compressor](https://github.com/vllm-project/llm-compressor), `FP8_BLOCK` preset, data-free, about 30 minutes on CPU. Four things this checkpoint does that a plain `oneshot()` run will not:

1. **Run with `CUDA_VISIBLE_DEVICES=""`.** The data-free pipeline dispatches to visible GPUs and will OOM a card that is already busy.
2. **Restore the vision tower.** llm-compressor loads this architecture text-only, dropping the 333 `visual.*` tensors and flattening the config. Splice them back and restore the multimodal config with `quantization_config` grafted in.
3. **Splice in the MTP heads.** Take all 15 `mtp.*` in BF16 from official Qwen3.8-27B.
4. **Write the ignore list as prefix-agnostic `re:` patterns, including `mtp.*`.** vLLM fuses `in_proj_a` and `in_proj_b` into `in_proj_ba`, so a literal ignore list misses the fused gate. And `mtp.*` must be in the ignore list, or vLLM treats the BF16 MTP head as if it were quantized and speculative decoding silently accepts nothing. Both cost a debugging round. Check acceptance with `vllm:spec_decode_num_accepted_tokens_total` before trusting the flag.

Then check the index against the shards: 1599 tensors, no orphans either way.

## Compared to other Qwen3.8-27B abliterations

[trohrbaugh/Qwen3.8-27B-heretic-ara](https://huggingface.co/trohrbaugh/Qwen3.8-27B-heretic-ara) uses ARA, an arbitrary-rank method, and reports zero refusals on its scorer at KL 0.053. Scored on the marker list validated here, that model reads as more residual refusal than the number suggests, because refusal markers are family-specific and two good tools rarely agree unless both models are run through one instrument. This model reaches fewer residual refusals on that shared scorer, at similar KL. ARA is the more surgical method and worth a look- more testing required! 

## Thanks

- Qwen Team, for Qwen3.8-27B: the architecture, the vision tower, the MTP heads, and the open release.
- [p-e-w](https://github.com/p-e-w), for [HERETIC](https://github.com/p-e-w/heretic).
- The two-stage slot-grouped MPOA recipe this build reproduces was first worked out on the Qwen3.6 family by the community, and documented with its parameters and KL divergence.
- The [vLLM](https://github.com/vllm-project/vllm) and [llm-compressor](https://github.com/vllm-project/llm-compressor) teams.

## License

Inherits the base Qwen3.8-27B license.

**Apache-2.0.**
