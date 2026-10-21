---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- nvfp4
- fp8
- kv-cache
- mtp
- speculative-decoding
- vllm
- llm-compressor
- hybrid
- gated-deltanet
datasets:
- HuggingFaceH4/ultrachat_200k
- openai/gsm8k
- sahil2801/CodeAlpaca-20k
- CohereLabs/aya_dataset
pipeline_tag: image-text-to-text
library_name: transformers
---

# Qwen3.8-27B-NVFP4-FP8KV-MTP by IG1

## Quantization

This model has been quantized using **llm-compressor v0.13.0** and **transformers v5.14.1**. It is based on the official example with a few modifications (see next section).

### Quantization particularities

The sequence length has been increased to 8192 and the number of samples to 1024.
The 1024 samples come from 4 different datasets:

- 256 general conversation samples (UltraChat)
- 256 math reasoning samples (GSM8K)
- 256 code samples (CodeAlpaca)
- 256 multilingual samples (Aya)

You can find the quantization script [here](nvfp4-kvfp8-mtp.py).

The recipe targets `Linear` layers with the `NVFP4` scheme, while excluding:
- `lm_head`
- `re:.*visual.*` (vision tower)
- `re:.*linear_attn.*` (Gated DeltaNet layers)

### About FP8 KV cache

Unlike our previous Qwen3.5/3.6 experiments, **FP8 KV cache works correctly on Qwen3.8** thanks to the observer fusion fix in llm-compressor 0.13.0.

The checkpoint contains **calibrated static FP8 scales** (`k_scale`/`v_scale`) for the 16 full-attention layers. These scales are **passive metadata**:

- **Without `--kv-cache-dtype fp8`**: vLLM ignores them and uses BF16 KV cache (default, recommended for maximum quality)
- **With `--kv-cache-dtype fp8`**: vLLM uses the calibrated scales for FP8 KV cache

**Important note on capacity**: On this hybrid architecture, FP8 KV cache does **not** significantly increase total KV cache capacity. The Gated DeltaNet state (48 layers) dominates memory usage and is not affected by `--kv-cache-dtype`. FP8 KV provides a modest decode speedup on the 16 attention layers, not more context length.

## Qwen3.8 Modes & Reverse Proxy

Qwen 3.8 natively supports several runtime modes (instruct, thinking, preserve thinking) and multiple reasoning effort levels. Activating them requires sending vendor-specific parameters such as `chat_template_kwargs`, which most standard OpenAI clients do not expose.

To solve this, we built a lightweight reverse proxy that exposes each official mode as a distinct virtual model name. Your client simply picks the model — the proxy automatically injects the correct `chat_template_kwargs`, sampling parameters, and reasoning effort:

- `qwen38-instruct` — Native instruct mode (no reasoning)
- `qwen38-thinking` — Thinking mode, `reasoning_effort` controllable by the client
- `qwen38-thinking-preserve` — Thinking mode with historical thinking preservation
- Optional pre-configured variants: `-low`, `-medium`, `-xhigh` for each thinking mode

View the project on our GitHub: [iguanesolutions/qwen38-rp](https://github.com/iguanesolutions/qwen38-rp)

## Inference

We run this model with vLLM, here is a sample execution command (from the official [vllm's recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)):

```bash
docker run --rm --name 'Qwen3.8' --gpus all --privileged --ipc=host -p 8000:8000 \
  -v /srv/cache:/root/.cache \
  vllm/vllm-openai:qwen38 ig1/Qwen3.8-27B-NVFP4-FP8KV-MTP \
  --served-model-name 'Qwen3.8-27B' \
  --max-model-len auto \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

> **Note**: The `vllm/vllm-openai:qwen38` image is a temporary build with Qwen3.8 support. Once the next official vLLM release is out, use the standard `vllm/vllm-openai:latest` (or the release tag) instead.

A few notes about some of the parameters:

- Adapt the `/srv/cache:/root/.cache` mount point to your liking. It contains files you want to keep between multiple runs (dynamo bytecode and AOT with torch compile but most importantly the huggingface folder for the model).
- `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder` are required for the reverse proxy's thinking modes and tool calls.
- `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` enables Multi-Token Prediction for lower latency. We recommend `2` over `3` based on acceptance rate measurements (~56% average draft acceptance with 3 tokens, better efficiency with 2).

### Optional: FP8 KV cache

To enable FP8 KV cache with the calibrated scales, add:

```bash
  --kv-cache-dtype fp8
```

Expected effects:
- ~10-20% decode speedup on attention layers
- No significant change in total KV cache capacity (see note above)
- Quality preserved thanks to calibrated scales

### Speculative Decoding (MTP)

The layers responsible for Multi-Token Prediction have not been quantized and are available separately in the `model_mtp.safetensors` file. vLLM loads them automatically when `--speculative-config` is provided.

This is recommended for latency-focused serving scenarios (not total throughput/massive concurrent requests).

## License

This model is a quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Please refer to the original model card for license and usage terms.
