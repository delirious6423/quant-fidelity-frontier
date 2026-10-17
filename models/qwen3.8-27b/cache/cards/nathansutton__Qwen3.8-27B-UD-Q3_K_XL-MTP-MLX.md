---
library_name: mlx
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
- apple-silicon
- coding
- unsloth-style-dynamic-quant
pipeline_tag: text-generation
---

# Qwen3.8-27B-UD-Q3_K_XL-MTP-MLX

An [MLX](https://github.com/ml-explore/mlx) quantization of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for running locally on Apple Silicon —
the default model for [chad](https://github.com/nathansutton/chad), a Claude-Code-style
local coding agent.

## Quant

- **Params:** 27B (dense qwen3_5 hybrid: 64 layers, 48 GatedDeltaNet + 16 full attention)
- **Scheme:** 3-bit group-64 body (MLP gate/up/down + attention) and 3-bit embeddings, with lm_head held at 5-bit
- **Footprint:** ~12.1 GB weights + a bundled 4-bit MTP head; 262k native context
- **RAM:** Built for 24 GB Apple Silicon. chad sizes its context window from the live Metal budget, so expect tens of thousands of tokens rather than the full 262k on a 24 GB box.

The naming follows [Unsloth's Dynamic 2.0](https://docs.unsloth.ai/) convention (`UD-Q3_K_XL` = a dynamic quant that spends extra bits on the layers that matter) so the *scheme* is recognizable at a glance. **It is not literally a llama.cpp k-quant** — this is an MLX group-64 affine quant produced by our own per-module predicate (the bulk low-bit, sensitive layers high), with the bit allocation chosen by measured held-out perplexity per gigabyte across candidate recipes.

## Multi-token prediction head

`mtp.safetensors` is the checkpoint's own trained MTP head, quantized to 4-bit
group-64 and bundled alongside the weights. chad loads it as a sidecar and uses it
for **self-speculative decoding**: draft several tokens with the head, verify them in
one batched forward, accept by exact rejection sampling — so greedy output is
token-identical to the unspeculated path and sampled output keeps the model's true
distribution at any temperature.

The filename matters. `mlx-lm` globs `model*.safetensors`, so `mtp.safetensors` is
invisible to the base loader — which is the point. Named `model-mtp.safetensors` it
would be picked up by the base loader, `has_mtp_weights` would go true in
`qwen3_5.TextModel.sanitize`, and every norm weight would get a second +1.0 shift:
silent corruption, no error.

## Chat template

`chat_template.jinja` differs from upstream in exactly one line: the default
`reasoning_effort` is **`medium`**, where upstream defaults to `xhigh`. Pass
`reasoning_effort` explicitly to override it.

## Use it

With chad (auto-downloads this model on first run):

```bash
uvx chad-code
```

Or directly with `mlx-lm`:

```python
from mlx_lm import load, generate
model, tok = load("nathansutton/Qwen3.8-27B-UD-Q3_K_XL-MTP-MLX")
print(generate(model, tok, "Write a haiku about quantization.", max_tokens=64))
```

## License

Inherits the license of the base model
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). Review it before use or redistribution.
