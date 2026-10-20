---
license: other
license_name: lfm1.0
license_link: https://www.liquid.ai/lfm-license
library_name: mlx
pipeline_tag: text-generation
base_model: LiquidAI/LFM2.5-24B-A2B
tags:
- mlx
- optiq
- quantized
- 4bit
- mixed-precision
- lfm2
- moe
- liquid-ai
- apple-silicon
- on-device
---

# mlx-community/LFM2.5-24B-A2B-OptiQ-4bit

> **Built with [mlx-optiq](https://mlx-optiq.com)**, the MLX-native toolkit to quantize, fine-tune, and serve LLMs locally on Apple Silicon, no PyTorch and no cloud. [All OptiQ quants](https://mlx-optiq.com/models) · [Docs](https://mlx-optiq.com/docs/) · [LFM2.5 family](https://mlx-optiq.com/docs/lfm2.5)

An OptiQ mixed-precision quant of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), the largest model in the LFM2.5 family. It is 16.7 GB on disk, down from 47.9 GB at bf16, with a 128k context.

LFM2.5-24B-A2B is a sparse mixture of experts on Liquid's hybrid backbone. Convolutional blocks alternate with full attention, so only 10 of its 40 layers keep a KV cache. Each token is routed to 4 of 64 experts. OptiQ measures each layer's sensitivity and assigns its bit-width from that.

## What it is

| Property | Value |
|---|---|
| Base | [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| Architecture | `lfm2_moe`: hybrid conv + attention, 40 layers (10 attention), 64 experts, 4 active |
| Method | OptiQ mixed-precision, sensitivity-driven (uniform-4-bit reference) |
| On disk | 16.7 GB (bf16: 47.9 GB) |
| Context | 128k |

Most of the routed expert weights sit at 4-bit: 87 of the 114 fused expert projections. Attention, the routers and nearly all of the convolution blocks stay at 8-bit.

## Capability Score

Six-metric mean, the standard OptiQ eval, run in reasoning mode because the model always thinks before it answers. The baseline is a uniform 4-bit quant of the same model, scored the same way.

| Metric | OptiQ | Uniform 4-bit |
|---|---:|---:|
| MMLU (generative) | 81.1% | 80.3% |
| GSM8K | 94.4% | 93.2% |
| IFEval (strict) | 88.0% | 84.5% |
| BFCL-V3 simple (200 calls) | 91.5% | 89.5% |
| HumanEval (164 problems, pass@1) | 91.5% | 67.1% |
| HashHop (long-context retrieval) | 1.0% | 0.0% |
| **Capability Score** (mean of 6) | **74.59** | **69.10** |

The OptiQ quant is ahead on every metric. The big gap is HumanEval: the uniform quant often writes a correct function and then keeps re-checking it until it runs out of tokens, even with a 16k reasoning budget. A reply that never finishes counts as a failure. HashHop is the weak spot for both, because the model copies the lookup table into its reasoning and runs out of budget before it answers.

## Run it

```bash
pip install mlx-optiq
optiq serve --model mlx-community/LFM2.5-24B-A2B-OptiQ-4bit
```

That gives you an OpenAI and Anthropic compatible endpoint with mixed-precision KV cache (`kv_config.json` in this repo), tool calling, and prompt caching.

The model reasons before every answer, so give it a generous `max_tokens`. Liquid's recommended sampling (`temperature 0.1`, `top_k 50`, `repetition_penalty 1.05`) ships in `generation_config.json`, and `optiq serve` applies it automatically.

Or from Python:

```python
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/LFM2.5-24B-A2B-OptiQ-4bit")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Summarise this contract clause in two sentences."}],
    add_generation_prompt=True, tokenize=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=4096))
```

## Links

- **Project website:** [mlx-optiq.com](https://mlx-optiq.com/)
- **All OptiQ quants:** [mlx-optiq.com/models](https://mlx-optiq.com/models)
- **PyPI:** [pypi.org/project/mlx-optiq](https://pypi.org/project/mlx-optiq/)
- **Base model:** [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
