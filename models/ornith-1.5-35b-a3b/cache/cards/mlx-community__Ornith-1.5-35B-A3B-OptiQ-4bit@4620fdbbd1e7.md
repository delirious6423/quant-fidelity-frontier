---
license: apache-2.0
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
language:
- en
tags:
- mlx
- optiq
- quantized
- 4bit
- mixed-precision
- qwen3.5
- moe
- mixture-of-experts
- reasoning
- multimodal
- vision
- image-text-to-text
- long-context
- mtp
- multi-token-prediction
- function-calling
- tool-use
- agentic
- apple-silicon
---

# mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit

> **Built with [mlx-optiq](https://mlx-optiq.com)**, the MLX-native toolkit to quantize, fine-tune, and serve LLMs locally on Apple Silicon, no PyTorch and no cloud. [All OptiQ quants](https://mlx-optiq.com/models) · [Docs](https://mlx-optiq.com/docs/)

OptiQ mixed-precision quant of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), a 35B mixture-of-experts vision-language model with 3B active parameters and a bundled MTP speculation head. 24 GB on disk.

## What it is

| Property | Value |
|---|---|
| Base | ornith-ai/Ornith-1.5-35B-A3B (Qwen3.5-MoE, 35B total / 3B active) |
| Method | OptiQ mixed-precision, per-layer 4/8-bit |
| Bit allocation | Reused from the [Ornith-1.0-35B OptiQ recipe](https://huggingface.co/mlx-community/Ornith-1.0-35B-OptiQ-4bit): the architecture is identical, so the per-layer sensitivity ranking transfers directly and no per-model sweep is needed |
| Layer split | 113 components at 4-bit, 399 at 8-bit |
| Group size | 64 |
| On disk | 24 GB |
| MTP | Speculation head preserved in `optiq/mtp.safetensors` |
| Vision | bf16 vision tower kept in `optiq/optiq_vision.safetensors` for image input |

Following the naming `llama.cpp` uses for its mixed quants, the "4bit" label denotes the family, not the weighted average.

## Run it

The MoE arch and the MTP/vision sidecars register through OptiQ, so `import optiq` once before loading:

```bash
pip install "mlx-optiq>=0.4.27"
```

```python
import optiq  # registers the arch + MTP/vision sidecars
from mlx_lm import load, generate

model, tok = load("mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit")
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Explain mixture-of-experts routing in two sentences."}],
    tokenize=False, add_generation_prompt=True,
)
print(generate(model, tok, prompt=prompt, max_tokens=400))
```

For image input plus an OpenAI- and Anthropic-compatible endpoint with mixed-precision KV cache:

```bash
optiq serve --model mlx-community/Ornith-1.5-35B-A3B-OptiQ-4bit
```

This is a reasoning model, so give it a generous token budget.

## Links

- **Project website:** [mlx-optiq.com](https://mlx-optiq.com/)
- **All OptiQ quants:** [mlx-optiq.com/models](https://mlx-optiq.com/models)
- **Base model:** [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)
