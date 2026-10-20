---
library_name: mlx
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
language:
- en
tags:
- mlx
- optiq
- quantized
- mixed-precision
- static
- moe
- mixture-of-experts
- 4bit
- 8bit
- apple-silicon
- image-text-to-text
- vision-language
- vision
- multimodal
- qwen3.5
- reasoning
- long-context
- mtp
- multi-token-prediction
- function-calling
- tool-use
- agentic
---

# Ornith-1.5-35B-A3B-OptiQ-static-5.5bpw

> Built with [mlx-optiq](https://mlx-optiq.com), the MLX-native toolkit to quantize, fine-tune, and serve LLMs locally on Apple Silicon. [Docs](https://mlx-optiq.com/docs/) · [All OptiQ quants](https://mlx-optiq.com/models)

A static mixed-precision MLX quant of [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), the mid-size Ornith 1.5 mixture-of-experts vision-language model (35B total, ~3B active per token, 256 experts, 8 routed). Produced with `optiq convert --method static --target-bpw 5.5`.

**This checkpoint takes images and text.** The language tower is mixed-precision affine MLX. The vision tower is kept at **bf16** in `optiq/optiq_vision.safetensors`. Stock `mlx-lm` loads the language tower only. Image input needs `mlx-optiq`.

Sizes below are decimal GB, matching what the Hub sidebar reports.

## Source

| Property | Value |
|---|---|
| Base model | [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| Relation | quantized (Hub `base_model_relation`) |
| Architecture | Qwen3.5 MoE VLM (`Qwen3_5MoeForConditionalGeneration`) |
| License | MIT, same as the base |

This is a static convert, not a KL-measured OptiQ allocation, and there is no 5-bit grid in the file. Bit-widths come from architecture rules, not from a sensitivity pass on Ornith 1.5.

## Quantization details

| Property | Value |
|---|---|
| Method | OptiQ `static` — structural per-layer allocation, no calibration |
| Convert target | 5.5 BPW |
| Achieved BPW (language tensors) | 5.50 |
| Predominant precision | still 4-bit by parameter count (62% of language parameters) |
| Tensors at 8-bit | 431 |
| Tensors at 4-bit | 79 |
| Tensors at 6-bit | 1 leftover, knapsack remainder to hit the BPW target |
| Total quantized tensors | 511 |
| Group size | 64 |
| Mode | affine |
| Language tower on disk | 26.16 GB |
| Vision tower | bf16, 333 tensors, 0.89 GB, `optiq/optiq_vision.safetensors` |
| MTP head | int4 gs64, 0.49 GB, `optiq/mtp.safetensors` |
| Whole artifact | 27.56 GB |

The name uses **5.5bpw** (the convert target), not "5-bit". Extra bits went into more routed-expert layers at 8-bit. Attention, router, shared expert, embeddings, `lm_head`, and the bf16 vision tower stay at the static high-bit recipe.

Reproduce:

```bash
pip install "mlx-optiq>=0.4.27"
optiq convert ornith-ai/Ornith-1.5-35B-A3B --method static --target-bpw 5.5
```

## Image input

The vision tower is **not quantized**. 4-bit vision tends to hurt OCR and fine detail, so OptiQ leaves it at bf16. Processor configs (`preprocessor_config.json`, `processor_config.json`, `video_preprocessor_config.json`) are copied from the base so the Hub artifact is a complete image-text-to-text package.

Image path requires [`mlx-optiq`](https://mlx-optiq.com/) **0.4.27 or newer** (sidecars live under `optiq/`, not the repo root).

```bash
pip install "mlx-optiq>=0.4.27"
```

```python
from PIL import Image
from optiq.runtime.engine import OptiqEngine

engine = OptiqEngine("ahmedihamdy/Ornith-1.5-35B-A3B-OptiQ-static-5.5bpw")
answer = engine.generate(
    "What is in this image?",
    images=[Image.open("photo.jpg")],
    max_tokens=512,
)
print(answer.text)
```

OpenAI- and Anthropic-compatible serving with vision:

```bash
optiq serve --model ahmedihamdy/Ornith-1.5-35B-A3B-OptiQ-static-5.5bpw
```

## Text

A stock `*.safetensors` glob ignores `optiq/`, so `mlx-lm` sees a clean language model. Register the MoE VLM arch and sidecars first:

```bash
pip install mlx-lm "mlx-optiq>=0.4.27"
```

```python
import optiq  # registers the arch + MTP/vision sidecars
from mlx_lm import load, generate

model, tokenizer = load("ahmedihamdy/Ornith-1.5-35B-A3B-OptiQ-static-5.5bpw")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Explain the difference between TCP and UDP."}],
    add_generation_prompt=True,
    tokenize=False,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=512))
```

This is a reasoning model: it thinks before answering, so give `max_tokens` room to finish.

### Speculative decoding (MTP)

This quant bundles a Multi-Token Prediction head at `optiq/mtp.safetensors`.

```bash
optiq serve --model ahmedihamdy/Ornith-1.5-35B-A3B-OptiQ-static-5.5bpw --mtp
```

## What this is not

- Not a KL-measured OptiQ allocation on Ornith 1.5.
- Not a 5-bit quant. Almost every tensor is 4-bit or 8-bit.
- Not a quantized vision tower. `mlx-lm` text-only loads will not see images.
- No public MMLU / GSM8K / SWE-bench numbers for this file.

Quantization does not change the behaviour or alignment of the base model. Use it under the same terms as [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).
