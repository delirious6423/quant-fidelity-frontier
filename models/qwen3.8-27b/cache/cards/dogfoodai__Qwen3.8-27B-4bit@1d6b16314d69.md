---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - qwen
  - qwen3.8
  - text-generation-inference
---

# Qwen3.8-27B — MLX 4-bit (text-only)

4-bit quantized [MLX](https://github.com/ml-explore/mlx) conversion of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), built with
[mlx-lm](https://github.com/ml-explore/mlx-lm) **0.31.3**.

## Features (from the official model card)

- **Thinking mode on by default.** The model emits a `<think>…</think>` reasoning block before the
  final response. Disable per request with `enable_thinking=False` (instruct mode).
- **`reasoning_effort`** — `xhigh` (default), `medium`, or `low` to tune reasoning depth and cost.
- **`preserve_thinking`** — on by default; retains thinking blocks from all historical messages for
  agentic continuity and KV-cache efficiency.
- **Context: up to 1,000,000 tokens.** The model's native context window is 1M tokens.
- **Recommended sampling** — thinking mode: `temperature=1.0, top_p=0.95, top_k=20`; instruct mode:
  `temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5`.
- These are chat-template-level features: the original `chat_template.jinja` ships with this repo.

## What this conversion is

- **Text-only.** The original is a vision-language model (`Qwen3_5ForConditionalGeneration`); mlx-lm
  converts the language model only — the vision tower (`model.visual.*`) is stripped. Image/video
  inputs are NOT supported.
- **MTP preserved.** The multi-token prediction module (`mtp.*`, ~0.42B params) is kept as
  `mtp.safetensors` (bf16, unquantized). mlx-lm's `qwen3_5` runtime does not execute multi-token
  prediction yet, so the weights are inert under mlx-lm but remain available for runtimes that
  implement multi-token prediction.
- **Quantization:** 4-bit affine group quantization, group size 64.
  Weights ~14.9 GiB (original bf16 ≈ 54 GiB).

## Usage

```bash
pip install mlx-lm

# chat (thinking mode is on by default via the shipped chat template)
python -m mlx_lm generate --model dogfoodai/Qwen3.8-27B-4bit \
    --prompt "Explain the Gated DeltaNet architecture in one paragraph."
```

```python
from mlx_lm import load, generate

model, tokenizer = load("dogfoodai/Qwen3.8-27B-4bit")
messages = [{"role": "user", "content": "Write a haiku about quantization."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=128))

# instruct mode: skip the <think> block
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False
)
```

## Conversion

Reproduced with the script in this repo (`convert_to_mlx_4bit.py`), which wraps mlx-lm's `convert` API:

```bash
python3 convert_to_mlx_4bit.py \
    --hf-path Qwen/Qwen3.8-27B \
    --mlx-path mlx_model \
    --upload-repo dogfoodai/Qwen3.8-27B-4bit
```

Internally this runs the equivalent of:

```bash
mlx_lm.convert --hf-path Qwen/Qwen3.8-27B --mlx-path mlx_model -q --q-bits 4 --q-group-size 64
```

## Caveats

- Benchmarks on the original model card are for the unquantized bf16 model and were not re-run on
  this 4-bit build.
- The vision tower (`model.visual.*`) is not part of this text-only conversion; for image/video
  understanding, use the original [transformers](https://huggingface.co/Qwen/Qwen3.8-27B) or vLLM checkpoint.
