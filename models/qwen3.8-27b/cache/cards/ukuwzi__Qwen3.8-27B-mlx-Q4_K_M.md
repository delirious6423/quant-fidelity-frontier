---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - qwen
  - text-generation
  - apple-silicon
  - quantized
---

# ukuwzi/Qwen3.8-27B-mlx-Q4_K_M

This model [ukuwzi/Qwen3.8-27B-mlx-Q4_K_M](https://huggingface.co/ukuwzi/Qwen3.8-27B-mlx-Q4_K_M) is an MLX-format conversion of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

It was converted using **mlx-lm version 0.32.0** and uses **Q4_K_M quantization** for efficient inference on Apple Silicon devices.

## Installation

```bash
pip install mlx-lm
```

## Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("ukuwzi/Qwen3.8-27B-mlx-Q4_K_M")

prompt = "Explain how neural networks learn."

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict=False,
    )

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=512,
    verbose=True,
)

print(response)
```

## Chat Usage

```python
from mlx_lm import load, generate

model, tokenizer = load("ukuwzi/Qwen3.8-27B-mlx-Q4_K_M")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short poem about the ocean."},
]

prompt = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=False,
)

response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=256,
    temp=0.7,
    verbose=True,
)

print(response)
```

## Command-Line Usage

```bash
mlx_lm.generate \
  --model ukuwzi/Qwen3.8-27B-mlx-Q4_K_M \
  --prompt "Write a Python function that checks whether a number is prime."
```

## Model Details

- Base model: `Qwen/Qwen3.8-27B`
- Format: MLX
- Quantization: Q4_K_M
- Pipeline: Text generation
- Conversion tool: `mlx-lm`
- Conversion version: `0.32.0`
- Intended hardware: Apple Silicon devices

## License

This model card is distributed under the Apache 2.0 license. Please refer to the original [Qwen/Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the base model’s license, usage restrictions, and additional information.