---
library_name: transformers
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
- mlx
---

# shawon/gemma-4-26B-A4B-it-mlx-8Bit

The Model [shawon/gemma-4-26B-A4B-it-mlx-8Bit](https://huggingface.co/shawon/gemma-4-26B-A4B-it-mlx-8Bit) was converted to MLX format from [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) using mlx-lm version **0.31.2**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("shawon/gemma-4-26B-A4B-it-mlx-8Bit")

prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
