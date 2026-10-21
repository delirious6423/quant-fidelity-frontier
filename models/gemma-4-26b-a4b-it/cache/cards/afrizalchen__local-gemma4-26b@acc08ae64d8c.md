---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B-it
tags:
- mlx
---

# local-gemma4-26b

This model [local-gemma4-26b](https://huggingface.co/local-gemma4-26b) was
converted to MLX format from [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
using mlx-lm version **0.31.3**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("local-gemma4-26b")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
