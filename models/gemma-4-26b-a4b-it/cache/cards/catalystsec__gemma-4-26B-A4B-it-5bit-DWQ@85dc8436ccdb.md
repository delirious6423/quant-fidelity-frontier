---
language: en
library_name: mlx
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
tags:
- mlx
---

# catalystsec/gemma-4-26B-A4B-it-5bit-DWQ

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("catalystsec/gemma-4-26B-A4B-it-5bit-DWQ")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
