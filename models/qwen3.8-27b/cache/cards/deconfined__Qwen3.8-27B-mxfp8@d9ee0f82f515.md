---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
tags:
- mlx
base_model: Qwen/Qwen3.8-27B
---

# connstruct/Qwen3.8-27B-mxfp8

This model [connstruct/Qwen3.8-27B-mxfp8](https://huggingface.co/connstruct/Qwen3.8-27B-mxfp8) was
converted to MLX format from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
using mlx-lm version **0.31.3**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("connstruct/Qwen3.8-27B-mxfp8")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
