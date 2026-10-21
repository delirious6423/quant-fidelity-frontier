---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
---

# erobey/Qwen3.8-27B-mlx-4bit

This model [erobey/Qwen3.8-27B-mlx-4bit](https://huggingface.co/erobey/Qwen3.8-27B-mlx-4bit) was
converted to MLX format from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
using mlx-lm version **0.31.3**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("erobey/Qwen3.8-27B-mlx-4bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
