---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
tags:
- mlx
---

# NexVeridian/Qwen3.8-27B-4bit

This model [NexVeridian/Qwen3.8-27B-4bit](https://huggingface.co/NexVeridian/Qwen3.8-27B-4bit) was
converted to MLX format from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
using mlx-lm version **0.32.0**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("NexVeridian/Qwen3.8-27B-4bit")

prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=False,
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
