---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- mlx
base_model: Qwen/Qwen3.8-27B
---

# kenotron/Qwen3.8-27B-mlx-4Bit

The Model [kenotron/Qwen3.8-27B-mlx-4Bit](https://huggingface.co/kenotron/Qwen3.8-27B-mlx-4Bit) was converted to MLX format from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) using mlx-lm version **0.31.2**.

## Use with mlx

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("kenotron/Qwen3.8-27B-mlx-4Bit")

prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
