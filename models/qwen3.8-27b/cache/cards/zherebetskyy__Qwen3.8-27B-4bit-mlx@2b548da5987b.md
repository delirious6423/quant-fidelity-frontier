---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: 
  - Qwen/Qwen3.8-27B
  - mlx-community/Qwen3.8-27B-bf16
tags:
  - mlx
  - vision
  - image-to-text
  - text-generation
  - 4-bit
  - 27b
num_parameters: 27780000000
---

# Qwen3.8-27B-4bit-mlx

This model was converted to MLX format from [`mlx-community/Qwen3.8-27B-bf16`](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16)
using mlx-vlm version **0.6.13**.
Refer to the [original model card](https://huggingface.co/mlx-community/Qwen3.8-27B-bf16) for more details on the model.

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model zherebetskyy/Qwen3.8-27B-4bit-mlx --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```


```python
from mlx_lm import load, generate

model, tokenizer = load("zherebetskyy/Qwen3.8-27B-4bit-mlx")

prompt="hello"

if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
```
