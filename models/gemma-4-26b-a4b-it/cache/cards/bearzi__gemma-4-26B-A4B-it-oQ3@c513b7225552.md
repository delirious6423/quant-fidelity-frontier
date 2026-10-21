---
base_model: google/gemma-4-26B-A4B-it
library_name: mlx
pipeline_tag: text-generation
license: apache-2.0
tags:
- mlx
- omlx
- oq
- oq3
- quantized
---

# gemma-4-26B-A4B-it-oQ3

oQ3 mixed-precision MLX quantization produced via [oMLX](https://github.com/jundot/omlx).

- **Quantization:** oQ3 (sensitivity-driven mixed precision, group_size=64)
- **Format:** MLX safetensors
- **Compatible with:** mlx-lm, mlx-vlm, oMLX on Apple Silicon

## Usage

```python
from mlx_lm import load, generate
model, tokenizer = load("bearzi/gemma-4-26B-A4B-it-oQ3")
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    add_generation_prompt=True,
)
print(generate(model, tokenizer, prompt=prompt, max_tokens=512, verbose=True))
```

## About oQ

oQ measures per-layer quantization sensitivity through calibration and allocates bits where they matter most — critical layers stay at higher precision, tolerant layers compress aggressively. Target averages of 2/3/4/6/8 bits are provided; actual per-layer bits vary by measured sensitivity.

See [oQ documentation](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md).

Comparative benchmarks and feedback welcome — please open a discussion.
