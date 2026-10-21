---
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- mixed-precision
base_model: Qwen/Qwen3.8-27B
---
# Qwen3.8-27B - MLX Quantized (Mixed 3/6-bit)

This repository contains weights for **Qwen/Qwen3.8-27B** converted to [MLX](https://github.com/ml-explore/mlx) format using `mlx-vlm`.

## Quantization Details

- **Tool:** `mlx-vlm`
- **Base Model:** `Qwen/Qwen3.8-27B`
- **Quantization Recipe:** `--quant-predicate mixed_3_6`
- **Group Size:** 64

---

## Usage

Ensure you have `mlx-vlm` installed:

```bash
pip install -U mlx-vlm
```

### CLI Inference

```bash
mlx_vlm.generate \
  --model tejones36/qwen3.8-27B-mixed-3_6 \
  --image "/path/to/image.png" \
  --prompt "Describe this image in detail." \
  --max-tokens 512
```

### Python API

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model_path = tejones36/qwen3.8-27B-mixed-3_6"
model, processor = load(model_path)
config = load_config(model_path)

formatted_prompt = apply_chat_template(processor, config, "Describe this scene.", num_images=1)
output = generate(model, processor, prompt=formatted_prompt, image=["path/to/image.jpg"], max_tokens=512)
print(output)
```

