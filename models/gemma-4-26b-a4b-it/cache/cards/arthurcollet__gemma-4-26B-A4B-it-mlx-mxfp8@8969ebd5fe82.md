---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
- mlx
---

# arthurcollet/gemma-4-26B-A4B-it-mlx-mxfp8

This model was converted to MLX format from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it)
using mlx-vlm version **0.4.5**.
Refer to the [original model card](https://huggingface.co/google/gemma-4-26B-A4B-it) for more details on the model.

## Use with mlx

```bash
pip install -U mlx-vlm
```

```bash
python -m mlx_vlm.generate --model arthurcollet/gemma-4-26B-A4B-it-mlx-mxfp8 --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
