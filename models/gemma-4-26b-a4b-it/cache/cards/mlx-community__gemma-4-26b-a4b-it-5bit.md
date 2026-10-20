---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
tags:
- mlx
- safetensors
- gemma4
- image-text-to-text
- conversational
- 5-bit
base_model: google/gemma-4-26B-A4B-it
---
# mlx-community/gemma-4-26b-a4b-it-5bit

This model was converted to MLX format from [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) at revision `20da991ab4afab98e8f910c4a2e8f4fbefc404ad`.

It was generated from the current `mlx-vlm` source checkout.

## Use with mlx-vlm

```bash
pip install -U mlx-vlm
mlx_vlm.generate --model mlx-community/gemma-4-26b-a4b-it-5bit --max-tokens 100 --temperature 0.0 --prompt "Describe this image." --image <path_to_image>
```
