---
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: mlx
tags:
- mlx
- vision
pipeline_tag: image-text-to-text
---

# ToPo-ToPo/Ornith-1.5-35B-A3B-mlx-4bit

MLX **4bit** conversion of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for Apple Silicon (mlx-vlm), **with the vision tower kept**.

## Provenance (self-converted from official weights)
- Source: [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) (license: mit)
- Tool: `mlx-vlm 0.7.0` (mlx 0.32.2) — equivalent to `mlx_vlm.convert --hf-path ornith-ai/Ornith-1.5-35B-A3B --mlx-path . -q --q-bits 4 --q-group-size 64`
- Effective: **4.649 bits/weight**
- Converting with `mlx-vlm 0.6.3` (mlx 0.31.2) produced byte-identical weight files and configs.
- Vision: the source checkpoint is `Qwen3_5MoeForConditionalGeneration` and contains the vision tower (333 tensors) and the image/video processor configs. This conversion keeps them, so the model accepts image input. The official [`ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-MLX-4bit) is a text-only conversion without the vision tower.
- The MTP (multi-token prediction) weights of the source are not included.
- Validation: image input tested through local-llm-server (mlx-vlm backend). The model correctly identified the shapes, colors and positions in a synthetic test image, and described the material layout in a topology-optimization result image.

## Usage
```python
from mlx_vlm import load, generate
model, processor = load("ToPo-ToPo/Ornith-1.5-35B-A3B-mlx-4bit")
```
