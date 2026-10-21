---
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: image-text-to-text
tags:
  - mlx
  - omlx
  - oq
  - oq6e
  - 6bit
  - mtp
  - vision
  - qwen3.8
---

# Qwen3.8-27B-oQ6e-mtp

**oQ6e** quantization of **Qwen/Qwen3.8-27B**, produced with **oMLX 0.6.0**.

## Quantization

- **Source:** `Qwen/Qwen3.8-27B` @ [`1d4bf0f`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- **Format:** oQ6e
- **Quantizer:** oMLX `0.6.0`
- **MTP:** embedded MTP weights preserved
- **Vision:** vision components preserved
- **Sensitivity model:** `mlx-community/Qwen3.8-27B-8bit` @ [`815b83c`](https://huggingface.co/mlx-community/Qwen3.8-27B-8bit/tree/815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9)
- **Sensitivity format:** 8-bit affine, group size 64