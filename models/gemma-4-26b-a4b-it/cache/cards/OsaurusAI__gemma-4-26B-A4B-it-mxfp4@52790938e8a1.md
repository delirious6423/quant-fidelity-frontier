---
language:
- en
library_name: mlx
license: gemma
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
base_model: google/gemma-4-26B-A4B-it
tags:
- quantized
- apple-silicon
- mlx
- gemma4
- vision
- multimodal
- mxfp4
---

<p align="center">
  <a href="https://osaurus.ai"><img src="https://cdn-avatars.huggingface.co/v1/production/uploads/69d00705ce8872981c6c4fce/GWKjOwezSOhW5iuKpDwq_.png" alt="Osaurus AI" width="120"></a>
</p>

<h3 align="center">Gemma 4 26B-A4B-it &mdash; MXFP4 (MLX)</h3>
<p align="center">Microscaling FP4 quantization with verified vision tower weights</p>

<p align="center">
  <a href="https://osaurus.ai"><img src="https://img.shields.io/badge/Web-osaurus.ai-blue" alt="Website"></a>&nbsp;
  <a href="https://huggingface.co/OsaurusAI"><img src="https://img.shields.io/badge/HF-OsaurusAI-yellow?logo=huggingface" alt="OsaurusAI"></a>
</p>

---

## Model Details

| Property | Value |
|----------|-------|
| **Base Model** | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Parameters** | 26B total, 4B active (Mixture of Experts) |
| **Quantization** | MXFP4 (Microscaling FP4), mixed-precision |
| **Avg Bits/Weight** | 4.604 |
| **Model Size** | 14.8 GB |
| **Architecture** | Gemma 4 (text + vision) |
| **Context Length** | 128K tokens |
| **Vocabulary** | 262K tokens |

## Weight Verification

Every tensor in the vision tower was loaded and checked for `max(abs(tensor)) > 0`. **Zero broken weights found.**

| Component | Tensor Count | Status |
|-----------|-------------|--------|
| **Vision Tower** (SigLIP) | 355 | All non-zero |
| **Language Model** (MoE) | 1,135 | All non-zero |
| **Total** | **1,490** | **All verified** |

## MXFP4 Quantization

MXFP4 (Microscaling FP4) uses block-scaled 4-bit floating point values, offering better precision characteristics than standard affine INT4 quantization at similar model sizes. MLP gate/up/down projections are kept at 8-bit for quality.

## Usage

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("OsaurusAI/gemma-4-26B-A4B-it-mxfp4")

# Text
prompt = apply_chat_template(processor, model.config, "Write a haiku about cats.")
output = generate(model, processor, prompt, max_tokens=200)
print(output.text)

# Vision
prompt = apply_chat_template(processor, model.config, "Describe this image.", num_images=1)
output = generate(model, processor, prompt, image="photo.jpg", max_tokens=200)
print(output.text)
```

## Conversion

Converted from `google/gemma-4-26B-A4B-it` using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) v0.4.4:

```bash
mlx_vlm.convert --hf-path google/gemma-4-26B-A4B-it \
  --mlx-path gemma-4-26b-a4b-it-mxfp4 \
  -q --q-mode mxfp4 --dtype bfloat16
```
