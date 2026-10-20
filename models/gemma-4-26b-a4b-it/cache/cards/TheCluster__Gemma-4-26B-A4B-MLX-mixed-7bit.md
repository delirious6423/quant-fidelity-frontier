---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model:
- google/gemma-4-26B-A4B-it
library_name: mlx
tags:
- 7bit
- mixed-precision
pipeline_tag: image-text-to-text
language:
- en
- zh
- ru
- es
- fr
- it
- ja
- ko
- af
- de
- ar
- tr
- is
- pl
- sw
- sv
- nl
- he
- id
- uk
- fa
- pa
- pt
- ms
- fi
- el
---
<div align="center">
  <img src=https://ai.google.dev/gemma/images/gemma4_banner.png>
</div>

# Gemma-4-26B-A4B

**Quality**: quantized (***mixed quants per tensor**, group size: 32, **7.551** bpw*)

The various layers use 6-, 7-, or 8-bit affine quantization with a group size 32; embeddings are saved in bf16.

Gemma is a family of open models built by Google DeepMind. Gemma 4 models are multimodal, handling text and image input (with audio supported on small models) and generating text output. This release includes open-weights models in both pre-trained and instruction-tuned variants. Gemma 4 features a context window of up to 256K tokens and maintains multilingual support in over 140 languages.

-----
### Source
This model was converted to MLX format from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) using mlx-vlm version **0.4.4**.
