---
language: en
license: mit
pipeline_tag: image-text-to-text
tags:
- mlx
library_name: mlx
base_model: ornith-ai/Ornith-1.5-35B-A3B
---

This model was converted to MLX format and quantized from [Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) using [oMLX](https://github.com/jundot/omlx).

>[!CAUTION]
>This model was quantized using oQ (oMLX v0.6.3rc2) mixed-precision quantization.


## What is "oQ"?

See ["oQ: oMLX Universal Dynamic Quantization"](https://github.com/jundot/omlx/blob/main/docs/oQ_Quantization.md) for details.

## Quantizations

| Vision-Language  | 
|------------------|
| [MLX-VL-oQ8][01] |
| [MLX-VL-oQ6][02] |

## What is "VL"?

"VL" is Vision-Language, meaning quantization preserves the original model's multimodality.


[01]: https://huggingface.co/liqinghhua/Ornith-1.5-35B-MLX-VL-oQ8e
[02]: https://huggingface.co/liqinghhua/Ornith-1.5-35B-MLX-VL-oQ6e