---
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- oqe
- imatrix
- quantized
- mtp
base_model: Qwen/Qwen3.8-27B
---

# Qwen3.8-27B-oQ6e-mtp

This model was quantized using [oQe](https://github.com/jundot/omlx) (oMLX v0.6.0) imatrix-enhanced mixed-precision quantization.

The model is converted from [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) and includes multi-token prediction (MTP) support.
Refer to the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B) for more details on the model.

## Quantization details
- **Model type**: qwen3_5
- **Bits**: 6
- **Group size**: 64
- **Enhanced (imatrix)**: yes
- **Format**: MLX safetensors
- **MTP**: included
