---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
license: mit
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - magnitude
  - glm
  - fp8
  - nvfp4
  - moe
---

# GLM-5.3-Flash GGUF

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) for [Magnitude](https://magnitude.dev), in two
variants, each lossless at its native size:

- **FP8:** Z.ai's release, the model exactly as trained.
- **NVFP4:** NVIDIA's 4-bit release, trained with quantization-aware distillation to hold the model's accuracy at 4
  bits, at 57% of FP8's size.

Both keep the weights' separate scale tensors as released. llama.cpp doesn't support separate scale tensors, so it
can't load these files; its own GGUFs of this model double the weights' size or round them.

## Variants

| Variant | Source | Size |
|---|---|---|
| FP8 | [`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash) at [`46a050b`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/46a050bec13002ad29ee795c16e61cf4fa3c4a9c) | 327.2 GB |
| NVFP4 | [`nvidia/GLM-5.3-Flash-FP4-Local`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) at [`76348dd`](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local/tree/76348ddb2be2e35e39ecfc00e75dc07703f4f3d5) | 185.8 GB |

Both include the MTP layer for speculative decoding, as the text model's last block. Each is converted from its
source and verified byte for byte against it.

## Files

```
FP8/GLM-5.3-Flash-FP8-00001-of-00007.gguf ... -00007-of-00007.gguf
NVFP4/GLM-5.3-Flash-NVFP4-00001-of-00004.gguf ... -00004-of-00004.gguf
mmproj-BF16.gguf     vision projector for FP8, as Z.ai ships it
mmproj-NVFP4.gguf    vision projector for NVFP4, as NVIDIA ships it (part of its vision tower is quantized)
```

Point Magnitude at the first shard of a variant, and at its projector for image input.

## Differences from llama.cpp GGUFs

llama.cpp's conversions of this model:
- **BF16** doubles the size, and still isn't exact: a code times its scale can need more precision than BF16 holds.
- **Q8_0, Q4_K and similar** keep the size small but round every weight onto a new grid.

| Variant | How the weights are stored here |
|---|---|
| FP8 | Projections and routed experts: `F8_E4M3` codes in `X.weight`, with one F32 scale per 128 x 128 tile in `X.scale` (Z.ai's `weight_scale_inv`). Everything else, output head included: `BF16` and `F32`. |
| NVFP4 | NVIDIA's 4-bit codes with their block and tensor scales, in Magnitude's NVFP4 type. Tensors NVIDIA kept at 8 bits stay in their 8-bit forms with their scales. Everything else: `BF16` and `F32`. |

Tensor names and metadata follow llama.cpp [v0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0)'s
`glm5-next` converter.

## License

MIT, as the source checkpoints.
