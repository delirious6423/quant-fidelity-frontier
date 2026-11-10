---
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
license: mit
library_name: gguf
pipeline_tag: image-text-to-text
tags:
  - gguf
  - magnitude
  - deepseek
  - mxfp4
  - fp8
  - moe
---

# DeepSeek-V4.1-Flash-1017 GGUF

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) for
[Magnitude](https://magnitude.dev), lossless at its native size. llama.cpp GGUFs of this model either double the size
of its FP8 weights or round them. This one keeps them exactly as DeepSeek released them, so it does not run in
llama.cpp.

Converted from revision
[`1bae0e1`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
and verified byte for byte. Tensor names and metadata follow llama.cpp
[v0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0)'s `deepseek41` converter.

## Differences from llama.cpp GGUFs

DeepSeek's dense and Engram weights are FP8: an 8-bit code per weight, times a scale shared by its block. Storing them
exactly at 8 bits takes both the codes and the scales. llama.cpp does not currently have a type for that, so its GGUFs
use one of two:
- **BF16:** exact, but twice the size.
- **Q8_0:** native size, but it rounds every weight onto its own grid.

This GGUF stores the FP8 codes unchanged in `F8_E4M3`, plus each block's scale in a separate tensor. Magnitude applies
the scales as it reads. llama.cpp does not currently support the scale tensors, so it does not load these files.

## Format

| Weights | Stored as |
|---|---|
| Routed experts | `MXFP4`, as shipped |
| Dense projections | `F8_E4M3` in `X.weight`, with UE8M0 scales per 32 x 32 tile in `X.scale` (`I8`) |
| Engram tables | `F8_E4M3` in `per_layer_token_embd.weight`, with UE8M0 scales per 32 in `per_layer_token_embd.scale` (`I8`) |
| Everything else | `BF16` and `F32`, as shipped |

`magnitude.scale_encoding = "ue8m0"` marks the `I8` scale tensors as UE8M0 exponents (value `2^(b - 127)`).

## Files

| File | Contents | Size |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-MXFP4_MOE-0000N-of-00008.gguf` | Text model | 501.4 GB (8 shards) |
| `DeepSeek-V4.1-Flash-1017-DSpark.gguf` | DSpark speculative drafter | 7.9 GB |
| `mmproj-DeepSeek-V4.1-Flash-1017.gguf` | Vision projector | 1.0 GB |

## License

MIT, as the source checkpoint.
