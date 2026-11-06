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

[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) in GGUF for
[Magnitude](https://magnitude.dev), at the precision DeepSeek ships it: every weight is the checkpoint's own bytes,
checked byte for byte after conversion.

Converted from revision
[`1bae0e1`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170)
with tensor names and metadata from llama.cpp [v0.9.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0)'s
`deepseek41` converter.

## Format

| Weights | Stored as |
|---|---|
| Routed experts | `MXFP4` |
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
