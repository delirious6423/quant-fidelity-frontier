---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek41-dspark
- speculative-decoding
---

# DeepSeek V4.1 Flash 1017 — DSpark drafter (GGUF)

The DSpark speculative-decoding drafter of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(source revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, recorded in GGUF
metadata `general.source.revision`).

| File | Bytes | SHA-256 |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-DSpark.gguf` | 8,633,942,016 | `3347e357646a8b71e11b55248ca49263aa8aa281d69c18ff0c73301182ed6b05` |

Recipe: not re-quantized; routed experts keep their native MXFP4 codes and
scales; dense tensors are expanded losslessly to F16 (only when exact) or F32.

It pairs **only** with a V4.1 target from the same 1017 checkpoint revision:
[flaviodelgrosso/deepseek-v4.1-flash-1017-gguf](https://huggingface.co/flaviodelgrosso/deepseek-v4.1-flash-1017-gguf).

## Use

- [Inferno](https://github.com/flaviodelgrosso/inferno) on Apple Silicon:
  `inferno serve --model DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --placement ssd --speculation opportunistic --dspark DeepSeek-V4.1-Flash-1017-DSpark.gguf`
  (`--speculation exact` preserves the sampling distribution).

## License

MIT, as the upstream DeepSeek checkpoint (see `LICENSE`).
