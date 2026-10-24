---
license: mit
base_model:
- deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- gguf
- deepseek41
- text-generation
---

# DeepSeek V4.1 Flash 1017 — IQ2_XXS/Q2_K imatrix (GGUF)

The 1017 Q2 quantization of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(source revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, recorded in GGUF
metadata `general.source.revision`).

| File | Bytes | SHA-256 |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf` | 365,713,686,528 | `5296e85ff64b42ecb62c8d0c0c54e1caa07e3df8e93112f2365ebbaed62ce93a` |

Recipe: IQ2_XXS routed gate/up + Q2_K routed down (2304 inputs); Q8_0
attention, shared experts and head; native FP8 Engram tables (layers 1, 14).
The imatrix is recorded in `quantize.imatrix.sha256`
(`75fadebe6d743fe7f751a5e5cdc13d198e2f3aae2504afeaf7f6acd24cce052f`).

The matching speculative-decoding drafter is
[flaviodelgrosso/deepseek-v4.1-flash-1017-dspark-gguf](https://huggingface.co/flaviodelgrosso/deepseek-v4.1-flash-1017-dspark-gguf).

## Use

- [Inferno](https://github.com/flaviodelgrosso/inferno) on Apple Silicon:
  `inferno serve --model DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --placement ssd --expert-cache-gib 32`
  (SSD expert streaming on 128 GiB Macs; resident needs about 192 GB).

## License

MIT, as the upstream DeepSeek checkpoint (see `LICENSE`).
