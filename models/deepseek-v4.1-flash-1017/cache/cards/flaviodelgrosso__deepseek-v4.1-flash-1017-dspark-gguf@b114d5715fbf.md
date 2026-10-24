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
at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, as a standalone
`deepseek41-dspark` support GGUF. Weights are not re-quantized: routed experts
keep their native MXFP4 codes and scales; dense tensors are expanded losslessly
to F16 (only when exact) or F32.

It pairs **only** with a V4.1 target converted from the same 1017 checkpoint
revision, for example
[flaviodelgrosso/deepseek-v4.1-flash-1017-gguf](https://huggingface.co/flaviodelgrosso/deepseek-v4.1-flash-1017-gguf).
A drafter from another checkpoint (e.g. the pre-1017 `df42c109`) is not compatible.

| File | Bytes | SHA-256 |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-DSpark.gguf` | 8,633,942,016 | `3347e357646a8b71e11b55248ca49263aa8aa281d69c18ff0c73301182ed6b05` |

GGUF metadata: `general.architecture = deepseek41-dspark`,
`dspark.target_architecture = deepseek41`, `general.source.revision =
1bae0e1088993b9b82de08f59de5be5a34e42170`, target layers 37–39, block 5, Markov
rank 256, 128 routed experts with top-3 routing, noise token 128799.

## Provenance

Converted offline with a lossless DSpark extraction script
(`deepseek41_dspark.py`):

```sh
python3 gguf-tools/deepseek41_dspark.py \
  --hf models/DeepSeek-V4.1-Flash-1017 \
  --source-revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --out DeepSeek-V4.1-Flash-1017-DSpark.gguf
```

Inputs (only the shards holding `mtp.*` tensors are needed):

| File | SHA-256 |
|---|---|
| `model-00044-of-00048.safetensors` | `3e8ac315c2bcca3e83c5e6e25c5758a75c03732063a6ff0c1b0e24b9ff49f6e6` |
| `model-00045-of-00048.safetensors` | `9c73b27c4343236790008979d5e0b565f687aa64440031b9e096fed1ab8039c5` |
| `model-00046-of-00048.safetensors` | `1dccc19a3741972c661de734572dc7d4b899e6e99bfbdf3ae4e057c707e5b339` |
| `config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `model.safetensors.index.json` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |

`deepseek41_dspark.py --verify` passes on all 81 tensors, and a second
conversion reproduces the same SHA-256.

## Use

- [Inferno](https://github.com/flaviodelgrosso/inferno):
  `inferno serve --model DeepSeek-V4.1-Flash-1017-IQ2_XXS-Q2_K-imatrix.gguf --placement ssd --speculation opportunistic --dspark DeepSeek-V4.1-Flash-1017-DSpark.gguf`
  (`--speculation exact` preserves the sampling distribution).

## License

MIT, as the upstream DeepSeek checkpoint (see `LICENSE`).
