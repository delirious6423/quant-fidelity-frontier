---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exl3-int8
  - rocm
  - quantized
---

# Qwen3.8-Flash-Next - EXL3-INT8 4.0 bpw

EXL3-INT8 quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). These are **not** standard Transformers /
vLLM weights, and not loadable by upstream ExLlamaV3.

## What "EXL3-INT8" is

EXL3-INT8 is [ExLlamaV3](https://github.com/turboderp-org/exllamav3)'s EXL3 trellis format with the codebook changed so
that every decoded weight lies on one fixed INT8 grid. The trellis state is multiplied by `0xDCD12D` (a 24-bit constant, one
`v_mul_u32_u24`), the four bytes of the product are summed, and `q = (B - 509) >> 2` is the INT8 weight; the real value is
`q * 4 * 887 * 2^-17`. Decoded weights therefore feed `iu8` dot-product / WMMA instructions directly, with INT8 activations
(per-128 group scale, fp32 epilogue with the original `svh`).

* It is **not** convertible from a standard EXL3 checkpoint: the codebook is part of the quantization, so this model was
  quantized with it (same quantizer, calibration, and bit allocation as the standard format).
* The multiplier is stored in each linear's `.mul1` tensor (the key name is inherited from EXL3; its value here is `0x00DCD12D`
  instead of the standard `0x83DCD12D`). `quantization_config.json` says `"codebook": "exl3-int8"`.
* **Upstream ExLlamaV3 cannot run this model.** It would decode the weights with the standard multiplier and produce garbage.

## Contents

| Component | Precision |
|---|---|
| Language-model linears, including all routed experts | EXL3-INT8 4.0 bpw |
| LM head | EXL3-INT8 6 bit |
| Vision / MTP | EXL3-INT8 6 bit / 4 bit |
| N-gram (PLE) embedding table (`ngram_embedding.safetensors`, 32.6 GB) | standard EXL3 trellis, not INT8; decoded to rows, no GEMM |
| Embeddings, norms, hyper-connection and recurrent controls | unquantized |

Total size about 95 GB (the n-gram table is 32.6 GB of it).

`config.json`, `quantization_config.json`, `model.safetensors.index.json` and the tokenizer / preprocessor files must stay
together. The original model card is preserved in `SOURCE_MODEL_CARD.md`; its claims describe the upstream model, not this
quantization. Calibration used 250 x 2048 tokens.

## Evaluation

This 4 bpw checkpoint exceeded the available VRAM in the original two-V620, layer-split test with the n-gram table in host RAM. This run ended before quality or speed could be measured. For measurements on the smaller checkpoint, see the [3.05 bpw model card](https://huggingface.co/kusuriya-labs/Qwen3.8-Flash-Next-EXL3-INT8-3.05bpw).

## Runtime

Shelf requires a supported PLE table for this checkpoint. Download the [companion Ngram4 PLE table](https://huggingface.co/kusuriya-labs/Qwen3.8-Flash-Next-PLE-Ngram4) and put its `ngram_manifest.json` and all `table_*.safetensors` in the model directory. The manifest takes precedence over the bundled standard EXL3 n-gram table, which shelf cannot read. Main commit [`1fb31fd9`](https://github.com/kusuriya-labs/shelf/commit/1fb31fd963cc4c71210aa0c9c461679af150be71) validates the table at load time and fixes row order for multi-position input.

The [public shelf runtime](https://github.com/kusuriya-labs/shelf) supports grouped EXL3-INT8 MoE experts on AMD RDNA4. This support was merged in [shelf PR #14](https://github.com/kusuriya-labs/shelf/pull/14). See the [multi-GPU guide](https://github.com/kusuriya-labs/shelf/blob/1fb31fd963cc4c71210aa0c9c461679af150be71/docs/user/multi-gpu.md) for expert placement and supported parallel layouts. The original V620 tests used a patched ROCm build of ExLlamaV3 with the `0xDCD12D` multiplier.

## License

Qwen Community License 1.0 (`LICENSE`), as the base model.
