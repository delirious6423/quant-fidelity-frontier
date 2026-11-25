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

# Qwen3.8-Flash-Next - EXL3-INT8 3.05 bpw

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
| Language-model linears, including all routed experts | EXL3-INT8 3.05 bpw |
| LM head | EXL3-INT8 5 bit |
| Vision / MTP | EXL3-INT8 5 bit / 3 bit |
| N-gram (PLE) embedding table (`ngram_embedding.safetensors`, 32.6 GB) | standard EXL3 trellis, not INT8; decoded to rows, no GEMM |
| Embeddings, norms, hyper-connection and recurrent controls | unquantized |

Total size about 80 GB (the n-gram table is 32.6 GB of it).

`config.json`, `quantization_config.json`, `model.safetensors.index.json` and the tokenizer / preprocessor files must stay
together. The original model card is preserved in `SOURCE_MODEL_CARD.md`; its claims describe the upstream model, not this
quantization. Calibration used 250 x 2048 tokens.

## Evaluation

Measured with the ROCm build of ExLlamaV3, 2x V620 (gfx1030), layer split, n-gram table in host RAM.

**3.05 bpw only.** KL divergence against BF16 logits, 21 cases (32,359 positions, chunk 256, teacher forced):
macro KLD 0.1530, top-1 agreement 0.8688. For reference, a separately quantized standard-EXL3 3.05 bpw model of the same
model scored 0.1314 / 0.8999 with the same harness; the two are different quantization runs, so the gap mixes the
codebook effect with run-to-run quantization noise. One text (`wiki2_r7`) is a clear outlier for the INT8 grid
(0.601 vs 0.158); its cause is not analyzed.
Speed with this FP16 path: prefill 258 / 412 / 389 tok/s at 512 / 2048 / 8192 tokens, decode 25 tok/s, identical to the standard format.

On two R9700s, shelf revision `882b2078` ran this 3.05 bpw checkpoint with EP=2, TP=1, no host expert offload, `octave:c5v5` KV and a 2,048-token input. Across three runs, median prefill was **1,516 tok/s** and ordinary greedy decode was **36.64 tok/s** (128 new tokens). With the separate **Flash-Next** EXL3-INT8 4 bpw DFlash2 drafter and seven drafts, decode reached **87.56 tok/s** (84.21–90.05). Generated IDs could vary between runs. This historical benchmark predates the PLE loading and row-order fixes below and does not identify the PLE table used. [shelf benchmark and settings](https://github.com/kusuriya-labs/shelf/blob/1fb31fd963cc4c71210aa0c9c461679af150be71/docs/perf/current.md)

## Runtime

Shelf requires a supported PLE table for this checkpoint. Download the [companion Ngram4 PLE table](https://huggingface.co/kusuriya-labs/Qwen3.8-Flash-Next-PLE-Ngram4) and put its `ngram_manifest.json` and all `table_*.safetensors` in the model directory. The manifest takes precedence over the bundled standard EXL3 n-gram table, which shelf cannot read. Main commit [`1fb31fd9`](https://github.com/kusuriya-labs/shelf/commit/1fb31fd963cc4c71210aa0c9c461679af150be71) validates the table at load time and fixes row order for multi-position input.

The [public shelf runtime](https://github.com/kusuriya-labs/shelf) supports grouped EXL3-INT8 MoE experts on AMD RDNA4. This support was merged in [shelf PR #14](https://github.com/kusuriya-labs/shelf/pull/14). See the [multi-GPU guide](https://github.com/kusuriya-labs/shelf/blob/1fb31fd963cc4c71210aa0c9c461679af150be71/docs/user/multi-gpu.md) for expert placement and supported parallel layouts. The original V620 tests used a patched ROCm build of ExLlamaV3 with the `0xDCD12D` multiplier.

## License

Qwen Community License 1.0 (`LICENSE`), as the base model.
