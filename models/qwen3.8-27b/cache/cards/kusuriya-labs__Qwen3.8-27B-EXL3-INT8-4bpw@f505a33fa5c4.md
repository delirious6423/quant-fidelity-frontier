---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exl3-int8
  - rocm
  - quantized
---

# Qwen3.8-27B - EXL3-INT8 4.0 bpw

EXL3-INT8 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). These are **not** standard Transformers /
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
| Language-model linears | EXL3-INT8 4.0 bpw |
| LM head / vision / MTP | EXL3-INT8 6 bit / 6 bit / 4 bit |
| Embeddings, norms, recurrent controls | unquantized |

Total size about 16 GB.

`config.json`, `quantization_config.json`, `model.safetensors.index.json` and the tokenizer / preprocessor files must stay
together. The original model card is preserved in `SOURCE_MODEL_CARD.md`; its claims describe the upstream model, not this
quantization. Calibration used 250 x 2048 tokens.

## Evaluation

KL divergence against BF16 (llama.cpp reference), 8 cases / 2,632 positions, measured with the ROCm build of ExLlamaV3:

| | mean KLD | top-1 agreement |
|---|---|---|
| standard EXL3 4 bpw (same quantizer) | 0.01927 | 0.9457 |
| EXL3-INT8 4 bpw, FP16 activations | 0.01973 | 0.9472 |
| EXL3-INT8 4 bpw, INT8 activations (fixed grid) | 0.02167 | 0.9403 |

On an R9700 (gfx1201), the `shelf` runtime prefill with hipBLASLt int8 GEMM was 7-15% faster than the standard-format
W16A16 path in earlier measurements (2048 tokens: 1112 ms, about 1840 tok/s). The ROCm ExLlamaV3 build decodes this
format natively on both R9700 and V620 (V620 decode about 22 tok/s, same as the standard format).

## Runtime

Native decoding is implemented for AMD `gfx1201` (RDNA4, e.g. R9700) and `gfx1030` (RDNA2, e.g. V620) in a patched ROCm
build of ExLlamaV3 and in the `shelf` runtime. **Neither is released yet**; there is no public runtime for this format at the
time of writing. The multiplier must match the build of the extension (`0xDCD12D`).

## License

Apache-2.0, as the base model.
