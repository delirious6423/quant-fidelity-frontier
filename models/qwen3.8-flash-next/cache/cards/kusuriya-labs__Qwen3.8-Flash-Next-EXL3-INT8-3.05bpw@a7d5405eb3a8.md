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

**4 bpw: not evaluated** (it does not fit two 32 GB V620s).

The INT8 compute path (INT8 activations) was not evaluated on this model. The `shelf` runtime's MoE path cannot run an
EXL3-INT8 expert (no grouped INT8 kernel), so on `shelf` this model is not usable; the ROCm ExLlamaV3 build runs it.

## Runtime

Native decoding is implemented for AMD `gfx1201` (RDNA4, e.g. R9700) and `gfx1030` (RDNA2, e.g. V620) in a patched ROCm
build of ExLlamaV3 and in the `shelf` runtime. **Neither is released yet**; there is no public runtime for this format at the
time of writing. The multiplier must match the build of the extension (`0xDCD12D`).

## License

Qwen Community License 1.0 (`LICENSE`), as the base model.
