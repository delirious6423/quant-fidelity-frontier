---
library_name: mlx
license: mit
license_link: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B/blob/main/LICENSE
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- omlx
- quantized
- fp16
- moe
- multimodal
- apple-silicon
---

# Ornith-1.5-35B-A3B-oQ4e-fp16

[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) quantized with [oQ](https://github.com/jundot/omlx) (oQ4e, imatrix-based mixed precision) in **oMLX 0.7.1**, saved with **fp16** scales instead of bf16 (M1/M2 GPUs have no native bf16).

Vision and the MTP head are preserved. Size: 21 GB.

## Quantization details

- **Method**: oQ4e (enhanced, imatrix-based); calibration set `oqe_code_multilingual` (128 samples × 512 tokens)
- **Bits**: routed experts 4-bit; shared experts, `embed_tokens` and `lm_head` 8-bit; attention and linear-attention projections 5-8 bit
- **Group size**: 64, affine
- **Dtype**: float16
- **Final weights**: quantized from the bf16 source
- **Calibration / sensitivity proxy**: uniform 4-bit proxy of the source (the bf16 model does not fit in 32 GB). The proxy was memory-mapped from disk instead of loaded into RAM, so a 4-bit proxy could be used on a 32 GB machine instead of falling back to 3-bit.
- **Tooling**: oMLX 0.7.1.dev1 (pre-release), MLX 0.32.3
- **Chat template**: unchanged from the base model
- **MTP**: included (42 tensors). On M2 it made MoE decode slower in our tests, so we run with MTP off.

`oq_imatrix_report.json` is included (523 entries, no zero-count experts).

## Speed

M2 Max, 32 GB, oMLX 0.7.1.dev1, TurboQuant off, ANE prefill off, MTP off, thinking off, single request, fused routed-expert decode engaged. 200 generated tokens.

| Prompt | Prefill | Decode |
|---|---|---|
| 1k | 992 tok/s | 89 tok/s |
| 19k | 681 tok/s | 68 tok/s |
| 35k | 471 tok/s | 57 tok/s |
| 68k | 342 tok/s | 42 tok/s |

For comparison, a uniform 4-bit fp16 conversion of the same model decodes at 109 / 75 / 63 / 44 tok/s. The gap comes from the 8-bit shared experts and `lm_head`, which are read for every token.

Notes:
- These numbers were measured with the weights memory-mapped from disk (an experimental loader, not part of oMLX). Decode speed with the stock loader should be similar, but the whole model is then resident (~21 GB).
- On M2, ANE prefill made this model slower (−14% to −26% prefill), and TurboQuant KV cost 10-19% decode.
- On M3 and newer, which have native bf16, expect little or no gain over a bf16 build.

## Usage

Tested with oMLX only. Put the folder in `~/.omlx/models/` and load `Ornith-1.5-35B-A3B-oQ4e-fp16`. It is a standard MLX safetensors checkpoint (`qwen3_5_moe` with vision), so other MLX runtimes that support it should load it, but that has not been tested. Keep the vision weights: oMLX's fast MoE decode kernel only engages on the vision-language path.

## License

MIT, same as the [base model](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).
