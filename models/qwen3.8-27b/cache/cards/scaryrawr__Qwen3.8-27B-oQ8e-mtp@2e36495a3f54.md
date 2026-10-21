---
base_model: Qwen/Qwen3.8-27B
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
  - mlx
  - omlx
  - qwen3.8
  - multimodal
  - quantized
  - mtp
---

# Qwen3.8-27B-oQ8e-MTP

An Apple Silicon MLX quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced
directly from the original FP16 checkpoint with oMLX enhanced affine
quantization.

- Full multimodal vision-language model; the vision tower is preserved.
- The checkpoint's embedded Lightning MTP head is preserved.
- 8-bit affine weights with group size 64.
- Activation-weighted clipping used 128 multilingual/code calibration samples
  of 512 tokens and 504 collected importance entries.
- Layer sensitivity was measured on a RAM-safe proxy rather than assigned
  uniformly.

## oMLX usage

```bash
hf download scaryrawr/Qwen3.8-27B-oQ8e-mtp \
  --local-dir ~/.models/scaryrawr/Qwen3.8-27B-oQ8e-mtp
omlx serve --model-dir ~/.models/scaryrawr
```

Enable `mtp_enabled` with `mtp_num_draft_tokens: 3` in the model's oMLX
settings. Text-only requests use native MTP draft/verify. Requests containing
images or video retain multimodal support and conservatively use standard
decode.

The affine 8-bit path can use oMLX's guarded custom MTP verify QMM kernel when
eligible; unsupported shapes automatically fall back to the standard path.

## Local validation

Measured on an M4 Max with 128 GB unified memory using adaptive MTP depth 3:

- Runtime model memory: 28.37 GB
- Warm predictable-code generation: 30.1 tokens/s
- Open-ended prose generation: 24.5 tokens/s

The generated LIS implementation passed 207/207 correctness cases. Focused
checks also passed arithmetic, strict structured JSON, French instruction
following, and image-color recognition. This is the highest-fidelity practical
affine variant in this set. These checks are useful smoke tests, not a
comprehensive benchmark suite.

## License

Apache 2.0, following the base model.
