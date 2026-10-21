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

# Qwen3.8-27B-oQ6e-MTP

An Apple Silicon MLX quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced
directly from the original FP16 checkpoint with oMLX enhanced affine
quantization.

- Full multimodal vision-language model; the vision tower is preserved.
- The checkpoint's embedded Lightning MTP head is preserved.
- Baseline 6-bit affine weights with data-driven mixed-precision boosts to
  8-bit for 35 modules, group size 64.
- Activation-weighted clipping used 128 multilingual/code calibration samples
  of 512 tokens and 504 collected importance entries.
- Layer sensitivity was measured on a RAM-safe proxy rather than assigned
  uniformly.

## oMLX usage

```bash
hf download scaryrawr/Qwen3.8-27B-oQ6e-mtp \
  --local-dir ~/.models/scaryrawr/Qwen3.8-27B-oQ6e-mtp
omlx serve --model-dir ~/.models/scaryrawr
```

Enable `mtp_enabled` with `mtp_num_draft_tokens: 3` in the model's oMLX
settings. Text-only requests use native MTP draft/verify. Requests containing
images or video retain multimodal support and conservatively use standard
decode.

Six-bit affine weights use the standard MTP verify path because the specialized
oMLX verify QMM kernel supports affine 4-bit and 8-bit weights.

## Local validation

Measured on an M4 Max with 128 GB unified memory using adaptive MTP depth 3:

- Runtime model memory: 22.54 GB
- Warm predictable-code generation: 33.7 tokens/s
- Open-ended prose generation: 27.8 tokens/s

On focused deterministic prompts, this model exactly matched the FP16 model's
code, arithmetic, and structured-JSON outputs. Its generated LIS implementation
passed 207/207 correctness cases, and French instruction-following and
image-color checks also passed. These checks are useful smoke tests, not a
comprehensive benchmark suite.

## License

Apache 2.0, following the base model.
