---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- mlx
---

# Qwen3.8-27B-oQ8e-mtp

An 8-bit enhanced oQ quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
for Apple Silicon and MLX-compatible runtimes.

## Quantization details

The full calibration summary is available in
[`oq_imatrix_report.json`](./oq_imatrix_report.json).

## License
Apache 2.0, inherited from the base model.