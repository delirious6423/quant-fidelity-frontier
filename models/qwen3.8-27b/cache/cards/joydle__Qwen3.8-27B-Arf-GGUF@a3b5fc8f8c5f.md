---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
tags:
  - gguf
  - arf
  - qwen3.8
  - group-64
  - imatrix
---

# Qwen3.8-27B · Arf group-64 GGUF

The weights [Arf](https://github.com/joydle/Arf) serves Qwen3.8-27B from on Apple silicon: a uniform
**4-bit group-64** quantization (MLX's affine layout, stored as exact Q4_1 pairs in a GGUF), fitted with
**importance weighting** from unsloth's imatrix. Arf's group-64 Metal kernels read it natively; the file is a
standard GGUF (llama.cpp reads the Q4_1 tensors too, but it is built for Arf).

| file | size | what |
|---|---:|---|
| `Qwen3.8-27B-arf-g64-imat-Q4_1.gguf` | 15.9 GB | the model (tokenizer and chat template embedded) |

Use it with the DFlash-2 draft ([incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2))
and, for images, unsloth's `mmproj-BF16.gguf` — `arf pull qwen3.8:27b` fetches all three.

## How it was made

`scripts/g64/quantize_g64.py` in the Arf repository: llama.cpp's Q4_1-with-imatrix recipe
(`quantize_row_q4_1_impl` + `make_qkx3_quants`) applied per 64-weight group, from
[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)'s `Qwen3.8-27B-Q8_0.gguf`, weighted by
its `imatrix_unsloth.gguf`, written into a copy of their Q4_K_M as the template. Reproduce it locally with
`scripts/get_qwen38.sh`.

## Quality (measured, teacher-forced perplexity through Arf's serving path, window 16)

| text (tokens) | Q4_K_M (unsloth) | this file | MLX 4-bit (mlx-community) |
|---|---:|---:|---:|
| Arf's DISCIPLINE.md (10,421) | 6.0642 | **6.0474** | 6.2111 |
| Arf's DEAD_ENDS.md, first 45K chars (13,982) | 7.5101 | **7.5197** | — |

Q4_K_M's quality in a uniform 4-bit layout that decodes faster on Apple GPUs. Every number traces to a dated
row in Arf's `docs/MEASURED_NUMBERS.md`.

## License

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Calibration data:
unsloth's imatrix (Apache-2.0).
