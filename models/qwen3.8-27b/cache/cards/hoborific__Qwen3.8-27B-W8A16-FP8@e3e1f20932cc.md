---
base_model: Qwen/Qwen3.8-27B
library_name: transformers
tags:
- quantized
- w8a16
- fp8
---

# Qwen3.8-27B-W8A16-FP8

Quantized version of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

## Format

Offline-quantized **W8A16 FP8** in the
[compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
`float-quantized` format: weights in `float8_e4m3fn` with per-output-channel
symmetric scales, activations kept in bf16/fp16.

## How it was quantized

For each linear layer, every output row gets its own scale starting from
`amax / 448`, refined by an MSE clip search over ~9 clip fractions
(0.8–1.0× amax) picking the lowest-error scale per row. Weights are then
quantized `q = e4m3(w / scale)` with round-to-nearest and saturation. This
per-channel + clipping scheme gives better SNR than vLLM's online per-tensor
`--quantization fp8` path.

Only 2D linear projection weights are quantized (attention q/k/v/o, MLP
gate/up/down). Embeddings, norms, lm_head, routers/experts, and the vision
tower stay in bf16 and are listed in the checkpoint's `ignore` list, so vLLM
leaves them untouched.

## Supported vLLM platforms

- **Intel XPU** — `XPUW8A16FP8LinearKernel` (the intended target).
- **NVIDIA CUDA** (SM75+, i.e. Turing and newer) —
  `HummingFP8ScaledMMLinearKernel` when the `humming` package is installed,
  otherwise `MarlinFP8ScaledMMLinearKernel`.
- **Not supported**: ROCm, CPU, TPU — vLLM has no W8A16-FP8 kernel for these
  backends yet, so loading will fail with a "no kernel" error.
