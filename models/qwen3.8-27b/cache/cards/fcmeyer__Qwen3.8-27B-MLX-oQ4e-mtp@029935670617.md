---
library_name: mlx
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
- mlx
- oq
- quantized
- mtp
- omlx
- qwen3_5
---

# Qwen3.8-27B-MLX-oQ4e-mtp

**oQ4e** (enhanced, imatrix-calibrated ~4-bit) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with the **native MTP head preserved**, produced with [oMLX](https://github.com/jundot/omlx) 0.5.7's oQ quantizer from the bf16 conversion [fcmeyer/Qwen3.8-27B-MLX-bf16-mtp](https://huggingface.co/fcmeyer/Qwen3.8-27B-MLX-bf16-mtp).

- **~16 GB** (≈4.9 bpw effective; 4-bit base, group size 64, affine, sensitivity-boosted tensors at higher bits, bit-protected MTP head)
- **Enhanced (`e`)**: imatrix calibration (128 samples × 512 tokens, `oqe_code_multilingual`) with Hessian-guided error compensation
- **Full VLM**: vision tower included (image + video understanding)
- **MTP-preserved**: `mtp_num_hidden_layers: 1`, enabling Lightning-MTP speculative decoding in oMLX (`mtp_enabled: true`)

## Measured on an M5 Max (128 GB), oMLX 0.5.7

- **~54 tok/s decode with MTP on** (depth 3), **81% draft acceptance** (2.9 tok/backbone-cycle) — quantization did not degrade the draft head thanks to oQ's MTP bit protection.
- Image grounding verified (color/position). See the bf16 repo for the unquantized baseline (~20 tok/s with MTP).

## Usage with oMLX

Place under `~/.omlx/models/<org>/Qwen3.8-27B-MLX-oQ4e-mtp` (or download via the oMLX admin dashboard) and enable MTP:

```json
"Qwen3.8-27B-MLX-oQ4e-mtp": { "mtp_enabled": true, "max_context_window": 262144 }
```

Sampling defaults: temperature 1.0, top_p 0.95, top_k 20. The repo includes `oq_imatrix_report.json` documenting the calibration.

Also loadable as a plain quantized MLX VLM with mlx-vlm ≥ 0.6.3 (MTP tensors are ignored by loaders without MTP support).
