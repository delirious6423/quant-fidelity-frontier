---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- mxfp6
- mxfp8
- fp8
---

# Qwen3.8-27B-MXFP6

This is a calibration-free quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It is sized to fit a single 32 GB GPU (RTX 5090) with vision and MTP speculative decoding.

| Layers | Quantization |
|---|---|
| Dense MLP | MXFP6 weight, MXFP8 activation |
| Attention (full + GDN) and MTP projections | MXFP8 weight, MXFP8 activation |
| LM Head | Per-channel FP8 weight, per-token dynamic FP8 activation |
| Input embedding, vision tower, GDN small tensors, norms, MTP input projection | BF16, unquantized |

The checkpoint is 24.30 GiB in total.

`quantize.py` rebuilds this exact checkpoint from the BF16 model.

```bash
python quantize.py --out Qwen3.8-27B-MXFP6
```
