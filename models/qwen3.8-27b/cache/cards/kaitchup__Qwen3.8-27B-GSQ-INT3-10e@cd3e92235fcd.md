---
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- gsq
- quantized
- compressed-tensors
---

# Qwen3.8-27B-GSQ-INT3-10e

These are research artifacts, not intended for practical use. I made these models to test GSQ with different hyperparameters. I trained them with limited resources, and they significantly underperform the original model, but they are not broken.

| Hyperparameter | Value |
| --- | --- |
| Base model | Qwen/Qwen3.8-27B |
| Method | GSQ |
| Weight dtype / groups | INT3 (group 128) |
| Epochs per layer | 10 |
| LM head / embeddings | INT8 / INT8 (group 64) |
| Retained floating dtype | bfloat16 |
| Training logits dtype | float32 |
| Logits / scale learning rate | 0.0001 / 5e-05 |
| Calibration mixture | reasoning-balanced |
| Calibration / GPTQ / reconstruction samples | 1024 / 512 / 64 |
| Sequence length | 4096 |
| Batch / microbatch | 16 / 1 |
| Seed | 42 |
| Format | Safetensors; compressed-tensors (pack-quantized) |
| MTP / vision | Original BF16 MTP preserved; vision excluded |
