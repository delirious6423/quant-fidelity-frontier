---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
pipeline_tag: text-generation
tags:
- liquid
- lfm2.5
- edge
- moe
- mlx-rco
- quantized
base_model: LiquidAI/LFM2.5-24B-A2B
---

# LFM2.5-24B-A2B-RCO-4.0

Mixed-precision MLX quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) using **RCO (Rate-Distortion Optimization) allocation with Hessian-weighted error database**, built via **streaming/layer-by-layer** error_db generation.

## Model

- **Base:** LiquidAI/LFM2.5-24B-A2B
- **Parameters:** 24B total, 2.45B active
- **Architecture:** LFM2.5-MoE (40 layers: 30 conv + 10 full-attention, 64 experts top-4)
- **Context:** 131,072 tokens
- **Quantization:** RCO mixed-precision affine, group_size=64, target 4.0 bpw

## Usage

```python
from mlx_lm import load

model, tokenizer = load("hf://hermitdave/LFM2.5-24B-A2B-RCO-4.0")
```

## Performance

| Metric | Value |
|--------|-------|
| Model size | 14.00 GB |
| Bits per weight (target) | 4.0 |
| Bits per weight (actual) | mixed (4–6 bpw) |
| Mode | affine |
| PPL (wikitext2) | 13.364 (first 2048 tokens) |
| Throughput | 67.1 tok/s |
| Degradation vs BF16 | +20.73% |

## Quantization Details

This quant was produced with the [rco-mlx](https://github.com/your-org/rco-mlx) pipeline:
1. Layer-by-layer introspection of the BF16 checkpoint
2. Hessian-weighted error database built from calibration text (Pride & Prejudice, 50 × 512 tokens)
3. RCO knapsack allocation under the 4.0 bpw budget
4. MLX affine quantization with expert stacking (`switch_mlp` 3D tensors)
5. Validation on wikitext2 test set (first 2048 tokens)

## A/B Comparison (4-bit RCO vs RTN)

| Model | PPL | TPS | Notes |
|-------|-----|-----|-------|
| **RCO 4.0 (float32 scales)** | **13.364** | **67.1** | This model |
| RCO 4.0 (float16 scales) | 13.262 | 35.5 | `hermitdave/LFM2.5-24B-A2B-RCO-4.0-streaming` |
| RTN 4.0 (uniform) | 13.83 | 36.4 | `hermitdave/LFM2.5-24B-A2B-RTN-4.0` |

**Key finding:** float32 scales/biases give **~2× faster throughput** with slightly worse PPL (+0.77%) compared to float16.

## Notes

- Scales and biases are stored in **float32** for optimal dequantization precision and throughput on Apple Silicon
- Router gate (`feed_forward.gate`) is kept in BF16 — it's not quantized
- MoE expert layers use `switch_mlp` projection with 64 experts stacked per layer

## License

This model inherits the license of the base model. See [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for details.
