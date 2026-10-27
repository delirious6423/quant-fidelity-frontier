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

# LFM2.5-24B-A2B-RCO-4.0-streaming

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

model, tokenizer = load("hf://hermitdave/LFM2.5-24B-A2B-RCO-4.0-streaming")
```

## Performance

| Metric | Value |
|--------|-------|
| Model size | 13.00 GB |
| Bits per weight (target) | 4.0 |
| Bits per weight (actual) | mixed (4–6 bpw) |
| Mode | affine |
| PPL (wikitext2) | 13.262 (first 2048 tokens) |
| Throughput | 36.4 tok/s |
| Degradation vs BF16 | +19.51% |

## Streaming Error DB

This quant was built using the **streaming/layer-by-layer** error_db approach (`build_error_db_streaming.py`). The error_db was built layer-by-layer from BF16 shards:

1. Phase 1: Load BF16 model, wrap linear layers to capture activations, run calibration (50 × 512 tokens from Pride & Prejudice)
2. Phase 2: Load BF16 weights one tensor at a time from safetensors shards, measure Hessian-weighted quant error at bitwidths {2,3,4,6,8}
3. RCO knapsack allocation under 4.0 bpw budget
4. MLX affine quantization with expert stacking (`switch_mlp` 3D tensors)
5. Validation on wikitext2 test set (first 2048 tokens)

## A/B Comparison (4-bit RCO vs RTN)

| Model | PPL | TPS | Notes |
|-------|-----|-----|-------|
| **RCO 4.0 (streaming)** | **13.262** | **36.4** | This model |
| RCO 4.0 (non-streaming) | 13.262 | 36.1 | `hermitdave/LFM2.5-24B-A2B-RCO-4.0` |
| RTN 4.0 (uniform) | 13.83 | 36.4 | `hermitdave/LFM2.5-24B-A2B-RTN-4.0` |

**Key finding:** Streaming and non-streaming error_db produce **identical PPL** (13.262), confirming the layer-by-layer approach is correct.

## Quantization Details

This quant was produced with the [rco-mlx](https://github.com/your-org/rco-mlx) pipeline:
1. Layer-by-layer introspection of the BF16 checkpoint
2. Hessian-weighted error database built from calibration text
3. RCO knapsack allocation under the 4.0 bpw budget
4. MLX affine quantization with expert stacking (`switch_mlp` 3D tensors)
5. Validation on wikitext2 test set

## License

This model inherits the license of the base model. See [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) for details.
