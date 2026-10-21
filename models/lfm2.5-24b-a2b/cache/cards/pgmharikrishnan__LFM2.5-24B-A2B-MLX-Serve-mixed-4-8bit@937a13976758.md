---
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
library_name: mlx-serve
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- mlx
- mlx-serve
- lfm2_moe
- moe
- apple-silicon
- text-only
---

# LFM2.5-24B-A2B for mlx-serve (mixed 4-bit / 8-bit)

An affine MLX quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B). It carries the source repository's LFM 1.0 license unchanged.

This is a **text-only** model. Its native architecture is `lfm2_moe` (`Lfm2MoeForCausalLM`), which is supported natively by **mlx-serve v26.10.2 and later**; it is not a Qwen-engine conversion.

## Benchmark results

Recorded comparison with `LiquidAI/LFM2.5-24B-A2B-MLX-4bit`:

| Metric | Stock 4-bit | Mixed 4/8-bit | Delta |
|---|---:|---:|---:|
| Plain decode | 85.68 tok/s | **189.75 tok/s** | **+121.5%** (2.21×) |
| Plain prefill | 409.71 tok/s | **733.09 tok/s** | **+78.9%** |
| Best decode | 180.45 tok/s | **184.95 tok/s** | **+2.5%** |
| Best prefill | 482.32 tok/s | **460.79 tok/s** | -4.5% |

### Long-context ladder

| Rung | Stock prefill / decode | Mixed prefill / decode | Needle |
|---|---|---|---|
| 4K | 1222 tok/s / 128 tok/s | **2644 tok/s / 143 tok/s** | Passed / Passed |
| 16K | 1263 tok/s / 84 tok/s | **2441 tok/s / 104 tok/s** | Passed / Passed |
| 32K | 1082 tok/s / 67 tok/s | **2156 tok/s / 79 tok/s** | Passed / Passed |
| 64K | 866 tok/s / 31 tok/s | **1674 tok/s / 52 tok/s** | Passed / Passed |
| 128K | — | — | **Both packs failed at the 131,072-token ceiling.** |

## Quantization

- Routed MoE experts: affine **4-bit**, group size 64.
- Trunk projections (attention, convolution, and dense MLP layers): affine **8-bit**, group size 64.
- Token embeddings: affine **6-bit**, group size 64.
- Router gates and norms: BF16; expert-selection bias: float32.

## Usage

Requires **mlx-serve v26.10.2 or later**.

```bash
mlx-serve serve --model pgmharikrishnan/LFM2.5-24B-A2B-MLX-Serve-mixed-4-8bit
```
