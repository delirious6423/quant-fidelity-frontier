---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: mlx
tags:
- mlx
- oq
- quantized
- omlx
- mtp
- speculative-decoding
- apple-silicon
- refusal-ablated
---

# Ornith-1.5-35B-A3B-refusal-ablated-MLX-oQ8-mtp

This model was quantized using [oQ](https://github.com/jundot/omlx) (oMLX v0.6.3rc3) mixed-precision quantization.

Refusal-ablated **Ornith-1.5-35B-A3B** with the **official Ornith MTP head grafted in at 8-bit**, so Lightning-MTP speculative decoding works. The refusal-ablated derivative ships without an MTP head; this build restores it from the upstream `ornith-ai/Ornith-1.5-35B-A3B` checkpoint and quantizes it in the same oQ8 pass as the body.

## Quantization details

- **Model type**: qwen3_5_moe
- **Bits**: 8
- **Group size**: 64
- **Format**: MLX safetensors
- **MTP head**: official Ornith, quantized at 8-bit with the body

## Sampling

```
temperature = 0.6
top_p       = 0.95
top_k       = 20
```

## Performance — Apple M1 Ultra (64-core GPU)

| Context | Prefill (tok/s) | Decode (tok/s) |
| ------- | --------------: | -------------: |
| 1K      |           1,670 |           81.8 |
| 4K      |           2,084 |           76.1 |
| 8K      |           1,965 |           85.1 |
| 16K     |           1,738 |           90.2 |
| 32K     |           1,402 |           79.2 |


## Note

Base weights were modified by directional ablation to suppress refusal behaviour. This model will produce harmful, illegal, or dangerously incorrect content more readily than the upstream instruction model. You own the moderation layer the original guardrails carried, and remain responsible for the base model's licence and applicable law.