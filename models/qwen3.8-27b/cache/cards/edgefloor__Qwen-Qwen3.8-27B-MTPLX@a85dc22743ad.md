---
base_model: Qwen/Qwen3.8-27B
library_name: mlx
pipeline_tag: text-generation
tags:
  - mlx
  - apple-silicon
  - mtplx
  - multi-token-prediction
  - qwen
---

# Qwen-Qwen3.8-27B-MTPLX

MTPLX-branded multi-token-prediction model for Apple Silicon (MLX).
Forged with [MTPLX Forge](https://github.com/youssofal/MTPLX) from
`Qwen/Qwen3.8-27B`.

## Verification

- Best depth: D3
- Multiplier vs autoregressive baseline: 2.24×
- Verified on: Apple M4 Pro
- Sampler: temperature 0.6 · top_p 0.95 · top_k 20

See `mtplx_runtime.json` for the full verification record.

## Usage

```bash
# MTPLX picks this model up automatically when downloaded:
mtplx pull edgefloor/Qwen-Qwen3.8-27B-MTPLX
mtplx start chat
```

## License

See LICENSE.