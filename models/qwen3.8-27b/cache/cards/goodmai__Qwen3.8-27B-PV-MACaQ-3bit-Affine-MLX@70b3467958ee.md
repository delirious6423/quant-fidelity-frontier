---
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - basalt
  - macaq
  - pv-tuning
  - 3-bit
---

# goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX

3-bit affine **PV-MACaQ** (block-64 Walsh-Hadamard rotation + optimal `pre_scale` + clip/V-step polish) quantization of `Qwen/Qwen3.8-27B` (via `mlx-community/Qwen3.8-27B-8bit`), built for fast 24 GB Apple Silicon Macs in [Basalt](https://github.com/goodmai/basalt).

- **Size**: `13.02 GB` (3 shards)
- **Quantized modules**: `496` 3-bit affine PV-MACaQ g64 H64 backbone layers + `2` 4-bit affine g64 (`embed_tokens` / `lm_head`) + `686` BF16 vision/norms/conv1d
- **Mean SQNR (vs 8-bit)**: `13.98 dB` (RTN 3-bit) -> **`15.54 dB`** (PV-MACaQ 3-bit, MSE `-29.5%`)

```bash
basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-3bit-Affine-MLX --speculative-lookup --kv-bits 4 --rest
```
