---
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - basalt
  - macaq
  - pv-tuning
  - 4-bit
---

# goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX

4-bit affine **PV-MACaQ** (block-64 Walsh-Hadamard rotation + optimal `pre_scale` + clip/V-step polish) quantization of `Qwen/Qwen3.8-27B` (via `mlx-community/Qwen3.8-27B-8bit`), built for 24 GB Apple Silicon Macs in [Basalt](https://github.com/goodmai/basalt).

- **Size**: `16.06 GB` (4 shards)
- **Quantized modules**: `496` 4-bit affine PV-MACaQ g64 H64 backbone layers + `2` 4-bit affine g64 (`embed_tokens` / `lm_head`) + `686` BF16 vision/norms/conv1d
- **Mean SQNR (vs 8-bit)**: `20.61 dB` (RTN 4-bit) -> **`21.59 dB`** (PV-MACaQ 4-bit, MSE `-19.5%`)

```bash
basalt serve --model goodmai/Qwen3.8-27B-PV-MACaQ-4bit-Affine-MLX --kv-bits 4 --prompt-cache-mb 0 --wired-limit --rest
```
