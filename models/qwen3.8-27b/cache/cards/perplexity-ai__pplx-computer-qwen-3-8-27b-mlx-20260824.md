---
license: other
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: mlx
pipeline_tag: image-text-to-text
tags:
- mlx
- mlx-vlm
- pplx-computer
- speculative-decoding
---

# pplx-computer-qwen-3-8-27b-mlx-20260824

MLX 4-bit build of the `qwen3.8_27b_rft_sdpo_10kdata_tools_88` RFT checkpoint
(Qwen3.8-27B, RFT/SDPO tools post-training) — the same weights as
`perplexity-ai/pplx-computer-qwen-3-8-27b-dflash2-20260824` (NVFP4), converted
here from the original **bf16** checkpoint, not from the NVFP4 quant.

- **Text + image input** (vision tower included), converted with `mlx-vlm` 0.7.6
- Quantization: affine 4-bit, group size 64 (4.695 bits/weight overall), ~15 GB
- `mtp/` — the checkpoint's native MTP head as a standalone speculative drafter
  (247 MB); output is identical with or without it

## Run (Apple Silicon)

```bash
pip install -U mlx-vlm
hf download perplexity-ai/pplx-computer-qwen-3-8-27b-mlx-20260824 \
  --local-dir ~/models/pplx-computer-qwen-3-8-27b-mlx-20260824
M=~/models/pplx-computer-qwen-3-8-27b-mlx-20260824

# text
mlx_vlm.generate --model $M --draft-model $M/mtp --max-tokens 1024 \
  --prompt "Write a haiku about search engines."

# image
mlx_vlm.generate --model $M --draft-model $M/mtp --max-tokens 1024 \
  --image photo.png --prompt "Describe this image."

# OpenAI-compatible server (/v1/chat/completions, image_url content parts)
mlx_vlm.server --model $M --draft-model $M/mtp --port 8080
```

Drop `--draft-model $M/mtp` to run without speculative decoding.

## Measured (MacBook Pro, 36 GB, greedy, 118-token code answer)

| Mode | Generation | Peak memory |
|---|---|---|
| plain | 22.4 tok/s | 16.4 GB |
| `--draft-model mtp` | **48.0 tok/s** (87% draft acceptance, 2.74 tokens/round) | 17.1 GB |
