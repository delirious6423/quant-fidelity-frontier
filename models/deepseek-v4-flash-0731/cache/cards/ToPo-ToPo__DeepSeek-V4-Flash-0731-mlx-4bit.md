---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
library_name: mlx
tags:
- mlx
- moe
pipeline_tag: text-generation
---

# ToPo-ToPo/DeepSeek-V4-Flash-0731-mlx-4bit

MLX **4bit** conversion of [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
for Apple Silicon (mlx-vlm). 304B total sparse MoE (43 layers, 256 routed experts top-6 + 1 shared),
text-only, 1M context.

## Provenance (self-converted from official weights)
- Source: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
  (license: mit) — the official release ships **FP8** (`e4m3`, block 128×128), 166.9 GB
- Tool: `mlx-vlm 0.6.8` — `mlx_vlm.convert --hf-path deepseek-ai/DeepSeek-V4-Flash-0731 --mlx-path . -q --q-bits 4 --q-group-size 64`
- Effective: **4.339 bits/weight** (144 GiB on disk, ~154 GB peak RSS at inference)
- No config edits were needed: the official `model_type` is `deepseek_v4`, which `mlx_vlm.models.deepseek_v4`
  matches directly, and its `sanitize` resolves the FP8 `.weight` / `.scale` pairs during conversion.

## Usage

```python
from mlx_vlm import load, generate
model, processor = load("ToPo-ToPo/DeepSeek-V4-Flash-0731-mlx-4bit")
```

The first load reads 144 GiB, so allow several minutes and raise any client-side startup timeouts.

`reasoning_effort` (`"low"` / `"high"` / `"max"`) is supported through the chat template.

## Speculative decoding (MTP)

Use [`ToPo-ToPo/DeepSeek-V4-Flash-MTP-bf16`](https://huggingface.co/ToPo-ToPo/DeepSeek-V4-Flash-MTP-bf16)
as the drafter:

```bash
mlx_vlm.server --model ToPo-ToPo/DeepSeek-V4-Flash-0731-mlx-4bit \
               --draft-model ToPo-ToPo/DeepSeek-V4-Flash-MTP-bf16 --draft-kind mtp
```

Note that the MTP module **bundled in this 0731 checkpoint cannot be used** with mlx-vlm 0.6.8: 0731 carries
three MTP modules built around `main_proj` (`[4096, 12288]`, three concatenated projections), whereas
`DeepseekV4MTPDraftModel` implements the earlier `enorm` / `hnorm` / `e_proj` / `h_proj` layout. The drafter
linked above is split from the earlier `deepseek-ai/DeepSeek-V4-Flash` release, which matches that layout and
shares the same `hidden_size`. Speculative decoding is lossless, so outputs are unchanged either way.
