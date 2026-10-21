---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
tags:
- mlx
- mlx-vlm
- speculative-decoding
- mtp
---

# ToPo-ToPo/Qwen3.8-27B-MTP-4bit

MTP (multi-token prediction) **drafter** for speculative decoding with
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) on Apple Silicon (mlx-vlm).

This is **not** a standalone chat model — it only works bound to a Qwen3.8-27B target model.

## Provenance
- Split from the official checkpoint's built-in `mtp.*` weights (15 tensors, isolated in
  `model-00018-of-00018.safetensors`) with mlx-vlm 0.6.9:
  `python -m mlx_vlm.speculative.drafters.qwen3_5_mtp.split --model Qwen/Qwen3.8-27B --output .`
- Precision: affine 4bit / group 64 — 4.501 bits/weight
- Size: 247 MB, `block_size: 3`, `model_type: qwen3_5_mtp`

## Speculative decoding notes

Gains are largest on structured output (code, JSON tool calls) and smallest on free-form prose.
Stable over 589 rounds with rejections — no rollback crash.

The bf16 and 4bit drafters accept identically under greedy decoding (quantization rarely flips
the argmax); bf16 was marginally better on code, 4bit uses less memory.

**Measure it yourself.** Speedup depends on hardware, target quantization, workload and the
mlx-vlm version — on the machine used here the reference pair
`Qwen3.6-27B-4bit` + `mlx-community/Qwen3.6-27B-MTP-4bit` showed no speedup at all, so any
number quoted elsewhere may not transfer to your setup.

## Usage
```bash
mlx_vlm.generate --model ToPo-ToPo/Qwen3.8-27B-mlx-4bit \
  --draft-model ToPo-ToPo/Qwen3.8-27B-MTP-4bit --draft-kind mtp \
  --prompt "..." --max-tokens 400
```
