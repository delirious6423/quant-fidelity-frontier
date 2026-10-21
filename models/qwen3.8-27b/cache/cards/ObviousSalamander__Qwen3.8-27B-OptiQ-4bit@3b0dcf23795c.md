---
language:
  - multilingual
tags:
  - mlx
  - qwen
  - quantized
  - optiq
  - vision
  - multimodal
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B-OptiQ-4bit (MLX, mixed-precision)

A mixed-precision 4.93 bpw quantized MLX build of `Qwen/Qwen3.8-27B`, produced with
[OptiQ](https://github.com/prncc/optiq) (data-driven sensitivity-based quantization),
packaged in the standard MLX layout with the vision tower restored into the main shards.

## What this is

- **Trunk:** `Qwen/Qwen3.8-27B` (bf16, 55.6 GB) — a *dense* multimodal model
  (`Qwen3_5ForConditionalGeneration`, vision + language).
- **Method:** OptiQ 0.4.22, `optiq convert --target-bpw 4.75 --candidate-bits 4,8`.
  OptiQ ran a per-layer KL-sensitivity analysis and chose the bit width per layer.
- **Result:** **4.93 bpw**, 249× 8-bit + 247× 4-bit layers. ~18.3 GB of weights
  (this repo ≈ 19 GB on disk).
- **Layout:** standard MLX — `model-*.safetensors` + `model.safetensors.index.json`.
  Vision tower is included as `model.visual.*` in the shards.

> ⚠️ **This is NOT a uniform 4-bit model.** The "4bit" in the name refers to the target
> budget (4.75 bpw); OptiQ protected ~half the layers at 8-bit. Effective size is between
> the community 4-bit and 8-bit packs. Do not compare it directly to a flat 4-bit pack.

## Why this build

OptiQ preserves the layers whose quantization hurts quality the most (embeddings, early/mid
attention q/k/v, parts of MLP) at 8-bit, keeping the rest at 4-bit. On a dense model this gives
a better quality-per-byte ratio than uniform 4-bit, at the cost of a slightly larger file.

## Why this was uploaded

This build took a long time to produce (download of the 55.6 GB bf16 original, the OptiQ
conversion, and re-merging the vision tower into a standard MLX layout). It is published so
that anyone can use the result directly instead of repeating that work.

## Benchmarks (mlx-vlm, Apple Silicon)

Measured on a **Mac mini M4 Pro (64 GB)** with `mlx-vlm` 0.6.13 (MTP-4bit drafter),
sustained decode, `max_tokens=256`, across 3 task types × 3 context lengths (2026-08-17).
Thinking and non-thinking decode throughput are equivalent, so the table below shows
the non-thinking runs.

| type     | ctx | mode        | tok/s  | prefill t/s |
|----------|-----|-------------|--------|-------------|
| creative | 2k  | non_think   | 17.58  | ~116        |
| creative | 8k  | non_think   | 17.84  | ~125        |
| creative | 16k | non_think   | 17.27  | ~106        |
| program  | 2k  | non_think   | 20.83  | ~116        |
| program  | 8k  | non_think   | 19.79  | ~125        |
| program  | 16k| non_think   | 18.57  | ~106        |
| decision | 2k  | non_think   | 20.13  | ~116        |
| decision | 8k  | non_think   | 18.86  | ~125        |
| decision | 16k| non_think   | 13.54  | ~106        |

**Summary:** ~17–21 tok/s decode, throughput stays stable out to 16k context — this build
matches the community 4-bit pack in speed while keeping higher fidelity (more 8-bit layers).
Prefill ~106–127 t/s (reading a 16k context ≈ 130–150 s, true prefill, no cache).

## Usage (text + vision)

### Text
```python
import mlx_lm
model, tokenizer = mlx_lm.load("ObviousSalamander/Qwen3.8-27B-OptiQ-4bit")
out = mlx_lm.generate(model, tokenizer, prompt="Hello")
```

### Vision (image)
This is a **multimodal** model — image + text works. Use `mlx_vlm` and format the prompt with
`apply_chat_template` (the same path the official `mlx_vlm.generate` CLI uses):

```bash
python -m mlx_vlm.generate \
  --model ObviousSalamander/Qwen3.8-27B-OptiQ-4bit \
  --prompt "Describe this image in 2-3 short sentences." \
  --image path/to/image.jpg
```

Python API:
```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

model, processor = load("ObviousSalamander/Qwen3.8-27B-OptiQ-4bit")
config = load_config("ObviousSalamander/Qwen3.8-27B-OptiQ-4bit")
prompt = apply_chat_template(processor, config, "Describe this image.", num_images=1)
out = generate(model, processor, prompt, ["path/to/image.jpg"])
```

> ⚠️ Do not pass a hand-written `<|image_pad|>` token into `mlx_vlm.generate` directly —
> it returns an empty output (immediate `<|im_end|>`). Always use `apply_chat_template`
> with `num_images=1`, exactly as the official CLI does.

## Usage with mtplx (native MTP speculative decoding)
Forge-probe/build this repo with `mtplx forge`, then serve it and use the OpenAI-compatible API
for text generation (see the mtplx docs). On the same M4 Pro hardware, OptiQ mtplx reaches
~13–27 tok/s engine decode at depth 3 (strongest on programming: 27 t/s at 2k/8k context).

## How it was built (reproducibility)

```bash
# 1. download the original bf16
huggingface-cli download Qwen/Qwen3.8-27B --local-dir Qwen3.8-27B-bf16-orig

# 2. convert with OptiQ 0.4.22 (data-driven mixed precision)
optiq convert Qwen3.8-27B-bf16-orig \
  --target-bpw 4.75 --candidate-bits 4,8 --reference auto \
  -o Qwen3.8-27B-OptiQ-4bit

# 3. the output contains optiq_mixed/ (this model) + intermediate artifacts
#    (static_mixed/ 3.97bpw and uniform_4bit/ 4.50bpw are NOT recommended for dense models;
#     a forced flat 4-bit on this dense model degenerates into repetition loops)
```

### Vision layout note
OptiQ stores the vision tower separately in `optiq/optiq_vision.safetensors` (prefix
`vision_tower.*`) and adds a custom `optiq_vision` field to config.json. This repo re-merges
those tensors into the main shards (renamed `model.visual.*`, matching the native Qwen prefix)
and updates `model.safetensors.index.json`, producing the standard MLX multimodal layout.
Vision is verified working on both the original sidecar layout and this unified layout
(`mlx_vlm`, mlx-vlm 0.6.13).

### Notes for agents republishing this model
- All weights are **real files** in this repo (no symlinks) — safe to mirror/convert directly.
- This is **multimodal**: set `pipeline_tag: image-text-to-text` and the `vision`/`multimodal`
  tags, as done in this card's frontmatter.
- The vision tower is already inside the shards as `model.visual.*`; do **not** strip it.
- `config.json` intentionally has no `optiq_vision` field — it was removed when the tower was
  merged in. Keep it that way.
- If you re-quantize, prefer `--candidate-bits 4,8` mixed precision. A flat 4-bit re-quant
  of this dense model degenerates into repetition loops.

## Model card (original)

`Qwen/Qwen3.8-27B` — Apache-2.0. See the [original model card](https://huggingface.co/Qwen/Qwen3.8-27B)
for intended use, license, and upstream Qwen3.8 details.
