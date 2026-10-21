---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - mlx
  - apple-silicon
  - quantized
  - vision
  - mtp
  - qwen3
language:
  - en
pipeline_tag: image-text-to-text
---

# Qwen3.8-27B · oQe Q3.5 bf16 · Vision + MTP Verified

Qwen3.8-27B quantized with oMLX's oQe (imatrix-weighted) pipeline at Q3.5 bit depth, bf16 weights. Vision tower and Multi-Token Prediction (Lightning MTP) tensors are **verified present** by direct inspection of the output checkpoint -- not inferred from the source or claimed from the conversion flags.

## Why this exists

HuggingFace has been flooded with Qwen3.x uploads claiming to have MTP retained and the vision tower intact. In my experience, that was rarely the case -- conversion pipelines silently strip those components, and there's no standard way to verify that from the outside without opening the checkpoint yourself. I downloaded enough of those models and got burned enough times that I started quantizing from the official original weights myself.

This model was quantized directly from `Qwen/Qwen3.8-27B` -- the official full-precision source, not a community re-upload. MTP and vision tensors were verified after quantization by inspecting the safetensors index directly.

## Verification

Tensor counts in this checkpoint, from the output `model.safetensors.index.json`:

| Component | Tensors present |
|---|---|
| Vision tower (`vision_tower.*`) | 333 |
| MTP heads (`language_model.mtp.*`) | 29 |
| Total | 2,209 |

If you want to verify yourself:

```python
import json
idx = json.load(open("model.safetensors.index.json"))
tensors = list(idx["weight_map"].keys())
print("MTP:", len([t for t in tensors if "mtp" in t]))
print("Vision:", len([t for t in tensors if "vision" in t or "visual" in t]))
```

## What oQe is

oQe is oMLX's imatrix-calibrated quantization. Rather than quantizing all weights uniformly, it first collects activation statistics over a calibration corpus, then uses those statistics to guide per-tensor quantization -- protecting the weights that matter most for accuracy from the worst precision loss. At any given file size, oQe typically outperforms plain round-to-nearest quantization on downstream benchmarks because the bit budget is spent where it counts.

## fp16 vs bf16 -- which to use

**M3 / M4:** Apple improved bf16 throughput in these generations -- bf16 is the natural choice and this checkpoint is built for it.

**M1 / M2:** fp16 runs meaningfully faster on these chips. Use the fp16 companion [`Robot-Haus/Qwen3.8-27B-original-oQ3.5e-fp16-mtp`](https://huggingface.co/Robot-Haus/Qwen3.8-27B-original-oQ3.5e-fp16-mtp) instead.

**CUDA:** bf16 is typically the native type on NVIDIA hardware, so this checkpoint may work there, but it was not tested on CUDA.

## Quantization details

| | |
|---|---|
| **Source** | `Qwen/Qwen3.8-27B` (official, unmodified) |
| **Method** | oQe -- oMLX imatrix-calibrated |
| **Bit depth** | Q3.5 equivalent (mixed precision, imatrix-guided) |
| **Weight dtype** | bf16 |
| **Vision** | Intact -- 333 tensors |
| **MTP** | Intact -- 29 tensors (Lightning MTP compatible) |
| **Size** | ~14 GB |

## Usage

Designed for [oMLX](https://omlx.ai). Load as a standard MLX model. Enable "Lightning MTP" in oMLX settings to activate the MTP drafting heads for faster decode throughput.

Vision inputs work out of the box -- no separate vision checkpoint needed, the tower is part of this file.

## What this is not

This is a straight quantization of the original weights. No fine-tuning, no abliteration, no merges. The model's behavior, instruction-following, and capabilities are the same as the base `Qwen/Qwen3.8-27B` -- the quantization changes precision, not the model.
