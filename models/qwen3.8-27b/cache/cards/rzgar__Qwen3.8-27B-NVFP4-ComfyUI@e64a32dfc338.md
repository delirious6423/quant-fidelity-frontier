---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- qwen
- qwen3.8-27b
---

**ComfyUI node update (Sep 17, 2026):**
```
Fix nvfp4 vision-tower crash with image/video input on ComfyUI 0.35.1+:
the vision tower now stays on the full-precision path instead of being switched
to W4A4 nvfp4 matmul (which can't take the float32 vision activations).

```
# Qwen3.8-27B ComfyUI NVFP4

A ComfyUI-native quantized conversion of [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)


_Unlike the core node, the custom node doesn’t run for `max_length` steps and finishes within seconds_

<a href="https://huggingface.co/rzgar/Qwen3.8-27B-NVFP4-ComfyUI/resolve/main/workflow/screenshot.png" target="_blank"><img src="https://huggingface.co/rzgar/Qwen3.8-27B-NVFP4-ComfyUI/resolve/main/workflow/screenshot.png" width="1280" height="720" /></a>

## File

| File | Download |
|---|---|
| `Qwen3.8-27B-nvfp4-comfy.safetensors` | [**Download 22.6 GB**](https://huggingface.co/rzgar/Qwen3.8-27B-NVFP4-ComfyUI/resolve/main/Qwen3.8-27B-nvfp4-comfy.safetensors) |
| `Qwen3.8-27B-nvfp4-all-comfy.safetensors` | [**Download 16GB**](https://huggingface.co/rzgar/Qwen3.8-27B-NVFP4-ComfyUI/resolve/main/Qwen3.8-27B-nvfp4-all-comfy.safetensors) |

## Usage

Place the file in `ComfyUI/models/text_encoders/` and load it with the LLM/text-encoder loader. Use with the **Generate Text** node.

**Recommended: the "Generate Text (Qwen3.8)" custom node** (install [`ComfyUI-Qwen3.8-Text`](https://huggingface.co/rzgar/Qwen3.8-27B-NVFP4-ComfyUI/resolve/main/ComfyUI-Qwen3.8-Text/ComfyUI-Qwen3.8-Text.zip))

The official **Generate Text** node for quantized checkpoints, it projects the output through

the embedding table instead of the real lm_head (untied), which produces incoherent text that **never stops and runs to `max_length`**. 

**The custom node** formats the prompt with the model's own chat template, uses the correct lm_head weights, and supports **image and video inputs** 
typical captions/answers **finish in seconds**.

## Comparison

| | `...-nvfp4-all-comfy` | `...-nvfp4-comfy` |
|---|---|---|
| Size | 16 GB | 22.6 GB |
| MLP | NVFP4 (fp4) | NVFP4 (fp4) |
| Attention / linear-attn / lm_head | NVFP4 (fp4) | FP8 (e4m3) |
| Embeddings | FP8 (per-tensor) | BF16 |
| Vision tower | NVFP4 | BF16 |
| Quality | slightly lower attention precision | fp8 attention, mirrors unsloth's design |


