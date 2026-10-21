---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: mlx
tags:
- mlx
- vision
pipeline_tag: image-text-to-text
---

# ToPo-ToPo/Qwen3.8-27B-mlx-4bit

MLX **4bit** conversion of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) for Apple Silicon (mlx-vlm).

## Provenance (self-converted from official weights)
- Source: [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) (license: apache-2.0)
- Tool: `mlx-vlm 0.6.9` — `mlx_vlm.convert --hf-path Qwen/Qwen3.8-27B --mlx-path . -q --q-bits 4 --q-group-size 64`
- Effective: **4.695 bits/weight**
- Architecture: `Qwen3_5ForConditionalGeneration` (`model_type: qwen3_5`), 64 layers with
  3:1 linear/full attention, vision tower included (333 vision tensors).

## Validation
- Text and image input verified with `mlx_vlm.generate`.
- OpenAI-compatible gateway ([local-llm-server](https://github.com/ToPo-ToPo-ToPo/local-llm-server)):
  text, image (data URI) and **native tool calling** (`finish_reason: tool_calls`) all verified.
- End-to-end agent runs (MCP tools, write → read → report) completed without errors.

Throughput and memory use depend on your hardware, so measure them on your own machine.

## Note on MTP
The official checkpoint declares `mtp_num_hidden_layers: 1`, but the mlx-vlm `qwen3_5`
conversion does **not** carry the `model.mtp.*` weights. A drafter split from the official
bf16 checkpoint is published separately as
[`ToPo-ToPo/Qwen3.8-27B-MTP-bf16`](https://huggingface.co/ToPo-ToPo/Qwen3.8-27B-MTP-bf16)
(a 4bit variant is also available).

## Usage
```python
from mlx_vlm import load, generate
model, processor = load("ToPo-ToPo/Qwen3.8-27B-mlx-4bit")
```
