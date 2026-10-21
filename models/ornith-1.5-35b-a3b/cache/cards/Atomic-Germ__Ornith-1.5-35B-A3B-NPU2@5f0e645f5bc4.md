---
license: apache-2.0
language:
- en
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
quantized_by: Atomic-Germ
pipeline_tag: image-text-to-text
tags:
- qwen3_5_moe
- qwen3.6-moe
- qwen3.6
- moe
- qwen
- vision
- fastflowlm
- q4nx
- npu
- text-generation
- conversational
- en
- vlm
---

# Ornith-1.5-35B-A3B-NPU2

**FastFlowLM Q4NX conversion of [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)** for AMD XDNA NPU inference.

This repository contains a quantized **Q4NX** port of the model, compiled for the FastFlowLM (FLM) runtime. It is **not** a GGUF file.

| Item | Value |
|------|-------|
| Source model | [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) |
| Source GGUF | `Ornith-1.5-35B-A3B-MXFP4_MOE_Q8_0-Imatrix.gguf` |
| Weights | `model.q4nx` (23.80 GB) |
| Modality | language / vision |
| FLM version | `1.0.1` |
| Converted | 2026-08-21 |

## Install and run

This repository works with `flm-add`, a small installer that copies the model
into the FastFlowLM user directory and registers the tag. It never
modifies the system FastFlowLM install.

`pip install flm-add` or `uv tool install flm-add`

```bash
uv tool install flm-add
flm-add Atomic-Germ/Ornith-1.5-35B-A3B-NPU2 --tag ornith1.5:35b-a3b --family qwen3.6-moe
FLM_CONFIG_PATH="$HOME/.config/flm/model_list.json" FLM_XCLBIN_PATH="$HOME/.config/flm" flm run Ornith-1.5-35B-A3B-NPU2
```

## Files

| File | Description |
|------|-------------|
| `model.q4nx` | Quantized weights (Q8_0 / Q4_1 / BF16) |
| `config.json` | FLM runtime configuration |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer configuration |
| `chat_template.jinja` | Chat template |
| `vision_weight.q4nx` | Vision model |


---

## Source model card

See the original model card: [ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)