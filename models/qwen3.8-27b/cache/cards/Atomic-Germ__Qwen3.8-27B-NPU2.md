---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
quantized_by: Atomic-Germ
pipeline_tag: image-text-to-text
tags:
- transformers
- safetensors
- qwen3_5
- image-text-to-text
- conversational
- license:apache-2.0
- eval-results
- endpoints_compatible
- deploy:azure
- deploy:sagemaker
- region:us
---
# c27

**OpenFlowLM Q4NX conversion of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B)** for AMD XDNA NPU inference.

This repository contains a quantized **Q4NX** port of the model, compiled for the OpenFlowLM (OFLM) runtime. It is **not** a GGUF file.

| Item | Value |
|------|-------|
| Source model | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Source repo | [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Source GGUF | `Qwen3.8-27B.Q8_0.gguf` |
| Weights | `model.q4nx` (18.48 GB) |
| Modality | language |
| OFLM version | `0.1.0` |
| Converted | 2026-10-03 |

## Reproduce this conversion

This container was produced by exactly this command, so a fine-tune of the
same base can be packed the same way:

```bash
oflm pack -i Qwen/Qwen3.8-27B -o /home/atomic-germ/Projects/q4nx-27b/c27 -s Qwen/Qwen3.8-27B
```


## Install and run

This repository works with `oflm-add`, a small installer that copies the model
into the OpenFlowLM user directory and registers the tag. It never
modifies the system OpenFlowLM install.

`pip install oflm-add` or `uv tool install oflm-add`

```bash
uv tool install oflm-add
oflm-add Atomic-Germ/c27 --family qwen3.5 --xclbin-from c27
OFLM_CONFIG_PATH="$HOME/.config/oflm/model_list.json" OFLM_XCLBIN_PATH="$HOME/.config/oflm" oflm run c27
```

## Files

| File | Description |
|------|-------------|
| `model.q4nx` | Quantized weights (Q8_0 / Q4_1 / BF16) |
| `config.json` | OFLM runtime configuration |
| `tokenizer.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer configuration |
| `chat_template.jinja` | Chat template |


---

## Source model card

See the original model card: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
