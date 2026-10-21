---
license: apache-2.0
language:
- en
base_model:
- google/gemma-4-26B-A4B-it
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- text-generation-inference
- llama.cpp
---

# **gemma-4-26B-A4B-it-F32-GGUF**

> Gemma-4-26B-A4B-it from Google is a 26B total parameter Mixture-of-Experts (MoE) multimodal model with only 3.8B-4B active parameters per forward pass (8 active + 1 shared expert from 128 total), delivering near-equivalent quality to the dense 31B sibling at dramatically lower compute/memory cost while supporting 256K context length, 1024-token sliding window, text+image modalities (variable aspect ratio/resolution), and advanced agentic capabilities. Featuring 30 layers and 262K vocabulary across 140+ languages, the instruction-tuned variant excels at reasoning (configurable thinking modes), coding, OCR/handwriting recognition, document parsing, UI analysis, chart comprehension, and object detection with pointing—optimized for high-throughput server/workstation deployment on NVIDIA/AMD GPUs via vLLM/llama.cpp with Apache 2.0 licensing. Positioned between edge-focused E2B/E4B and flagship 31B models in the Gemma 4 family, it balances frontier-level multimodal intelligence with production-scale efficiency for enterprise agents, function calling, and structured data workflows.

## Quick start with llama.cpp

```
llama-server -hf prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF:F32
```

## Model Files

   File Name | Quant Type | File Size | File Link |
 | - | - | - | - |
 | gemma-4-26B-A4B-it.BF16.gguf | BF16 | 50.5 GB | [Download](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.BF16.gguf) |
 | gemma-4-26B-A4B-it.F16.gguf | F16 | 50.5 GB | [Download](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.F16.gguf) |
 | gemma-4-26B-A4B-it.F32.gguf | F32 | 101 GB | [Download](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.F32.gguf) |
 | gemma-4-26B-A4B-it.Q8_0.gguf | Q8_0 | 26.9 GB | [Download](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.Q8_0.gguf) |
 | gemma-4-26B-A4B-it.mmproj-bf16.gguf | mmproj-bf16 | 1.19 GB | [Download](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.mmproj-bf16.gguf) |
 | gemma-4-26B-A4B-it.mmproj-f16.gguf | mmproj-f16 | 1.19 GB | [Download](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.mmproj-f16.gguf) |
 | gemma-4-26B-A4B-it.mmproj-f32.gguf | mmproj-f32 | 2.29 GB | [Download](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.mmproj-f32.gguf) |
 | gemma-4-26B-A4B-it.mmproj-q8_0.gguf | mmproj-q8_0 | 806 MB | [Download](https://huggingface.co/prithivMLmods/gemma-4-26B-A4B-it-F32-GGUF/blob/main/GGUF/gemma-4-26B-A4B-it.mmproj-q8_0.gguf) |

 ## Quants Usage 

(sorted by size, not necessarily quality. IQ-quants are often preferable over similar sized non-IQ quants)

Here is a handy graph by ikawrakow comparing some lower-quality quant
types (lower is better):

![image.png](https://www.nethype.de/huggingface_embed/quantpplgraph.png)