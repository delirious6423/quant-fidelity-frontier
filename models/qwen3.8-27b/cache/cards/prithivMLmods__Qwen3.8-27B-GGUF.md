---
license: apache-2.0
language:
- en
- zh
library_name: transformers
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- text-generation-inference
- llama-cpp
- qwen3.8
- mtp
- multimodal
- vision-language-model
- vlm
- image-text-to-text
- image-understanding
- visual-question-answering
- visual-reasoning
- document-understanding
- OCR
- image-captioning
- video-captioning
- video-understanding
---

# **Qwen3.8-27B-GGUF**

> **[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** is a 27-billion-parameter dense causal language model with a native vision encoder from the Qwen team, built on the Qwen3.5 architectural foundation as a compact, deployment-friendly member of the newly introduced Qwen3.8 generation — the most capable in the Qwen open-model family to date. Its 64-layer hybrid architecture interleaves Gated DeltaNet linear-attention blocks with periodic Gated Attention layers, trained with Multi-Token Prediction, and supports a native 262,144-token context window (extensible to 1M via YaRN scaling), native image and video understanding from STEM diagrams to hour-scale videos, and flexible thinking control via a `reasoning_effort` parameter (xhigh/medium/low) with thinking enabled by default and historical reasoning preserved across turns. It delivers substantial gains over its predecessor Qwen3.6-27B and often rivals or exceeds larger models like Muse Glimmer-30B and even Opus 4.6 Max on several benchmarks — scoring 73.0 on Terminal-Bench 2.1, 61.7 on SWE-bench Pro, 84.3 on OSWorld-Verified computer-use, 81.9 on AndroidWorld mobile-use, and 90.3 on LiveCodeBench v6 — reflecting particular strength in agentic coding, computer/browser/mobile-use tasks, and multimodal tool use, while remaining competitive on general reasoning benchmarks like GPQA Diamond (89.2) and IFBench (79.5). It's compatible with Hugging Face Transformers, vLLM, SGLang, and TokenSpeed out of the box, released under Apache-2.0.

> [!NOTE]
Multi-Token Prediction (MTP) GGUF is a specialized GGUF model file format extension that integrates speculative decoding directly into the model weights to significantly accelerate local inference.  Unlike traditional speculative decoding which requires a separate, smaller "draft" model, MTP GGUF files include additional output heads within the main model architecture that predict multiple future tokens in a single forward pass.

## Model Files

   File Name | Quant Type | File Size | File Link |
 |-----------|------------|-----------|-----------|
 | Qwen3.8-27B.BF16.gguf | BF16 | 54.7 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.BF16.gguf) |
 | Qwen3.8-27B.F16.gguf | F16 | 54.7 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.F16.gguf) |
 | Qwen3.8-27B.Q2_K.gguf | Q2_K | 10.9 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q2_K.gguf) |
 | Qwen3.8-27B.Q3_K_L.gguf | Q3_K_L | 14.6 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q3_K_L.gguf) |
 | Qwen3.8-27B.Q3_K_M.gguf | Q3_K_M | 13.5 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q3_K_M.gguf) |
 | Qwen3.8-27B.Q4_0.gguf | Q4_0 | 15.7 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q4_0.gguf) |
 | Qwen3.8-27B.Q4_K_M.gguf | Q4_K_M | 16.8 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q4_K_M.gguf) |
 | Qwen3.8-27B.Q4_K_S.gguf | Q4_K_S | 15.8 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q4_K_S.gguf) |
 | Qwen3.8-27B.Q5_0.gguf | Q5_0 | 19 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q5_0.gguf) |
 | Qwen3.8-27B.Q5_K_M.gguf | Q5_K_M | 19.5 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q5_K_M.gguf) |
 | Qwen3.8-27B.Q5_K_S.gguf | Q5_K_S | 19 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q5_K_S.gguf) |
 | Qwen3.8-27B.Q6_K.gguf | Q6_K | 22.4 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q6_K.gguf) |
 | Qwen3.8-27B.Q8_0.gguf | Q8_0 | 29 GB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.Q8_0.gguf) |
 | Qwen3.8-27B.mmproj-bf16.gguf | mmproj-bf16 | 931 MB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.mmproj-bf16.gguf) |
 | Qwen3.8-27B.mmproj-f16.gguf | mmproj-f16 | 931 MB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.mmproj-f16.gguf) |
 | Qwen3.8-27B.mmproj-q8_0.gguf | mmproj-q8_0 | 629 MB | [Download](https://huggingface.co/prithivMLmods/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B.mmproj-q8_0.gguf) |

## llama.cpp

LLM inference in C/C++ — https://github.com/ggml-org/llama.cpp

```
llama cli -hf prithivMLmods/Qwen3.8-27B-GGUF:Q4_K_M
```