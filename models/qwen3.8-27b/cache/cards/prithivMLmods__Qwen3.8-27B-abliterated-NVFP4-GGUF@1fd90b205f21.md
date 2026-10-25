---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- text-generation-inference
- llama-cpp
- nvfp4
- blackwell
- mtp
- qwen3_5
- speculative-decoding
- nvidia-blackwell
- rtx-pro-6000
- abliterated
- qwen3.8
- multimodal
- vision-language-model
- image-text-to-text
- image-understanding
- visual-question-answering
- visual-reasoning
- document-understanding
- image-captioning
- dense
- 27B
- llama-cli
language:
- en
---

# **Qwen3.8-27B-abliterated-NVFP4-GGUF**

> **Qwen3.8-27B** is a 27-billion-parameter dense vision-language model from the Qwen team, built on the Qwen3.5 architectural foundation. Its 64-layer hybrid architecture combines Gated DeltaNet linear attention with periodic Gated Attention layers, supports Multi-Token Prediction, a native 262,144-token context window (extendable to 1M via YaRN), and multimodal image and video understanding. Thinking is enabled by default, with configurable `reasoning_effort` (`xhigh`, `medium`, `low`). It excels in agentic coding and computer-use tasks, scoring 73.0 on Terminal-Bench 2.1, 61.7 on SWE-bench Pro, 84.3 on OSWorld-Verified, 81.9 on AndroidWorld, and 90.3 on LiveCodeBench v6, alongside 89.2 on GPQA-Diamond and 79.5 on IFBench. It supports Transformers, vLLM, SGLang, and TokenSpeed and is released under Apache-2.0.

> [!NOTE]
> MTP — Speculative Decoding: Yes

> [!NOTE]
Multi-Token Prediction (MTP) GGUF is a specialized GGUF model file format extension that integrates speculative decoding directly into the model weights to significantly accelerate local inference.  Unlike traditional speculative decoding which requires a separate, smaller "draft" model, MTP GGUF files include additional output heads within the main model architecture that predict multiple future tokens in a single forward pass.

## Model Files

| File Name | Quant Type | File Size | File Link | Description |
|-----------|------------|-----------|-----------|-------------|
| Qwen3.8-27B-abliterated-NVFP4.gguf | NVFP4 | 19.7 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-abliterated-NVFP4-GGUF/blob/main/Qwen3.8-27B-abliterated-NVFP4.gguf) | NVFP4 quantization. Highly optimized for specific hardware accelerators, offering a strong balance of size and performance. |
| Qwen3.8-27B-abliterated-NVFP4.mmproj.gguf | mmproj | 931 MB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-abliterated-NVFP4-GGUF/blob/main/Qwen3.8-27B-abliterated-NVFP4.mmproj.gguf) | Multimodal projection file. Used for vision/language models. |

## NVFP4 Test Run

Tested `Qwen3.8-27B-abliterated-NVFP4.gguf` on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU with `96 GB` of VRAM, using `llama-cli` with full GPU offloading and MTP speculative decoding.

### Performance Metrics

| Metric | Value |
| --- | --- |
| Average Prompt Processing Speed | 220.54 tokens/s |
| Average Generation Speed | 111.67 tokens/s |
| Average Wall Time | 11.44 seconds |
| Total Wall Time | 57.2 seconds |
| Context Length | 16,384 tokens |
| KV Cache (K) | `q8_0` |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU VRAM Capacity | 96 GB |
| Peak GPU VRAM Usage | 18.22 GiB |
| Peak GPU Utilization | 96% |
| GPU Layers Offloaded | 99 (`-ngl 99`) |
| Random Seed | 42 |
| Speculative Decoding | MTP (`draft_n_max=2`) |
| Test Runs | 5 |

### Test Shots

| Shot | Test |
| --- | --- |
| Shot 1 | C: Insert Node at End of Singly Linked List |
| Shot 2 | Jogger and Train Passing Time |
| Shot 3 | Two Pipes Filling a Tank |
| Shot 4 | Students in Examination Rooms |
| Shot 5 | Father and Son Ages |

### Run Inference

```bash
./build/bin/llama-cli \
  -m /path/to/models/Qwen3.8-27B-abliterated-NVFP4.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ngl 99 \
  -c 16384 \
  --cache-type-k q8_0 \
  --seed 42 \
  -n 8192 \
  -st \
  -p "Your prompt here"
```

*Performance results reflect this specific hardware and inference configuration. Actual throughput may vary depending on the prompt, model, software build, and runtime settings.*

## llama.cpp

LLM inference in C/C++ — https://github.com/ggml-org/llama.cpp

---

## Snapshots (5)

![download (1)](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/p4SMcuA-bf_ghULikObAK.png)
![download (2)](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/UzhFbZ53kSJJxnlt9ej49.png)
![download (3)](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/egE6q65zA0u9dZQh0O_MT.png)
![download (4)](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/2i6wTH-UfJEWqLuKRuFyF.png)
![download (5)](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/fgMfqOReignhlim6YG70Q.png)
![download](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/ajQuBYJBlF4DaZicAHx1Y.png)

---

## **License and Attributions**

* **Base Model**: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).

* **Source Model**: [huihui-ai/Huihui-Qwen3.8-27B-abliterated](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-27B-abliterated): Used for NVFP4 compression and quantization.

* **Quantization**: Performed using `llama.cpp` by [**prithivMLmods**](https://huggingface.co/prithivMLmods).

* **License**: This model follows the **Apache 2.0** license of the original [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) base model.