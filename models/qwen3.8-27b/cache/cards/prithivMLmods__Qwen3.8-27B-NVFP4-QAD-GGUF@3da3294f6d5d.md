---
license: other
license_name: nvidia-open-model-license
license_link: https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4-QAD/blob/main/LICENSE
library_name: transformers
base_model:
- Qwen/Qwen3.8-27B
language:
- en
pipeline_tag: image-text-to-text
tags:
- REPACKED
- mtp
- text-generation-inference
- QAD
- qwen3_5
- stable
- Model Optimizer
- ModelOpt
- llama-cpp
- nvfp4
- nvidia
- llama-cli
- speculative-decoding
- nvidia-blackwell
- rtx-pro-6000
- SM120
- GB202
- B200
- B300
- FP4
- blackwell
- Qwen3.5
- Qwen3.6
- Qwen3.8
- 27B
- multimodal
- mixed-precision
- conversational
- quantized
- hf-bucket
- vLLM
- Quantization-Aware-Distillation
- Alibaba
- Nemotron
- SFT
datasets:
- nvidia/Nemotron-Cascade-2-SFT-Data
---

![qwen3_8_27b_nvfp4_qad_benchmark_card](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/cHJxEDCXAdC-PjyeRZbYr.png)

> [!NOTE]
> NVFP4 QAD GGUF Test Run Log – HF Bucket: [qwen3-8-27b-nvfp4-qad-11-28-2026](https://huggingface.co/buckets/prithivMLmods/qwen3-8-27b-nvfp4-qad-11-28-2026/tree/session_20261128_110618_IST)

# **Qwen3.8-27B-NVFP4-QAD-GGUF**

> **Qwen3.8-27B-NVFP4-QAD**, released on November 4, 2026, is an NVFP4-quantized multimodal version of Alibaba's **Qwen3.8-27B** (`Qwen3_5ForConditionalGeneration`) produced with NVIDIA Model Optimizer (v0.48.0) using Quantization-Aware Distillation (QAD) over 1.26B tokens from `Nemotron-Cascade-2-SFT-Data`. By quantizing the weights and activations of linear operators inside transformer blocks while updating underlying weights against token-level BF16 teacher probabilities, it cuts disk and GPU memory footprints by roughly 2.8× with virtually zero accuracy loss—matching or slightly exceeding the BF16 baseline across GPQA Diamond (87.94 vs. 88.19), IFBench (79.04 vs. 79.25), MMMU Pro (75.61 vs. 75.78), SciCode (48.82 vs. 48.34), Terminal-Bench 2.1 (76.69 vs. 77.25), and AA-LCR (79.13 vs. 78.63). Supporting text, image, and video inputs across a 262,144-token context window on NVIDIA Blackwell GPUs via `vLLM` (`--quantization modelopt`) and `SGLang`, it is licensed for commercial and non-commercial use under the **NVIDIA Open Model Agreement**.

> [!NOTE]
> MTP — Speculative Decoding: Yes

> [!NOTE]
> Multi-Token Prediction (MTP) GGUF is a specialized GGUF model file format extension that integrates speculative decoding directly into the model weights to significantly accelerate local inference.  Unlike traditional speculative decoding which requires a separate, smaller "draft" model, MTP GGUF files include additional output heads within the main model architecture that predict multiple future tokens in a single forward pass.

## Model Files

| File Name | Quant Type | File Size | File Link | Description |
|-----------|------------|-----------|-----------|-------------|
| Qwen3.8-27B-NVFP4-QAD.gguf | NVFP4 | 19.1 GB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-GGUF/blob/main/Qwen3.8-27B-NVFP4-QAD.gguf) | **NVFP4**: Highly optimized for **NVIDIA Blackwell** hardware accelerators **(e.g., RTX Pro 6000, RTX 50-series, B200, B300)**, offering native hardware-accelerated FP4 Tensor Core speeds and a strong balance of size and performance. |
| Qwen3.8-27B-NVFP4-QAD.mmproj-bf16.gguf | mmproj-bf16 | 931 MB | [Link](https://huggingface.co/prithivMLmods/Qwen3.8-27B-NVFP4-QAD-GGUF/blob/main/Qwen3.8-27B-NVFP4-QAD.mmproj-bf16.gguf) | Multimodal projection file in BF16 format. Used for vision/image capabilities. |

## NVFP4 Test Run

Tested `Qwen3.8-27B-NVFP4-QAD.gguf` on an NVIDIA RTX PRO 6000 Blackwell Server Edition GPU with `96 GB` of VRAM, using `llama-cli` with full GPU offloading, Flash Attention enabled, reasoning enabled, and MTP speculative decoding.

### Performance Metrics

| Metric | Value |
| --- | --- |
| Average Prompt Processing Speed | 177.89 tokens/s |
| Average Generation Speed | 132.12 tokens/s |
| Total Wall Time | 41.6 seconds |
| Context Length | 16,384 tokens |
| Flash Attention | On (`--flash-attn on`) |
| KV Cache (K) | `bf16` |
| Thinking / Reasoning | On (`--reasoning on`) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| GPU VRAM Capacity | 96 GB |
| Peak GPU VRAM Usage | 17.79 GiB |
| Peak GPU Utilization | 97% |
| GPU Layers Offloaded | 99 (`-ngl 99`) |
| Random Seed | 42 |
| Speculative Decoding | MTP (`draft_n_max=2`) |
| HF Bucket | `qwen3-8-27b-nvfp4-qad-11-28-2026` |
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
  -m /path/to/models/Qwen3.8-27B-NVFP4-QAD.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --reasoning on \
  --flash-attn on \
  -ngl 99 \
  -c 16384 \
  --cache-type-k bf16 \
  --seed 42 \
  -n 8192 \
  -st \
  -p "Your prompt here"
```

*Performance results reflect this specific hardware and inference configuration. Actual throughput may vary depending on the prompt, model, software build, and runtime settings.*

## llama.cpp

LLM inference in C/C++ — https://github.com/ggml-org/llama.cpp

Quantized using llama.cpp by [prithivMLmods](https://huggingface.co/prithivMLmods).