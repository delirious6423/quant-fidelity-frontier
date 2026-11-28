---
license: other
license_name: nvidia-open-model-agreement
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - ar
  - en
tags:
  - tawkeed
  - gpt-t1
  - tawkeed-t1
  - nvfp4
---

# GPT T1

The language model for the Tawkeed T1 device, part of the GPT T1 family from [Tawkeed](https://tawkeed.ai).

| | |
|---|---|
| Device | Tawkeed T1 with two NVIDIA RTX PRO 6000 Blackwell Max-Q (192 GB in total). A two-GPU T1 Max runs it too |
| Base model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Precision | 4-bit (NVFP4) routed experts, 8-bit (MXFP8) attention |
| Size | About 177.5 GB, sized for two GPUs (about 166 GiB of GPU memory in use) |
| Architecture | 320B parameters, mixture of experts, 18B active per token, hybrid sparse and linear attention |
| Input | Text, images and video |
| Context length | Up to 1M tokens |
| License | NVIDIA Open Model Agreement (see `LICENSE`); base model under the MIT License |

## Arabic

Tawkeed plans to post-train GPT T1 on Saudi-dialect Arabic to improve its Arabic answers. The post-trained version will be published in this repository.

## A T1 with one GPU

These weights need two GPUs. A T1 with one GPU is planned to run the same base model from a separate 4-bit GGUF build, Unsloth's [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (UD-Q4_K_XL), served by llama.cpp with part of the model held in the device's system memory. That build is not in this repository and has not yet been measured on a T1.

## Run

Follow the serving instructions on the source build's card (see [Source and license](#source-and-license)), using this repository's id. Until upstream vLLM supports this checkpoint's scale format, it needs the community vLLM container named there. The model has not yet been measured on a T1.

## The GPT T1 family

| Model | Device | GPU memory | Base model |
|---|---|---|---|
| [gpt-t1-mini](https://huggingface.co/tawkeed-sa/gpt-t1-mini) | T1 Mini | 32 or 48 GB, one GPU | Qwen3.8-27B |
| [gpt-t1](https://huggingface.co/tawkeed-sa/gpt-t1) | T1 | 192 GB, two GPUs | GLM-5.3-Flash |
| [gpt-t1-max](https://huggingface.co/tawkeed-sa/gpt-t1-max) | T1 Max | 384 GB, four GPUs | DeepSeek-V4.1-Flash-1017 |

## Source and license

These weights are NVIDIA's FP4-Local build of GLM-5.3-Flash, made with quantization-aware distillation and copied unchanged from [nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) at revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`. Every file except this card is byte-identical to that revision; `LICENSE`, `NOTICE` and `NVIDIA-Open-Model-Agreement.pdf` were added. NVIDIA lists 2x NVIDIA RTX PRO 6000 Blackwell and 2x NVIDIA DGX Spark as its test hardware, and reports these scores against the full-precision base model:

| Precision | GPQA Diamond | ToolEvalBench |
|---|---|---|
| BF16 (base model) | 92.1% | 91.9% |
| 4-bit (these weights) | 91.3% | 92.4% |

GLM-5.3-Flash is by Z.ai, released under the MIT License. The build is licensed by NVIDIA Corporation under the NVIDIA Open Model Agreement. A copy of that agreement is in `NVIDIA-Open-Model-Agreement.pdf`, and `LICENSE` holds both notices.
