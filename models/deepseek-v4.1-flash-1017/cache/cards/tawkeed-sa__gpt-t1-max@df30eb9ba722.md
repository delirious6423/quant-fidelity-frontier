---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - ar
  - en
tags:
  - tawkeed
  - gpt-t1
  - tawkeed-t1-max
  - nvfp4
---

# GPT T1 Max

The language model for the Tawkeed T1 Max device, part of the GPT T1 family from [Tawkeed](https://tawkeed.ai).

| | |
|---|---|
| Device | Tawkeed T1 Max with four NVIDIA RTX PRO 6000 Blackwell Server Edition (384 GB in total) and 512 GB of system memory. A two-GPU T1 Max runs [gpt-t1](https://huggingface.co/tawkeed-sa/gpt-t1) |
| Base model | [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) |
| Precision | 4-bit (NVFP4) routed experts, 4-bit (MXFP4) Engram lookup tables |
| Size | About 429 GB. The Engram lookup tables (about 97 GiB) are served from system memory, and the rest of the model goes on the GPUs |
| Architecture | 552B-parameter mixture-of-experts backbone (384 routed experts, 6 active per token, about 16B active parameters when generating), plus 196B parameters of Engram lookup tables |
| Input | Text and images |
| Context length | Up to 1M tokens |
| License | MIT (see `LICENSE`) |

## Arabic

Tawkeed plans to post-train GPT T1 Max on Saudi-dialect Arabic to improve its Arabic answers. The post-trained version will be published in this repository.

## Run

Follow the serving instructions on the source build's card (see [Source and license](#source-and-license)), using this repository's id. It needs a vLLM build that includes [vllm-project/vllm#61530](https://github.com/vllm-project/vllm/pull/61530), or an SGLang build from 2026-11-04 or later ([sgl-project/sglang#47935](https://github.com/sgl-project/sglang/pull/47935)), with the Engram tables placed in system memory. The model has not yet been measured on a T1 Max.

## The GPT T1 family

| Model | Device | GPU memory | Base model |
|---|---|---|---|
| [gpt-t1-mini](https://huggingface.co/tawkeed-sa/gpt-t1-mini) | T1 Mini | 32 or 48 GB, one GPU | Qwen3.8-27B |
| [gpt-t1](https://huggingface.co/tawkeed-sa/gpt-t1) | T1 | 192 GB, two GPUs | GLM-5.3-Flash |
| [gpt-t1-max](https://huggingface.co/tawkeed-sa/gpt-t1-max) | T1 Max | 384 GB, four GPUs | DeepSeek-V4.1-Flash-1017 |

## Source and license

These weights are NVIDIA's NVFP4 build of DeepSeek-V4.1-Flash-1017, copied unchanged from [nvidia/DeepSeek-V4.1-Flash-1017-NVFP4](https://huggingface.co/nvidia/DeepSeek-V4.1-Flash-1017-NVFP4) at revision `3ae9c0d3fbbea5370bd06e679d22044db15883f1`. Every file except this card is byte-identical to that revision. NVIDIA lists 4x NVIDIA GB300 as its test hardware, and reports these scores against DeepSeek's own release (MXFP4 routed experts):

| Precision | GPQA Diamond | AA-LCR | SciCode | IFBench | MMMU-Pro | Terminal-Bench 2.1 |
|---|---|---|---|---|---|---|
| DeepSeek's release | 92.77 | 80.41 | 57.58 | 76.33 | 78.38 | 81.81 |
| These weights | 92.93 | 80.19 | 57.32 | 76.47 | 78.73 | 81.11 |

DeepSeek-V4.1-Flash-1017 is by DeepSeek, released under the MIT License, and NVIDIA's build is released under the same license. The license text is kept in `LICENSE`.
