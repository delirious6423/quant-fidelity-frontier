---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
language:
  - ar
  - en
tags:
  - tawkeed
  - gpt-t1
  - tawkeed-t1-mini
  - nvfp4
---

# GPT T1 Mini

The language model for the Tawkeed T1 Mini device, part of the GPT T1 family from [Tawkeed](https://tawkeed.ai).

| | |
|---|---|
| Device | Tawkeed T1 Mini: one NVIDIA RTX PRO 4500 Blackwell (32 GB) or RTX PRO 5000 Blackwell (48 GB) |
| Base model | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Precision | 4-bit (NVFP4) for the MLP layers and output head, FP8 for the attention layers |
| Size | About 21.9 GB, sized for the device's GPU memory |
| Architecture | 27.8B parameters, dense, hybrid linear and full attention |
| Input | Text and images |
| Context length | 262,144 tokens |
| License | Apache-2.0 (see `LICENSE`) |

## Arabic

Tawkeed plans to post-train GPT T1 Mini on Saudi-dialect Arabic to improve its Arabic answers. The post-trained version will be published in this repository.

## Run

Follow the serving instructions on the source build's card (see [Source and license](#source-and-license)), using this repository's id. The model has not yet been measured on a T1 Mini.

## The GPT T1 family

| Model | Device | GPU memory | Base model |
|---|---|---|---|
| [gpt-t1-mini](https://huggingface.co/tawkeed-sa/gpt-t1-mini) | T1 Mini | 32 or 48 GB, one GPU | Qwen3.8-27B |
| [gpt-t1](https://huggingface.co/tawkeed-sa/gpt-t1) | T1 | 192 GB, two GPUs | GLM-5.3-Flash |
| [gpt-t1-max](https://huggingface.co/tawkeed-sa/gpt-t1-max) | T1 Max | 384 GB, four GPUs | DeepSeek-V4.1-Flash-1017 |

## Source and license

These weights are NVIDIA's NVFP4 build of Qwen/Qwen3.8-27B, copied unchanged from [nvidia/Qwen3.8-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) at revision `482ca0f3832238542f8f5295dde86b5f22711d80`. Every file except this card is byte-identical to that revision. NVIDIA lists GB300 as its test hardware.

Qwen3.8-27B is by the Qwen team, released under the Apache License 2.0, and NVIDIA's build is released under the same license. The license text is kept in `LICENSE`.
