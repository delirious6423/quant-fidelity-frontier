---
library_name: transformers
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: image-text-to-text
tags:
  - qwen
  - nvfp4
  - modelopt
  - moe
---

# Qwen3.8-Flash-Next W4A16 NVFP4

An NVIDIA ModelOpt W4A16 NVFP4 checkpoint of
[`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

The routed experts use NVFP4 weights with group size 16 and BF16 activations.
Attention, vision, shared experts, routers, PLE, embeddings, output head, and
MTP weights retain their source precision.

This repository contains the Hugging Face checkpoint, not GGUF. A runtime with
support for this ModelOpt W4A16 NVFP4 architecture is required.

The original model's Qwen Community License 1.0 is included. Architecture,
usage, chat format, and limitations are documented in the
[official model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
