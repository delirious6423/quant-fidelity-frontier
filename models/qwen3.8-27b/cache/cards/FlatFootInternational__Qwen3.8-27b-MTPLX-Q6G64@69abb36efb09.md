---
license: apache-2.0
library_name: mlx
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - mlx
  - apple-silicon
  - macos
  - speculative-decoding
  - multi-token-prediction
  - qwen
  - qwen3.8
  - mtp
  - mtplx
  - local-ai
  - coding
---

# MTPLX Qwen 3.8 27B 6-bit dynamic quant.


Every weight matrix at 6-bit, sensitive parts at 16-bit, native multi-token-prediction head kept, so MTPLX still drafts ahead and verifies in one pass. 


## Speeds

Measured on an Macbook M5 (NOT pro or max) with 32GB unified memory, fans verified at max, single stream, generation running to the model's own stop, official Qwen 3.8 sampling (temperature 1.0, top-p 0.95, top-k 20).


## How it is built

- Every weight matrix at 6-bit with 64-weight groups.

- The GDN convolution kernels and recurrent state parameters, every norm, and the whole MTP head stay 16-bit.

- Download	23.6 GB

- Context window	262,144 tokens

- MTP depth	3

- Sampling: temperature 1.0, top-p 0.95, top-k 20 (the official Qwen 3.8 contract)


## Use it

You want 32 GB of unified memory or more for this one. 


## Command line:

pip install mtplx

mtplx serve --model FlatFootInternational/qwen3.8-27b-MTPLX-6bit
