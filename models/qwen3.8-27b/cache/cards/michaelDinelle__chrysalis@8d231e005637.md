---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
- Qwen/Qwen3.8-27B
- Qwen/Qwen3.8-Flash-Next
tags:
- gguf
- chrysalis
- cuda
- rocm
- metal
---

# Chrysalis model files

Qwen3.8-Flash-Next and Qwen3.8-27B for [Chrysalis](https://github.com/michaelDinelle/chrysalis-inference), on NVIDIA,
AMD and Apple Silicon. Chrysalis downloads the size that fits your machine.

| File | Model | Size | Error vs the reference |
| --- | --- | ---: | ---: |
| `flash-next/qwen3.8-flash-next-monarch.gguf` | Qwen3.8-Flash-Next | 63.4 GB | KL 0.033 vs Q8_0 |
| `qwen3.8-27b-monarch.gguf` | Qwen3.8-27B | 15.9 GB | KL 0.014 vs BF16 |
| `vision/mmproj-*-bf16.gguf` | Each model's vision tower | 0.9 GB each | |

Error is the mean KL divergence over 1,082 tokens of prose, a technical explanation and Python. Every file is the whole
model: every expert, the full 262K context and the MTP head. Flash-Next also reads its n-gram table from unsloth's Q8_0
shard 3, which Chrysalis fetches.

The files are laid out for Chrysalis's engine, with Flash-Next's experts grouped by how often they are used, and are
not meant for llama.cpp.

## Licenses

- Qwen3.8-27B and its files: Apache-2.0 ([`LICENSE-qwen3.8-27b`](LICENSE-qwen3.8-27b)).
- Qwen3.8-Flash-Next and its files: the Qwen Community License 1.0
  ([`LICENSE-qwen3.8-flash-next`](LICENSE-qwen3.8-flash-next)). It allows redistributing these derivative weights with
  its notice; a business offering a hosted model service or a commercial coding or office assistant needs a separate
  license from Qwen.
