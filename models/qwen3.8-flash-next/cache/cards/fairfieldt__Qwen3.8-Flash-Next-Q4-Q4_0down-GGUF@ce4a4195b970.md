---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
- Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- gguf
- qwen4exp
---

# Qwen3.8-Flash-Next Q4 GGUF, Q4_0 down experts

The Q4 from [antirez/qwen3.8-flash-next-gguf](https://huggingface.co/antirez/qwen3.8-flash-next-gguf)
(tensors from Ivan Fioravanti's [Q4](https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4)),
with the routed down experts quantized to Q4_0 straight from Qwen's BF16 checkpoint (revision `de4b8e4`)
instead of MXFP4. Every other tensor, including the BF16 n-gram table, is unchanged.

NLL on ds4's Alibaba fixtures goes from 0.2904 to 0.2863 (short) and 0.1256 to 0.1228 (long), at the
same speed. Needs a ds4 build with the Q4_0 down-expert kernels:
[fairfieldt/ds4](https://github.com/fairfieldt/ds4). Tested on an RTX PRO 6000.

`Qwen3.8-Flash-Next-Q4-Q4_0down.gguf`: 178,564,792,320 bytes, SHA-256
`951121e2cd69862a4b59f532b3b026ff2e7011e56ba94fba5e7a1efa47de9a90`.
