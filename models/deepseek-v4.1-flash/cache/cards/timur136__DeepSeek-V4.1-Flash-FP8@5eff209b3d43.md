---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
pipeline_tag: image-text-to-text
tags:
  - deepseek-v4.1
  - fp8
  - quantized
  - multimodal
---

# DeepSeek V4.1 Flash FP8

This is a weight-only FP8 conversion of DeepSeek V4.1 Flash from source revision
`dba1be0a40aa45a94ad051997016db3960a90277`. The architecture, tokenizer, and non-expert tensor values are
unchanged. There was no training or fine-tuning.

Packed E2M1 FP4 routed-expert weights were expanded to E4M3 FP8. Their UE8M0
scales were repacked for 32 × 32 FP8 blocks. The conversion retains the original
tensor names and 48 shard assignments. Dense and other non-expert tensors were
copied byte for byte, including the Engram tables.

The checkpoint contains 96,085 tensors across 48 safetensors shards, with
772,004,202,840 bytes of tensor payload.

The original MIT copyright and permission notice are preserved in `LICENSE`.
