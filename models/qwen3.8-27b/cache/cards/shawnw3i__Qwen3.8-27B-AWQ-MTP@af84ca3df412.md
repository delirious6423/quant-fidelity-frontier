---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- uncensored
- awq
- mtp
- vllm
---

# shawnw3i/Qwen3.8-27B-AWQ-MTP

## Highlights

- **AWQ Marlin** kernel supported (auto-converted by vLLM at runtime)
- **MTP speculative decoding** supported out of the box
- **110+ tok/s** on a single A800 80GB (vLLM 0.27.1, MTP enabled, fp8 KV cache)
- **Vision** inputs supported.

## Update
`09-27-2026` Because Qwen3.8 utilizes non-traditional attention, the group size was reduced to 64 to prevent precision collapse.


## vLLM

```
vllm serve shawnw3i/Qwen3.8-27B-AWQ-MTP \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```