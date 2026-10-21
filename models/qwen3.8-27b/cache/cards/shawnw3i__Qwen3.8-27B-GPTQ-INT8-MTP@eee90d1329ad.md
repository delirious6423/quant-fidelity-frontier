---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
tags:
- uncensored
- mtp
- vllm
- gptq
---

# shawnw3i/Qwen3.8-27B-GPTQ-INT8-MTP

## Highlights

- **GPTQ Marlin** kernel supported (auto-converted by vLLM at runtime)
- **MTP speculative decoding** supported out of the box
- **80+ tok/s** on a single A800 80GB (vLLM 0.27.1, MTP enabled, fp8 KV cache)
- **Vision** inputs supported.


## vLLM

```
vllm serve shawnw3i/Qwen3.8-27B-GPTQ-INT8-MTP \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```