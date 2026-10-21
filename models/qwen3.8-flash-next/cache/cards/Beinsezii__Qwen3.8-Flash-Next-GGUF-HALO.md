---
license: mit
base_model:
- Qwen/Qwen3.8-Flash-Next
---

Quant optimized for quality / speed on a Strix Halo 128GiB system. Possibly also beneficial on DGX Spark and similar systems. Refer to [tensor types](./tensor_types.txt) for the recipe.

This quant is deliberately larger than total resident memory as the ngram is expected to be mmapped to internal NVME. If you want a fully resident solution, I recommend instead just using https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
