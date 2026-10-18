---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
tags:
- deepseek_v4
- conversational
- ik_llama.cpp
- llama.cpp
---

# DeepSeek-V4-Flash-0731-GGUF

This is a GGUF model for DeepSeek-V4-Flash, created with the convert_hf_to_gguf.py script, which does 2 things:
* Repacks routed experts tensors in FP4 to MXFP4.
* Quantizes other tensors in FP8 to Q8_0.

The GGUF model can be considered as lossless when compared to the original weights in safetensors.

It works fine in both ik_llama.cpp and llama.cpp. It is faster with ik_llama.cpp, but support for SWA is still in progress.
