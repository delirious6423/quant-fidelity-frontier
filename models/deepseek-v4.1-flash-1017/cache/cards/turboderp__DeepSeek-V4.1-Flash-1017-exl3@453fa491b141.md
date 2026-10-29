---
license: mit
pipeline_tag: image-text-to-text
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
quantized_by: turboderp
tags:
- exl3
---

EXL3 quants of [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)

⚠️ Requires ExLlamaV3 v1.6.6 (or `dev` branch)

# SC quants:

[2.25 bits per weight](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3/tree/2.25bpw)    

*(more quants to follow, along with benchmarks)*

If you get corrupted output on short prompts, it is likely just a failure mode of the model itself. 
See [here](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/33).