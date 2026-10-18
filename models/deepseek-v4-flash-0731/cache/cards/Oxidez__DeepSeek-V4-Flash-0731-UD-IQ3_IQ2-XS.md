---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
library_name: gguf
tags:
  - deepseek
  - deepseek-v4
  - gguf
  - llama.cpp
  - dspark
  - speculative-decoding
  - quantization
  - iq3-xxs
  - iq2-xxs
  - blackwell
pipeline_tag: text-generation
---

# DeepSeek-V4-Flash-0731 – Custom IQ3_XXS / IQ2_XXS GGUF

This repository contains a custom mixed-quantization GGUF of
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

The model was created from the original DeepSeek-V4-Flash-0731 GGUF and
uses an experimental mixed quantization recipe designed to significantly
reduce memory usage and increase inference speed on NVIDIA Blackwell GPUs.

> **Experimental quantization:** This model has not been extensively
> evaluated for quality. Please benchmark it against the original model
> before using it for production workloads.

---

## Base Model

**DeepSeek-V4-Flash-0731**

Original model:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731

License: MIT

This repository contains a derived/quantized GGUF representation of the
original model.

---

## Quantization Recipe

The model uses the following custom tensor-type recipe:

```text
^token_embd\.weight$=Q8_0
^output\.weight$=Q8_0
^blk\.[0-9]+\.(attn|indexer).*\.weight$=Q6_K
^blk\.[0-9]+\.ffn_(gate|up|down)_shexp\.weight$=Q6_K
^.*hc_.*\.weight$=Q8_0
^blk\.(0|1|2|3|4|5|6|7|8|9|34|35|36|37|38|39|40|41|42)\.ffn_(gate|up|down)_exps\.weight$=IQ3_XXS
^blk\.(10|11|12|13|14|15|16|17|18|19|20|21|22|23|24|25|26|27|28|29|30|31|32|33)\.ffn_(gate|up|down)_exps\.weight$=IQ2_XXS