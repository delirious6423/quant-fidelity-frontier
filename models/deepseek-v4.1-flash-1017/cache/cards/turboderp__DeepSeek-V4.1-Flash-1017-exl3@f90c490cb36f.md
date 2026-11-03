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

[1.68 bits per weight](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3/tree/1.68bpw)    
[2.25 bits per weight](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3/tree/2.25bpw)    
[2.68 bits per weight](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3/tree/2.68bpw)    

![qb_kld](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/Af3bmi0w2x6Hi6qfgbxUg.png)
![qb_ppl_vram](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/sgrGQ61hCCKxtiAcJjXR7.png)

quant | AIME-25 | GPQA-D
:--|--:|--:
1.68 bpw | 96.7% | 90.4%
2.25 bpw | 100% | 90.4%
2.68 bpw | 100% | 89.9%
baseline | 100%¹ | 90.2%²

¹ Assumed, given quant scores, couldn't find a source    
² [From OpenRouter](https://openrouter.ai/benchmarks/gpqa-diamond)

*(more quants to follow, along with more benchmarks)*

If you get corrupted output on short prompts, it is likely just a failure mode of the model itself. 
See [here](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/discussions/33).