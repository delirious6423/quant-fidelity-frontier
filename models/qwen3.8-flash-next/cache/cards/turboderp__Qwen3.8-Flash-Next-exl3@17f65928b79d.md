---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
quantized_by: turboderp
tags:
- exl3
---

EXL3 quants of [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)

# Plain quants:

[2.05 bits per weight / H4, NG4](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/2.05bpw_h4_ng4)    
[3.05 bits per weight / H5, NG5](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.05bpw_h5_ng5)    
[4.05 bits per weight / H6, NG6](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.05bpw_h6_ng6)    
[5.05 bits per weight / H6, NG6](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/5.05bpw_h6_ng6)    
[6.05 bits per weight / H6, NG6](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/6.05bpw_h6_ng6)    

# Self-calibrated quants

[1.66 bits per weight / H4, NG3](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/1.66bpw_h4_ng3) ¹    
[2.18 bits per weight / H6, NG8](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/2.18bpw_h6_ng8)    
[2.68 bits per weight / H6, NG8](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/2.68bpw_h6_ng8)    
[3.18 bits per weight / H6, NG8](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.18bpw_h6_ng8)    
[3.68 bits per weight / H6, NG8](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/3.68bpw_h6_ng8)    
[4.18 bits per weight / H6, NG8](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3/tree/4.18bpw_h6_ng8)    

Note `ngram_embedding.safetensors` is swappable between quants.

¹ 1.66 bpw quant: mostly K=1.5 expert tensors, 4 bpw lm_head and input embeddings, 3 bpw n-gram table (swappable, 
usually streamed from disk anyway) and 4 bpw vision tower, fits on one headless RTX 5090 without CPU offloading 
along with MTP head and vision, for 200+ t/s in agentic workloads (with limited context though: 64k tokens at K5V4 
cache quantization.)

Some limited benchmarks (more to come, probably):

| Model | GPQA Diamond |
|:--|--:|
| SC 1.66 bpw / H4, NG3 | 86.4% |
| SC 2.18 bpw | 89.4% |
| SC 2.68 bpw | 91.4% |
| SC 3.18 bpw | 91.4% |
| SC 3.68 bpw | 91.9% |
| GSQ-RCO IQ2_XS, 2.40 bpw | 87.4% |
| GSQ-RCO IQ3_XXS, 2.90 bpw | 91.4% |
| Original | 91.7% |

![qb_kld](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/e3PezJzzDxLGuTs6Am4rg.png)
![qb_ppl](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/gz1lo7P1z9YCURBcogBQZ.png)
![qb_kld_hist_combined](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/Ut4S8nt-ceHFbBnnfPBPp.png)
