---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
license: apache-2.0
tags:
- unsloth
- imatrix
- quantization
- llama.cpp
- qwen3.8
- qwen
- ubergarm
- 16GB
- 12GB
---

# Qwen3.8-27B (GGUF target for 16GB VRAM)
- This repository provides GGUF quantizations for **Qwen3.8-27B** optimized using **ZB-ZipBrain**, a custom mixed-precision quantization methodology that optimizes LLM tensor bit allocation using rate-distortion marginal cost combined with importance matrix calibration. It automatically identifies Pareto-optimal BPW "sweet spots" to maximize model quality while fitting precise VRAM and memory footprints.
- Specifically optimized to fit mainstream GPUs within a 16GB VRAM budget at around 4 BPW, and even 12GB VRAM cards at around 3 BPW.
- For filenames marked with **v5**, I combined ZB + Pelicanmaxxing for visual evaluation, iteratively tuning until the output reached the most stable quality before locking it in.


![image](https://cdn-uploads.huggingface.co/production/uploads/6954f858965adc3d768ce0d8/yVHec0_YPgFQpakk4toBh.png)

## Benchmark & Evaluation Results

### EvalPlus Benchmark Results  
- HumanEval: 164 tasks
- MBPP: 378 tasks
- Scores are pass@1, no-thinking mode , kvcache ctk q4_0, ctv q4_0
- “—” indicates data not provided.

| Quantization              | HumanEval | HumanEval+ | MBPP  | MBPP+ | Note |
|---------------------------|-----------|------------|-------|-------|------|
| [ZB4.00-MIN-v5.1-IQ4_XS](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF/blob/main/Qwen3.8-27B-ZB4.00-MIN-v5.1-IQ4_XS.gguf)   | 0.945 | **0.921**   | 0.897 | **0.780** | 👑💀 |
| Qwen3.8-27B-IQ4_NL       | **0.951**    | 0.915      | 0.902 | 0.778 | bartowski |
| [ZB3.88-MIN-v5-IQ3_M_XL](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF/blob/main/Qwen3.8-27B-ZB3.88-MIN-v5-IQ3_M_XL.gguf)   | 0.945     | 0.915      | **0.91** | 0.775 | — |
| ~~*ZB4.00-MIN-v5-IQ4_XS*~~ | ~~0.927~~   | ~~0.902~~      | —     | —     |  ~~*oldver*~~ |
| [GSQ-RCO-BuffedMod](https://huggingface.co/tooltd/Qwen3.8-27B-GSQ-RCO-BuffedMod-GGUF/blob/main/Qwen3.8-27B-GSQ-RCO-BuffedMod-IQ3_S_XL-mtp-IQ4XS.gguf) | 0.921     | 0.890      | 0.897 | 0.772 | ISTA+Mod |
| Qwen3.8-27B-Ridge-3.7bpw | 0.933     | 0.896      | 0.902 | 0.765 | empero-ai |
| [ZB3.73-MIN-v5.1-IQ3_M_L](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF/blob/main/Qwen3.8-27B-ZB3.73-MIN-v5.1-IQ3_M_L.gguf)  | 0.927     | 0.896      | 0.881 | 0.757	| — |
| ~~*ZB3.70-MIN-v4-IQ3_M_L*~~ | ~~0.915~~   | ~~0.896~~     | ~~0.873~~ | ~~0.754~~| ~~*oldver*~~  |
| [ZB3.00bpw-IQ3_XXS](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF/blob/main/Qwen3.8-27B-ZB3.00bpw-IQ3_XXS.gguf)        | 0.927     | 0.884      | 0.865 | 0.743	| — |
| UD3-IQ4_XS               | 0.890     | 0.866      | 0.897 | 0.751 | unsloth |
| UD3-Q3_K_XL              | 0.823     | 0.805      | 0.881 | 0.751 | unsloth |




### Comprehensive Comparison Table

Command llama-perplexity.exe -f /wikitext-2-raw/wiki.test.raw --kl-divergence --kl-divergence-base q38f16baseline.kld -ngl 99 -m model.gguf

All models were evaluated against the **BF16 baseline** (`Mean PPL = 6.950493`) using standard Perplexity (PPL) and KL Divergence metrics.

| Label                     | Provider     | Size (GB) | Mean KLD  | Same Top-p (%) | Mean PPL  |
|---------------------------|--------------|-----------|-----------|----------------|-----------|
| UD-Q8_K_XL                | unsloth old  | 29.30     | 0.000850  | 98.970%        | 6.953800  |
| UD-Q6_K_XL                | unsloth old  | 24.14     | 0.001380  | 98.520%        | 6.953600  |
| Q6_K                      | unsloth old  | 21.31     | 0.002290  | 97.860%        | 6.950700  |
| Q5_K_M                    | unsloth old  | 18.47     | 0.006220  | 96.700%        | 6.974200  |
| UD-Q4_K_XL                | unsloth old  | 16.69     | 0.008606  | 96.091%        | 6.979220  |
| ZB4.97-GOD-IQ4_XS         | ZB-GOD       | 15.82     | 0.012249  | 95.337%        | 7.004243  |
| UD3-Q4_K_S                | unsloth UD3  | 14.30     | 0.013652  | 95.149%        | 6.969514  |
| Autoround-Q4_K_M          | Autoround    | 15.66     | 0.014657  | 94.859%        | 6.950294  |
| ZB4.65-PRO-IQ4_XS         | ZB-PRO       | 14.81     | 0.015466  | 94.766%        | 7.017278  |
| Q4_K_M                    | unsloth old  | 15.93     | 0.015490  | 94.650%        | 6.956100  |
| ZB4.60-PRO-IQ4_XS         | ZB-PRO       | 14.65     | 0.016162  | 94.668%        | 7.030895  |
| ZB4.55-PRO-IQ4_XS         | ZB-PRO       | 14.49     | 0.016647  | 94.613%        | 7.032263  |
| IQ4_NL                    | bartowski    | 15.20     | 0.018427  | 94.230%        | 7.006472  |
| IQ4_XS                    | unsloth old  | 14.63     | 0.018652  | 94.270%        | 7.012695  |
| UD3-IQ4_XS              | unsloth UD3  | 13.27     | 0.018772  | 93.975%        | 7.004732  |
| ZB4.48-STD-IQ4_XS         | ZB-STD       | 14.26     | 0.018892  | 94.199%        | 7.050096  |
| Q4_K_S                    | unsloth old  | 15.01     | 0.018921  | 94.235%        | 6.966826  |
| IQ4_XS-i1                 | mradermacher | 14.26     | 0.019271  | 94.141%        | 7.012810  |
| Q4_K_S-i1                 | mradermacher | 14.74     | 0.019805  | 93.996%        | 6.989551  |
| ZB4.36-STD-IQ4_XS         | ZB-STD       | 13.88     | 0.020556  | 93.951%        | 7.054811  |
| ZB4.36-STD-v4-IQ4_XS      | ZB-STD| 13.88 | 0.021552| 93.886%    | **6.993211**|
| Q4_0-AutoRound-Code       | webhie       | 14.64     | 0.026586  | 92.970%        | 7.067142  |
| ZB4.14-MIN-IQ4_XS         | ZB-MIN       | 13.19     | 0.029334  | 92.799%        | 7.045689  |
| **⭐[ZB4.00-MIN-v5.1-IQ4_XS](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF/blob/main/Qwen3.8-27B-ZB4.00-MIN-v5.1-IQ4_XS.gguf)**  | ZB-MIN    | **12.74** | **0.033810**| **92.309%**    | 7.106519|
| ~~*ZB4.00-MIN-v5-IQ4_XS*~~  | ZB-MIN    | 12.79 | 0.034577| 92.277%    | **7.090583**|
| **⭐[ZB3.88-MIN-v5-IQ3_M_XL](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF/blob/main/Qwen3.8-27B-ZB3.88-MIN-v5-IQ3_M_XL.gguf)**  | ZB-MIN    | **12.34** | **0.042164**| **91.452%**    | **7.132594**|
| IQ4_XS-3.84bpw | byteshape | 12.18 |   0.049513  | 90.989%  | 7.158078 |
| **⭐[ZB3.73-MIN-v5.1-IQ3_M_L](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF/blob/main/Qwen3.8-27B-ZB3.73-MIN-v5.1-IQ3_M_L.gguf)**    | ZB-MIN | **11.88**  | **0.048117**  | **90.759%** | 7.199937  |
| **⭐[GSQ-RCO-BuffedMod](https://huggingface.co/tooltd/Qwen3.8-27B-GSQ-RCO-BuffedMod-GGUF/blob/main/Qwen3.8-27B-GSQ-RCO-BuffedMod-IQ3_S_XL-mtp-IQ4XS.gguf)** | ISTA+Mod | **11.49** | 0.051738  | 90.535%  | **7.032142** |
| GSQ-RCO-IQ3_S| ISTA | 11.29 |   0.055475  | 89.657%  | 7.062697 |
| ~~*ZB3.70-MIN-v4-IQ3_M_L*~~ | ZB-MIN    | 11.82     | 0.052972  | 90.363%        | 7.160063  |
| IQ4_XS-Smaller_3.96       | jrell        | 12.61     | 0.055499  | 90.090%        | 7.252766  |
| Ridge-3.7bpw              | empero-ai    | 11.73     | 0.118430  | 85.907%        | 7.547496  |
| [ZB3.0BPW-IQ3_XXS](https://huggingface.co/tooltd/Qwen3.8-27B-IQ4-XS-16GB-VRAM-GGUF/blob/main/Qwen3.8-27B-ZB3.00bpw-IQ3_XXS.gguf)       | ZB-MIN       | 9.62         | **0.120503**| **85.320%**    | **7.474669**|



**Update: Aug 20, 2026**
- The newly released Unsloth Dynamic v3 is truly the best value for performance right now.
- My ZB is just an experiment, feel free to check it out for fun :)

**Update: Aug 22, 2026**
- Quant release ZB-4.00 BPW runs cleanly on *16GB VRAM with MTP support and up to 95K context length.*

**Update: Aug 24, 2026**
- **ZBv3-4.00BPW** update maintaining size with KLD and Same top-p performs slightly better. ZBv2-4.00BPW -> ZBv3-4.00BPW: output weights were bumped from Q5_K to Q6_K. Testing shows sharper, more consistent outputs and better one-shot performance. => Go with ZBv3-4.00BPW.
- **ZB3.7-MIN** ⚔️ **empero-ai/Qwen3.8-27B-Ridge.** 🤣
- I have just updated **empero-ai/Qwen3.8-27B-Ridge.** benchmarks; comparing my ZB 3.7bpw metrics, it looks like it easily beats down Qwen3.8-27B-Ridge

**Update: Aug 26, 2026**
- **ZB3.7-MIN-v4** update PPL slightly better

**Update: Aug 28, 2026**
- All use *Q6_K* for *output.weight*
- Please prioritize **later development versions**, I have removed older ones because I was not satisfied with them.
- Quant release 4.00bpw & 3.88bpw v5 (v5 = ZB + Pelicanmaxxing and visually check for other aspects of stability. )
- **ZB3.88-MIN-v5-IQ3_M_XL** ⚔️ **UD3-Q3_K_XL** 😎 If anyone has used this pair, please let me know what you think.
- **3.0BPW-IQ3_XXS**: New release size only 9.62 GB 🙀 with metric comparable to Ridge—3.7 bpw.

**Update: Sep 5, 2026**
- **ZB4.00-MIN-v5.1-IQ4_XS** This version has been updated so that all tensors are ≥ IQ3_XXS , previous version contained some IQ2_S tensors. Quality is slightly improved, new file size saves 50MB.

**Update: Sep 8, 2026**
- **ZB3.73-MIN-v5.1-IQ3_M_L** Added new 3.73 bpw. Updated all tensors ≥ IQ3_XXS, quality is slightly improved. Change for ZB3.70-MIN-v4-IQ3_M_L.
- Added HumanEval, MBPP benchmark
  
**Recommended Settings:** Set `reasoning_effort` to **medium**.
At this BPW level, it delivers much more stable outputs and fits well in agentic workflows. 
You can also use the default settings for higher quality, though it will take longer.

`
llama-server 
-m models/qwen38/Qwen3.8-27B-ZB4.00-MIN-IQ4_XS.gguf  
-mm models/qwen38/Qwen3.8-27B-mmproj-BF16.gguf 
--host 0.0.0.0 
--port 8080 
--temp 1 
--top-p 0.95 
--top-k 20 
--min-p 0.00 
--reasoning-preserve 
-ctk q4_0 -ctv q4_0 -fa on 
--ubatch-size 384 --batch-size 384  
--no-mmproj-offload  
--spec-type draft-mtp,ngram-mod 
--spec-draft-n-max 2
--spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 24 --spec-ngram-mod-n-max 32 
-ngl 99 -t 7 
--ctx-size 95000 
-np 1 
--load-mode mlock 
--image-min-tokens 1024 
--image-max-tokens 2048 
--chat-template-kwargs '{\"reasoning_effort\": \"medium\"}'
`


## ZB Tiers & Recommendations

* **ZB-GOD :** God. A singularity appears. Reaches `0.012249` Mean KLD and `95.34%` top-probability match.
* **ZB-PRO :** Pro. For 16 GB VRAM GPUs with offloading on CPU. Balances quality output with substantial size savings.
* **ZB-STD :** Standard. Similar to other standard IQ4_XS models currently available.
* **ZB-MIN :** Minimal. Optimal footprint for tight 16 GB memory setups, allowing headroom for longer context windows

---
## Credits & Acknowledgements

* **Base Model:** [Qwen3.8 27B](https://huggingface.co/Qwen) by Alibaba Cloud / Qwen Team.
* **BF16 Base GGUF:** Provided by [Unsloth AI](https://huggingface.co/unsloth).
* **Importance Matrix (imatrix):** Generated and curated by [ubergarm](https://huggingface.co/ubergarm).
* **Inference & Quantization Framework:** [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov and contributors.