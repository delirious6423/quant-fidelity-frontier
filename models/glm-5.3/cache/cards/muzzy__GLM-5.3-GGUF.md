---
quantized_by: muzzy
pipeline_tag: text-generation
base_model: zai-org/GLM-5.3
base_model_relation: quantized
license: mit
tags:
  - imatrix
  - conversational
  - glm_moe_dsa
  - ik_llama.cpp
language:
  - en
---
This repo contains the same cooked quants that Ubergarm made for GLM 5.1, just for GLM 5.3 now. 

Status 09/25: uploading quants, perplexity coming soon after.

Imatrix calculated with: 

```./build/bin/llama-imatrix -m ~/GLM-5.3/GLM-5.3-00001-of-00115.gguf -f ~/ubergarm-imatrix-calibration-corpus-v02.txt -o ~/ubergarm-imatrix.dat --defer-experts -ngl 999 --layer-similarity --ctx-size 512```

Perplexity (wikitext-103 test split, 289,280 tokens / 565 chunks, ctx 512, ubatch 512, flash attention on, CPU-only 112 threads):

| Quant | PPL | +/- |
|---|---|---|
| IQ1_KT | 4.7472 | 0.02803 |
| IQ2_KS | 3.9404 | 0.02232 |
| IQ2_KL | in progress | - |
| IQ3_KS | - | - |
| IQ4_K | - | - |
| Q4_0_R8 | - | - |
| Q4_K_R4 | - | - |
| Q6_K_R4 | - | - |
| Q8_K_R8 | - | - |
| Q8_0_R8 | - | - |

Calculated with:

```./build/bin/llama-perplexity -m <QUANT>/GLM-5.3-<QUANT>-00001-of-00015.gguf -f wiki.test.txt --ctx-size 512 --ubatch-size 512 -fa on --seed 1337 --threads 112```


