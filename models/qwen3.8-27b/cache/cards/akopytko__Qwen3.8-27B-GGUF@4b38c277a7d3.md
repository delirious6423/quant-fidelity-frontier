---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
pipeline_tag: image-text-to-text
tags:
- qwen3.8
---

[Qwen3.8-27B Quant Finder](https://sanmai.github.io/Qwen3.8-27B-Quant-Finder/)


## Released files

<!-- table:released -->

| Quant type | GiB | Mean KLD | Same top-p |
| --- | ---: | ---: | ---: |
| [Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q6_K.gguf) | 20.56 | 0.002047 | 97.9% |
| [75% Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-75-Q6_K.gguf) | 19.69 | 0.002870 | 97.5% |
| [55% Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-55-Q6_K.gguf) | 18.97 | 0.003585 | 97.2% |
| [25% Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-25-Q6_K.gguf) | 17.91 | 0.005609 | 96.6% |
| [50% Q4_K, 30% Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-50-Q4_K-30-Q6_K.gguf) | 15.66 | 0.011348 | 95.3% |
| [80% IQ4_XS, 20% IQ3_S](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-80-IQ4_XS-20-IQ3_S.gguf) | 12.66 | 0.033656 | 91.5% |
| [45% IQ3_S, 20% Q4_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-45-IQ3_S-20-Q4_K.gguf) | 11.67 | 0.045880 | 90.7% |
| [100% IQ3_S](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-100-IQ3_S.gguf) | 10.77 | 0.070473 | 87.5% |
| [25% Q2_K, 20% IQ3_XXS](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-25-Q2_K-20-IQ3_XXS.gguf) | 8.74 | 0.129901 | 84.6% |
| [30% Q2_K, 15% IQ3_XXS](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-30-Q2_K-15-IQ3_XXS.gguf) | 8.61 | 0.153802 | 83.2% |

<!-- /table:released -->

## From best to worst

<!-- table:front -->

| Quant type | GiB | Mean KLD | Same top-p |
| --- | ---: | ---: | ---: |
| Q8_0 | 26.62 | 0.000798 | 98.7% |
| [UD-Q8_K_L](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q8_K_L.gguf) | 25.78 | 0.001008 | 98.6% |
| 65% Q8_0 | 24.05 | 0.001187 | 98.5% |
| [UD-Q6_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q6_K_XL.gguf) | 23.22 | 0.001376 | 98.3% |
| [BW-Q6_K_L](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a/Qwen3.8-27B-Q6_K_L.gguf) | 23.01 | 0.001412 | 98.2% |
| 45% Q8_0 | 22.76 | 0.001458 | 98.3% |
| 30% Q8_0 | 22.19 | 0.001479 | 98.2% |
| [BW-Q6_K](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a/Qwen3.8-27B-Q6_K.gguf) | 21.99 | 0.001606 | 98.1% |
| 20% Q8_0 | 21.51 | 0.001723 | 98.1% |
| [BW-Q6_K_S](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a/Qwen3.8-27B-Q6_K_S.gguf) | 21.05 | 0.001726 | 98.0% |
| [Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-Q6_K.gguf) | 20.56 | 0.002047 | 97.9% |
| [UD-Q6_K](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q6_K.gguf) | 20.14 | 0.002685 | 97.7% |
| 80% Q6_K | 19.85 | 0.002854 | 97.5% |
| [75% Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-75-Q6_K.gguf) | 19.69 | 0.002870 | 97.5% |
| [55% Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-55-Q6_K.gguf) | 18.97 | 0.003585 | 97.2% |
| [UD-Q5_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q5_K_M.gguf) | 18.08 | 0.004621 | 96.9% |
| [25% Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-25-Q6_K.gguf) | 17.91 | 0.005609 | 96.6% |
| [UD-Q5_K_S](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q5_K_S.gguf) | 17.05 | 0.006485 | 96.5% |
| [UD-Q4_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf) | 16.02 | 0.008672 | 95.8% |
| [50% Q4_K, 30% Q6_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-50-Q4_K-30-Q6_K.gguf) | 15.66 | 0.011348 | 95.3% |
| [UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_M.gguf) | 15.00 | 0.012829 | 95.1% |
| [BW-IQ4_XS](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/blob/0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a/Qwen3.8-27B-IQ4_XS.gguf) | 14.18 | 0.016674 | 94.6% |
| [UD-Q4_K_S](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_S.gguf) | 13.97 | 0.017447 | 94.3% |
| 100% IQ4_XS | 13.32 | 0.023571 | 92.8% |
| [UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ4_XS.gguf) | 12.94 | 0.024496 | 93.1% |
| [80% IQ4_XS, 20% IQ3_S](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-80-IQ4_XS-20-IQ3_S.gguf) | 12.66 | 0.033656 | 91.5% |
| 70% IQ4_XS, 25% IQ3_S | 12.42 | 0.036471 | 91.3% |
| [UD-Q3_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q3_K_XL.gguf) | 11.91 | 0.037520 | 91.6% |
| [45% IQ3_S, 20% Q4_K](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-45-IQ3_S-20-Q4_K.gguf) | 11.67 | 0.045880 | 90.7% |
| [GSQ-RCO-IQ3_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/d562806dbafae37109975e970aae91b43e73b440/Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf) | 10.95 | 0.052658 | 89.6% |
| [UD-IQ3_S](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ3_S.gguf) | 10.88 | 0.055409 | 89.7% |
| [100% IQ3_S](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-100-IQ3_S.gguf) | 10.77 | 0.070473 | 87.5% |
| [UD-IQ3_XXS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ3_XXS.gguf) | 9.85 | 0.078996 | 87.8% |
| [GSQ-RCO-IQ3_XXS](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/d562806dbafae37109975e970aae91b43e73b440/Qwen3.8-27B-GSQ-RCO-IQ3_XXS.gguf) | 9.39 | 0.099289 | 86.2% |
| 25% IQ3_XXS, 15% IQ2_S | 9.07 | 0.120202 | 85.0% |
| [UD-Q2_K_XL](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q2_K_XL.gguf) | 8.82 | 0.120409 | 85.1% |
| [25% Q2_K, 20% IQ3_XXS](https://huggingface.co/akopytko/Qwen3.8-27B-GGUF/blob/main/Qwen3.8-27B-25-Q2_K-20-IQ3_XXS.gguf) | 8.74 | 0.129901 | 84.6% |
| [GSQ-RCO-IQ2_S](https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF/blob/d562806dbafae37109975e970aae91b43e73b440/Qwen3.8-27B-GSQ-RCO-IQ2_S.gguf) | 8.61 | 0.137302 | 83.9% |
| [UD-IQ2_S](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ2_S.gguf) | 7.79 | 0.196810 | 81.3% |
| [UD-IQ2_XXS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ2_XXS.gguf) | 6.76 | 0.333410 | 75.6% |
| [UD-IQ1_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ1_M.gguf) | 6.26 | 0.469642 | 71.5% |
| [UD-IQ1_S](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-IQ1_S.gguf) | 5.76 | 0.718030 | 65.3% |
| 100% IQ1_M | 5.49 | 0.890156 | 59.2% |
| 70% Q1_0, 30% Q4_K | 4.52 | 8.366632 | 0.8% |
| 100% Q1_0 | 3.53 | 8.575583 | 0.0% |

<!-- /table:front -->
