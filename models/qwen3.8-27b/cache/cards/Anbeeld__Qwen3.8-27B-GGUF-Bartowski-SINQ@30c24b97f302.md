---
quantized_by: Anbeeld
pipeline_tag: image-text-to-text
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
---

# BeeLlama.cpp SINQ Quantizations of Qwen 3.8 27B

These GGUFs follow the [Bartowski Qwen3.8-27B quantization recipes](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) 1:1, with SINQ applied during quantization. SINQ is supported only in [BeeLlama.cpp v0.5.0](https://github.com/Anbeeld/beellama.cpp). In the matched evaluation against the BF16 reference, each selected SINQ quant has lower mean KLD than its corresponding stock Bartowski quantization.

## Comparison with Bartowski, Unsloth, and ISTA

[![Mean KLD and model-size comparison of stock Bartowski, selected Bartowski recipe plus SINQ, Unsloth, and ISTA quantizations.](stock-vs-sinq-vs-unsloth-vs-ista.png)](stock-vs-sinq-vs-unsloth-vs-ista.png)

[![Mean KLD and model-size comparison of stock Bartowski, Bartowski plus SINQ, and Unsloth Q5_K_S, Q6_K, and Q8_0 quants.](stock-vs-sinq-vs-unsloth-5-6-8-bit.png)](stock-vs-sinq-vs-unsloth-5-6-8-bit.png)

## KLD benchmark results for all recipes

The table reports one selected SINQ variant per recipe. Measurements use the full Wikitext-2 test corpus and the BF16 Qwen3.8-27B model as the reference. KLD is KL(BF16 reference || selected quantization), evaluated over 297,193 aligned next-token targets with a 65,536-token (64K) context, BF16 K/V cache, batch 2048, and microbatch 512. Top-1 agreement and PPL use the same targets.

| Recipe | Size (GB) | Width | Strength | Mean KLD | Median KLD | P90 KLD | P95 KLD | P99 KLD | P99.9 KLD | Max KLD | Top-1 | PPL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [IQ3_XXS](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ3_XXS-SINQ-w64-s1.gguf) | 12.326 | 64 | 1 | 0.088562 | 0.024939 | 0.135741 | 0.244027 | 0.945144 | 9.322315 | 32.415568 | 89.420% | 6.2762 |
| [Q3_K_S](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q3_K_S-SINQ-w16-s1.gguf) | 12.746 | 16 | 1 | 0.081897 | 0.021106 | 0.113599 | 0.204130 | 0.805402 | 10.549262 | 34.638633 | 90.207% | 6.3037 |
| [Q3_K_M](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q3_K_M-SINQ-w16-s1.gguf) | 13.410 | 16 | 1 | 0.069874 | 0.017942 | 0.095148 | 0.168726 | 0.658497 | 9.585826 | 32.733418 | 91.014% | 6.1224 |
| [Q3_K_L](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q3_K_L-SINQ-w32-s0.5.gguf) | 14.129 | 32 | 0.5 | 0.058107 | 0.013971 | 0.076026 | 0.134979 | 0.518894 | 7.628747 | 33.732112 | 91.878% | 6.1459 |
| [IQ4_XS](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ4_XS-SINQ-w16-s1.gguf) | 15.482 | 16 | 1 | 0.029491 | 0.005315 | 0.029945 | 0.054288 | 0.221963 | 3.993512 | 35.428775 | 94.921% | 6.0682 |
| [Q4_K_S](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q4_K_S-SINQ-w16-s0.5.gguf) | 16.369 | 16 | 0.5 | 0.028394 | 0.004844 | 0.027155 | 0.048714 | 0.197507 | 4.574282 | 34.413405 | 95.054% | 6.0809 |
| [Q4_K_M](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q4_K_M-SINQ-w16-s0.75.gguf) | 17.448 | 16 | 0.75 | 0.024358 | 0.004080 | 0.023149 | 0.041734 | 0.165366 | 3.075818 | 32.903476 | 95.452% | 6.1228 |
| [Q4_K_L](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q4_K_L-SINQ-w64-s0.75.gguf) | 18.827 | 64 | 0.75 | 0.021131 | 0.003266 | 0.018261 | 0.032319 | 0.122655 | 2.750248 | 35.510508 | 95.892% | 6.0752 |
| [Q5_K_S](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q5_K_S-SINQ-w32-s0.75.gguf) | 19.573 | 32 | 0.75 | 0.016288 | 0.001800 | 0.009740 | 0.017285 | 0.069539 | 2.143039 | 33.459243 | 96.948% | 6.0516 |
| [Q5_K_M](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q5_K_M-SINQ-w16-s0.75.gguf) | 20.930 | 16 | 0.75 | 0.013948 | 0.001591 | 0.008384 | 0.014786 | 0.057504 | 1.856113 | 33.619304 | 97.070% | 6.0552 |
| [Q5_K_L](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q5_K_L-SINQ-w16-s0.5.gguf) | 21.543 | 16 | 0.5 | 0.013524 | 0.001467 | 0.007810 | 0.013697 | 0.053507 | 1.773615 | 35.820641 | 97.246% | 6.0512 |
| [Q6_K_S](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q6_K_S-SINQ-w16-s0.5.gguf) | 22.863 | 16 | 0.5 | 0.007257 | 0.000666 | 0.003046 | 0.005111 | 0.019263 | 0.548761 | 26.986455 | 98.103% | 6.0482 |
| [Q6_K](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q6_K-SINQ-w16-s0.75.gguf) | 23.866 | 16 | 0.75 | 0.006030 | 0.000617 | 0.002692 | 0.004427 | 0.016257 | 0.421356 | 30.661686 | 98.179% | 6.0426 |
| [Q6_K_L](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q6_K_L-SINQ-w16-s0.75.gguf) | 24.965 | 16 | 0.75 | 0.006042 | 0.000531 | 0.002254 | 0.003576 | 0.012050 | 0.336644 | 32.301142 | 98.305% | 6.0604 |
| [Q8_0](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q8_0-SINQ-w16-s1.gguf) | 29.122 | 16 | 1 | 0.004727 | 0.000219 | 0.001000 | 0.001695 | 0.006794 | 0.309545 | 27.576670 | 98.874% | 6.0549 |
| [IQ2_XXS](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ2_XXS-SINQ-w16-s1.gguf) | 8.887 | 16 | 1 | 0.292708 | 0.116543 | 0.569978 | 0.977161 | 3.240525 | 13.060524 | 36.840487 | 79.728% | 7.2898 |
| [IQ2_XS](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ2_XS-SINQ-w16-s1.gguf) | 9.094 | 16 | 1 | 0.258898 | 0.098707 | 0.475378 | 0.830525 | 2.945753 | 14.316963 | 30.677830 | 81.241% | 6.7813 |
| [IQ2_S](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ2_S-SINQ-w16-s1.gguf) | 9.690 | 16 | 1 | 0.222801 | 0.080479 | 0.411583 | 0.720749 | 2.589578 | 12.608808 | 33.349592 | 82.538% | 6.6952 |
| [IQ2_M](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ2_M-SINQ-w32-s0.5.gguf) | 10.528 | 32 | 0.5 | 0.184837 | 0.054905 | 0.297730 | 0.537112 | 2.252287 | 15.059180 | 33.968627 | 84.948% | 6.9293 |
| [Q2_K](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q2_K-SINQ-w16-s1.gguf) | 10.827 | 16 | 1 | 0.172353 | 0.056458 | 0.298685 | 0.526776 | 1.963626 | 12.511502 | 33.813399 | 84.839% | 6.5684 |
| [IQ3_XS](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ3_XS-SINQ-w32-s0.5.gguf) | 12.805 | 32 | 0.5 | 0.114714 | 0.024009 | 0.143946 | 0.279934 | 1.471601 | 13.973704 | 31.178352 | 89.276% | 6.7982 |
| [Q2_K_L](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q2_K_L-SINQ-w32-s0.5.gguf) | 13.087 | 32 | 0.5 | 0.171494 | 0.052233 | 0.274217 | 0.496292 | 2.017182 | 15.131550 | 33.728927 | 85.281% | 6.7191 |
| [IQ3_M](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ3_M-SINQ-w32-s1.gguf) | 14.866 | 32 | 1 | 0.068673 | 0.014856 | 0.084981 | 0.157961 | 0.704197 | 9.328254 | 31.191362 | 91.450% | 6.4201 |
| [Q4_0](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q4_0-SINQ-w32-s1.gguf) | 16.355 | 32 | 1 | 0.038434 | 0.007841 | 0.041399 | 0.072272 | 0.280283 | 6.417007 | 31.511324 | 93.903% | 6.1088 |
| [Q3_K_XL](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q3_K_XL-SINQ-w32-s0.75.gguf) | 16.398 | 32 | 0.75 | 0.058178 | 0.014490 | 0.075041 | 0.133006 | 0.517923 | 8.096432 | 33.240114 | 91.809% | 6.1885 |
| [IQ4_NL](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-IQ4_NL-SINQ-w32-s0.75.gguf) | 17.448 | 32 | 0.75 | 0.026042 | 0.004506 | 0.025145 | 0.045236 | 0.181013 | 3.596341 | 33.789907 | 95.289% | 6.0553 |
| [Q4_1](https://huggingface.co/Anbeeld/Qwen3.8-27B-GGUF-Bartowski-SINQ/blob/main/Qwen3.8-27B-Q4_1-SINQ-w16-s0.75.gguf) | 17.831 | 16 | 0.75 | 0.030065 | 0.005632 | 0.029811 | 0.052624 | 0.207985 | 4.845014 | 30.746869 | 94.684% | 6.1510 |
