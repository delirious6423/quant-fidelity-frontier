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