---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- dashq
- quantized
---

![DASH-Q](https://raw.githubusercontent.com/JaeminK/dashq/main/assets/dashq_banner.png)

# gemma-4-26B-A4B-it - DASH-Q 2-bit GGUF

llama.cpp GGUF files of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) quantized with
[DASH-Q](https://github.com/JaeminK/dashq), at several 2-bit-class sizes. Every file uses only standard
llama.cpp tensor types (no tensor above 4 bits) and loads in any recent llama.cpp build.

| File | Type | Size | Bits / weight |
| --- | --- | --- | --- |
| `gemma-4-26B-A4B-it-DASHQ-IQ2_XXS.gguf` | IQ2_XXS | 10.10 GB | 3.20 |
| `gemma-4-26B-A4B-it-DASHQ-IQ2_XS.gguf` | IQ2_XS | 10.60 GB | 3.36 |
| `gemma-4-26B-A4B-it-DASHQ-IQ2_M.gguf` | IQ2_M | 11.11 GB | 3.52 |
| `gemma-4-26B-A4B-it-DASHQ-Q2_K_XL.gguf` | Q2_K_XL | 11.27 GB | 3.57 |

Text only: the vision tower is not included.

## Quality vs. F16 (KL divergence)

Raw-text perplexity is not a meaningful metric for gemma-4-it (the F16 model itself scores in the thousands on WikiText-2 while generating normally),
so quality is reported as KL divergence of the next-token distribution to F16 (`llama-perplexity --kl-divergence`, WikiText-2, 4 x 2048 tokens):

| Type | Model | Size | Mean KLD | Median KLD | Top-1 token agreement |
| --- | --- | --- | --- | --- | --- |
| IQ2_XXS | llama.cpp IQ2_XXS (imatrix) | 9.28 GB | 6.34 | 5.29 | 24.1% |
| IQ2_XXS | unsloth UD-IQ2_XXS | 9.92 GB | 2.27 | 1.02 | 50.8% |
| IQ2_XXS | **DASH-Q IQ2_XXS** | **10.10 GB** | **2.25** | **1.01** | **51.5%** |
| IQ2_XS | llama.cpp IQ2_XS (imatrix) | 9.80 GB | 4.91 | 3.92 | 30.4% |
| IQ2_XS | **DASH-Q IQ2_XS** | **10.60 GB** | **2.16** | **0.87** | **52.1%** |
| IQ2_M | llama.cpp IQ2_M (imatrix) | 10.36 GB | 3.79 | 2.96 | 33.3% |
| IQ2_M | unsloth UD-IQ2_M | 10.01 GB | 2.28 | 1.02 | 51.1% |
| IQ2_M | **DASH-Q IQ2_M** | **11.11 GB** | **1.85** | **0.72** | **55.2%** |
| Q2_K_XL | llama.cpp Q2_K (imatrix) | 10.58 GB | 5.25 | 4.48 | 28.0% |
| Q2_K_XL | **DASH-Q (this file)** | **11.27 GB** | **1.83** | **0.75** | **54.6%** |

## Usage

```bash
llama-cli -m gemma-4-26B-A4B-it-DASHQ-IQ2_M.gguf -ngl 99 -c 8192
```

## License

Inherits the license of the base model ([google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)).
