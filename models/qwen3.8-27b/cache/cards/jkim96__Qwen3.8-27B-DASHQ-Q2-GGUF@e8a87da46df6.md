---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
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

# Qwen3.8-27B - DASH-Q 2-bit GGUF

llama.cpp GGUF files of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized with
[DASH-Q](https://github.com/JaeminK/dashq), at several 2-bit-class sizes. Every file uses only standard
llama.cpp tensor types (no tensor above 4 bits) and loads in any recent llama.cpp build.

| File | Type | Size | Bits / weight |
| --- | --- | --- | --- |
| `Qwen3.8-27B-DASHQ-IQ2_XXS.gguf` | IQ2_XXS | 7.77 GB | 2.28 |
| `Qwen3.8-27B-DASHQ-IQ2_XS.gguf` | IQ2_XS | 8.55 GB | 2.50 |
| `Qwen3.8-27B-DASHQ-IQ2_S.gguf` | IQ2_S | 8.70 GB | 2.55 |
| `Qwen3.8-27B-DASHQ-Q2_K_XL.gguf` | Q2_K_XL | 9.80 GB | 2.87 |

Text only: the vision tower is not included.

## Perplexity (lower is better)

| Type | Model | Size | WikiText-2 | C4 |
| --- | --- | --- | --- | --- |
| IQ2_XXS | llama.cpp IQ2_XXS (imatrix) | 8.67 GB | 7.77 | 12.63 |
| IQ2_XXS | unsloth UD-IQ2_XXS | 7.27 GB | 7.34 | 12.53 |
| IQ2_XXS | **DASH-Q IQ2_XXS** | **7.77 GB** | **7.62** | **13.39** |
| IQ2_XS | llama.cpp IQ2_XS (imatrix) | 9.33 GB | 7.30 | 11.93 |
| IQ2_XS | **DASH-Q IQ2_XS** | **8.55 GB** | **6.75** | **11.61** |
| IQ2_S | unsloth UD-IQ2_S | 8.37 GB | 6.87 | 11.46 |
| IQ2_S | **DASH-Q IQ2_S** | **8.70 GB** | **6.72** | **11.51** |
| IQ2_M | llama.cpp IQ2_M (imatrix) | 10.24 GB | 6.91 | 11.33 |
| Q2_K_XL | **DASH-Q Q2_K_XL** | **9.80 GB** | **6.40** | **10.92** |

`llama-perplexity`, context 2048; WikiText-2 test, C4 validation (256 x 2048 tokens).

## Usage

```bash
llama-cli -m Qwen3.8-27B-DASHQ-Q2_K_XL.gguf -ngl 99 -c 8192
```

## License

Inherits the license of the base model ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)).
