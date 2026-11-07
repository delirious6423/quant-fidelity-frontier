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
| `Qwen3.8-27B-DASHQ-Q2_K_XL.gguf` | Q2_K_XL | 9.80 GB | 2.87 |

Text only: the vision tower is not included.

## Usage

```bash
llama-cli -m Qwen3.8-27B-DASHQ-Q2_K_XL.gguf -ngl 99 -c 8192
```

## License

Inherits the license of the base model ([Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)).
