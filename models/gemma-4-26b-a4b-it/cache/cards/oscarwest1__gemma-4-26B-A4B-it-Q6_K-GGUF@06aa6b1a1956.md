---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
tags:
- gemma4
- gguf
- Q6_K
- llama.cpp
- mixture-of-experts
- instruction-tuned
- multilingual
---

# Gemma 4 26B-A4B Instruct — Q6_K GGUF

This repository contains a Q6_K quantized GGUF
conversion of Google's Gemma 4 26B-A4B instruction-tuned model.

## Model Information

| Property | Value |
|---|---|
| Architecture | Gemma 4 MoE |
| Original model | google/gemma-4-26B-A4B-it |
| Developer | Google DeepMind |
| Conversion and quantization | oscarwest1 |
| Total parameters | Approximately 26B |
| Active parameters | Approximately 4B |
| Source precision | BF16 |
| Format | GGUF |
| Quantization | Q6_K |
| License | Apache License 2.0 |

## Conversion

Converted from the original BF16 Safetensors weights
to BF16 GGUF using llama.cpp, then quantized to Q6_K.

No additional training or fine-tuning was performed.

## Usage

Run with a recent llama.cpp build supporting Gemma 4:

    llama-cli -m gemma-4-26B-A4B-it-Q6_K.gguf -cnv

## License and Attribution

Original model developed by Google DeepMind.

Original model:
https://huggingface.co/google/gemma-4-26B-A4B-it

License:
https://www.apache.org/licenses/LICENSE-2.0

This is an unofficial third-party conversion,
not affiliated with or endorsed by Google.

Users must comply with the applicable license terms
and preserve required notices and attribution.

## Acknowledgments

Thanks to Google DeepMind, Hugging Face,
and the llama.cpp community.

---

I miss you NYZ
