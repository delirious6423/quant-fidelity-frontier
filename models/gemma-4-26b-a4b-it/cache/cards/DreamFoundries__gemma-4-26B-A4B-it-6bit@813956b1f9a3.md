---
library_name: mlx
base_model: google/gemma-4-26B-A4B-it
license: apache-2.0
pipeline_tag: text-generation
tags:
- mlx
- mlx-lm
- quantized
- 6-bit
- base_model:google/gemma-4-26B-A4B-it
---

[![Open in MLXHub](https://mlxhub.app/assets/badge-open-in-mlxhub.svg)](https://mlxhub.app/open-model?repo=DreamFoundries/gemma-4-26B-A4B-it-6bit)

# gemma-4-26B-A4B-it MLX 6-bit

This repository contains an MLX-LM conversion of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it).

## Conversion Details

- Original model: `google/gemma-4-26B-A4B-it`
- Model family: Gemma4
- Source model type: `gemma4`
- Model size: 26,544,131,376 parameters
- Quantization: MLX-LM affine quantization
- Bits: 6-bit
- Group size: 64
- Local MLX folder size at upload time: 19.13 GiB
- Local safetensors weight size at upload time: 19.10 GiB

This Gemma conversion follows the MLX-LM Gemma 4 shared-KV topology and uses non-strict checkpoint loading so extra HF tensors outside that topology are discarded during conversion.

## Usage

```bash
mlx_lm.generate --model DreamFoundries/gemma-4-26B-A4B-it-6bit --prompt "Hello" --max-tokens 64
```

## Benchmarks

No comparative benchmarks have been run yet. The repository does not currently provide quality, speed, memory, or benchmark comparisons against the original weights or other quantizations.

## License

This is a converted/quantized derivative of the original model. Please refer to the original model repository for the upstream license and usage terms: https://huggingface.co/google/gemma-4-26B-A4B-it

---

[![Download MLXHub](https://mlxhub.app/assets/badge-download-mlxhub.svg)](https://apps.apple.com/app/apple-store/id6766485144?pt=121945436&ct=HuggingFace&mt=8)
