---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: image-text-to-text
tags:
- mlx
base_model: google/gemma-4-26B-A4B-it
---

[Gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) optimized for MLX.
This quant supports image input and requires a vision-enabled MLX server.

**EDIT April 5, 2026:** Updated model for better performance / speed tradeoff, plus full precision vision embedding. Added benchmarks.

# Usage

```sh
# Start server at http://localhost:8080/chat/completions
uvx --from mlx-vlm --with torchvision \
  mlx_vlm.server \
  --host 127.0.0.1 \
  --port 8080 \
  --model spicyneuron/Gemma-4-26B-A4B-MLX-4.7bit-vision
```

# Methodology

Quantized using a custom script inspired by Unsloth/AesSedai/ubergarm style mixed-precision GGUFs.
MLX quantization options differ than llama.cpp, but the principles are the same:

- Sensitive layers like MoE routing, attention, and output embeddings get higher precision
- More tolerant layers like MoE experts get lower precision

# Benchmarks

 metric | mlx-community/gemma-4-26b-a4b-it-4bit | unsloth/gemma-4-26b-a4b-it-UD-MLX-4bit | 4.7 bit (this model)
--------|---------|---------|---------
bpw | 4.587 | 4.743 | 4.704
peak mem | 15.312 | 15.804 | 15.681
prompt proc (1024) | 2514.427 | 2508.474 | 2501.617
token gen (512) | 97.214 | 92.890 | 91.860
perplexity | 146.663 ± 0.921 | 211.070 ± 1.378 | 133.539 ± 0.827
hellaswag | 0.525 ± 0.011 | 0.531 ± 0.011 | 0.532 ± 0.011
piqa | 0.72 ± 0.01 | 0.712 ± 0.011 | 0.719 ± 0.01
winogrande | 0.635 ± 0.014 | 0.635 ± 0.014 | 0.639 ± 0.014

- Bits per weight calculated against only the `language_model` weights.
- Perplexity in Gemma 4 was surprisingly high but seemed consistent across my trials.
  Could be a side effect of using `allenai/tulu-3-sft-mixture`.
  Best to interpret it as weaker signal than the other benchmark results.

Tested with:

```
mlx_lm.perplexity --sequence-length 4096 --seed 123
mlx_lm.benchmark --prompt-tokens 1024 --generation-tokens 512 --num-trials 5
mlx_lm.evaluate --tasks hellaswag --seed 123 --num-shots 0 --limit 2000
mlx_lm.evaluate --tasks piqa --seed 123 --num-shots 0 --limit 2000
mlx_lm.evaluate --tasks winogrande --seed 123 --num-shots 0 --limit 2000
```