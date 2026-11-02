---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - quantized
  - mtp
---

# Qwen3.8-27B EXL3 5.0 bpw

![Qwen3.8-27B EXL3 5.0 bpw](assets/banner.png)

EXL3 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (**27B parameters**) for [ExLlamaV3](https://github.com/turboderp-org/exllamav3) / [TabbyAPI](https://github.com/theroyallab/tabbyAPI). All credit for the model goes to the Qwen team.

## vs the Unsloth GGUFs: lower error, ~1.37x faster

| | [UD-IQ4_XS GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) | [UD-Q5_K_M GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (llama.cpp) | **This EXL3 5.0 bpw (ExLlamaV3)** |
|---|---|---|---|
| **KL divergence vs BF16, mean** | 0.0224 | 0.0069 | **0.0056** (20 % lower than Q5_K_M) |
| KL divergence vs BF16, median | 0.0069 | 0.0014 | **0.0012** |
| KL divergence vs BF16, 99th percentile | 0.232 | 0.061 | **0.045** |
| Same top token as BF16 | 94.0 % | 97.0 % | **97.5 %** |
| **Decode speed** (same GPUs) | | 36.5 tok/s | **50.0 tok/s** (1.37x) |
| File size | 14.3 GB (no vision) | 19.8 GB (no vision) | 19 GB (vision included) |

### How KL divergence was measured

- Reference: the original BF16 weights, run with ExLlamaV3 (layer streaming).
- Text: WikiText-2 raw test, 20 rows of 2048 tokens. KLD is scored on the second half of each row (20,460 tokens), the same positions llama.cpp uses.
- The BF16 log-probabilities were written in llama.cpp's `--kl-divergence-base` format, so all quants are compared against **the exact same reference and the exact same tokens**:
  - GGUF: `llama-perplexity --kl-divergence-base <bf16 file> --kl-divergence -c 2048`
  - EXL3: ExLlamaV3 `eval/model_diff.py` (A = EXL3, B = BF16), restricted to the same positions.
- "Same top token" for the EXL3 is computed over all positions of the rows (llama.cpp's figure uses the second half only), so treat that line as indicative.

### How speed was measured

RTX 5060 Ti 16 GB + RTX 3070 8 GB, both models split across the two GPUs, MTP speculative decoding (2 draft tokens for the GGUF, 3 for the EXL3), temperature 1, reasoning on. Same script for both: 4 prompts (3 coding, 1 explanation), 600 tokens each, run twice, average decode speed after the first token.

| | Runs |
|---|---|
| GGUF UD-Q5_K_M | 34.7, 33.7, 43.5, 32.0, 35.7, 35.4, 40.7, 36.1 |
| EXL3 5.0 bpw | 60.2, 42.8, 50.2, 40.9, 63.6, 41.0, 61.3, 39.8 |

The EXL3 run used a tensor-parallel split tuned for this GPU pair (41/59, attention on the 16 GB card) and a 64k context with an 8-bit cache. With a default split, expect somewhat less.

## Quantization

| Part | Bits |
|---|---|
| Decoder layers | 5.0 bpw |
| Output head | 6 bpw |
| Vision tower | 6 bpw |
| MTP head | 5 bpw |
| Embeddings | 16 bit |

Settings: `-b 5.0 -hb 6 -vb 6 -mb 5 --out_scales always`, default calibration. Size: ~19 GB.

The MTP head and the vision tower are included.

## Fitting in 24 GB

With 24 GB of VRAM in total (16 + 8 GB here), the weights take ~19 GB, which leaves room for about 64k tokens of context with an 8-bit cache. A 131k context did not fit on this setup.

## License

Same as the base model: Apache 2.0.
