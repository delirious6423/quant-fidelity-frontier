---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: glyd
pipeline_tag: text-generation
tags:
- glyd
- kestrel
- lossy
---

# Qwen3.8-27B · Glyd kestrel

**22.0 GB, 59% smaller than bf16 (53.8 GB).** Glyd's kestrel level of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (commit [`1d4bf0f2`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)): about 6.5 bits per weight, quantized once from the original bf16 weights. Not lossless: KL divergence from bf16 is 0.00477 (lower is closer).

```sh
curl -LsSf https://getglyd.com/install.sh | sh
glyd run Qwen/Qwen3.8-27B:kestrel
```

## Compared

| | Size | Smaller than bf16 | KL vs bf16 | RTX 4090 |
| :-- | --: | --: | --: | --: |
| bf16 (original) | 53.8 GB | – | 0 | – |
| Qwen FP8 | 29.5 GB | 45% | 0.00465 | – |
| [penguin](https://huggingface.co/glyd/Qwen3.8-27B-penguin) (lossless) | 37.0 GB | 31% | 0 | – |
| **kestrel (this repo)** | **22.0 GB** | **59%** | **0.00477** | **40 tok/s** |
| [swift](https://huggingface.co/glyd/Qwen3.8-27B-swift) | 18.6 GB | 65% | 0.0139 | 48 tok/s |

Size: the files you download. KL: how far the model's next-token probabilities move from bf16, measured on WikiText-2.

## Speed and memory

| GPU | Tokens/s | First token | Memory at 4k | 32k context |
| :-- | --: | --: | --: | :-- |
| RTX 4090 | 40 | 464 ms | 24.1 GB | no |
| L40S | 31 | 446 ms | 24.2 GB | yes |
| RTX A6000 | 26 | 728 ms | 24.0 GB | yes |

Measured 2026-10-09 with glyd 0.29.4: one GPU each, a 4k-token context; memory is the GPU memory in use.

Fits a 32k context on: L40S, RTX A6000.

## Benchmarks

| Benchmark | this level | bf16 |
| :-- | --: | --: |
| GSM8K | 97.3% | 96.7% |

Run through `glyd serve` with Inspect AI 0.3.278 (inspect_evals 0.24.0); the bf16 column is the source model in bf16 served by vLLM, with the same harness and settings. GSM8K: 10-shot, numeric match, non-thinking, 1319 questions.

## Good to know

- Runs on Glyd's engine (`glyd run`): Linux with an NVIDIA GPU, driver 580 or newer. Not for vLLM or transformers yet.
- Text only: the model's vision part is not included.
- The Hub's "8-bit precision" and "Model size" labels count packed bytes: this checkpoint is about 6.5 bits per weight, from a model with 26,895,998,464 parameters.

## License

Weights: Apache-2.0, from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Glyd's engine, which runs them: BUSL-1.1, free for personal and non-commercial use on your own computers; commercial use needs a license. [getglyd.com](https://getglyd.com)
