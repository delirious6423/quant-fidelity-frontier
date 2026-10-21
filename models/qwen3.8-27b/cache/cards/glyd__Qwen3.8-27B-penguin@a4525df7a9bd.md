---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: glyd
pipeline_tag: text-generation
tags:
- glyd
- penguin
- lossless
---

# Qwen3.8-27B · Glyd penguin

**37.0 GB, 31% smaller than bf16 (53.8 GB).** Glyd's lossless compression of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (commit [`1d4bf0f2`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)): every weight decodes to its bf16 value, bit for bit.

```sh
curl -LsSf https://getglyd.com/install.sh | sh
glyd run Qwen/Qwen3.8-27B:penguin
```

```python
import glyd
model = glyd.from_pretrained("glyd/Qwen3.8-27B-penguin")   # pip install "glyd[gpu]"
```

## Compared

| | Size | Smaller than bf16 | KL vs bf16 | RTX 4090 |
| :-- | --: | --: | --: | --: |
| bf16 (original) | 53.8 GB | – | 0 | – |
| Qwen FP8 | 29.5 GB | 45% | – | – |
| **penguin (lossless, this repo)** | **37.0 GB** | **31%** | **0** | **–** |
| [kestrel](https://huggingface.co/glyd/Qwen3.8-27B-kestrel) | 22.0 GB | 59% | 0.00477 | 40 tok/s |
| [swift](https://huggingface.co/glyd/Qwen3.8-27B-swift) | 18.6 GB | 65% | 0.0139 | 48 tok/s |

Size: the files you download. KL: how far the model's next-token probabilities move from bf16, measured on WikiText-2.

## Good to know

- Linux with an NVIDIA GPU (Ampere or later), driver 580 or newer; glyd 0.29.4 or later.
- Text only: the model's vision part is not included.
- The Hub's "8-bit precision" and "Model size" labels count packed bytes: this checkpoint stores about 11.0 bits per weight, from a model with 26,895,998,464 parameters.

## License

Weights: Apache-2.0, from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Glyd's engine, which runs them: BUSL-1.1, free for personal and non-commercial use on your own computers; commercial use needs a license. [getglyd.com](https://getglyd.com)
