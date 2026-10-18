---
license: mit
language:
- en
- zh
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
- gguf
- deepseek_v4
- speculative-decoding
- dspark
- llama.cpp
---

# DeepSeek V4 Flash 0731 DSpark Drafter GGUF

This repository contains a locally requantized **DSpark speculative-decoding drafter** for DeepSeek V4 Flash 0731. It is not a standalone chat model and must be loaded together with a compatible DeepSeek V4 Flash 0731 main model.

## File

| File | Size | SHA-256 |
|---|---:|---|
| `dspark-DeepSeek-V4-Flash-0731-EXPERT-Q2_K.gguf` | 6.963 GB (6.485 GiB) | `4a05eaa5d94a73c00160bf988b567b55f0b32e2b6f322714eaefe46b828b218c` |

## Quantization

The source drafter was `dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf` from [`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF).

This is a mixed quantization produced with `llama-quantize`:

- `blk.[0-2].ffn_(down|gate|up)_exps.weight`: `Q2_K`
- `markov_w1.weight`, `markov_w2.weight`: `F16`
- output baseline: `Q8_0`
- `--allow-requantize --pure`

```bash
llama-quantize \
  --allow-requantize \
  --pure \
  --tensor-type 'blk\.[0-2]\.ffn_(down|gate|up)_exps\.weight=q2_k' \
  --tensor-type 'markov_w[12]\.weight=f16' \
  dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf \
  dspark-DeepSeek-V4-Flash-0731-EXPERT-Q2_K.gguf \
  Q8_0
```

## llama.cpp usage

Use a llama.cpp build with DeepSeek V4 Flash and DSpark support.

```bash
llama-server \
  -m /models/main/DeepSeek-V4-Flash-0731-UD-Q2_K_XL-00001-of-00003.gguf \
  --spec-draft-model /models/draft/dspark-DeepSeek-V4-Flash-0731-EXPERT-Q2_K.gguf \
  --spec-type draft-dspark \
  --spec-draft-ngl all \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0.0 \
  --ctx-size 524288 \
  --n-gpu-layers all \
  --flash-attn on
```

The local validation configuration used `DeepSeek-V4-Flash-0731-UD-Q2_K_XL` as the main model and loaded this file through `--spec-draft-model`.

## License and attribution

The upstream model and source GGUF repository declare the MIT license. See:

- [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- [`unsloth/DeepSeek-V4-Flash-0731-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF)

