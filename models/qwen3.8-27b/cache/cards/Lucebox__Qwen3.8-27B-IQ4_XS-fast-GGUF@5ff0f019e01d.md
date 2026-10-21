---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
- gguf
- qwen3.8
- iq4_xs
- lucebox
- speculative-decoding
- vision
- image-text-to-text
---

# Qwen3.8-27B pure IQ4_XS: +12% speed, a measured quality step below UD

The **speed pick** for [lucebox](https://github.com/Luce-Org/lucebox) serving of Qwen3.8-27B on bandwidth-bound GPUs (measured on one AMD Radeon AI PRO R9700). It is [bartowski's imatrix IQ4_XS](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF) requantized with `llama-quantize --allow-requantize --pure`, which drops the mixed-precision quality hedge so every tensor is IQ4_XS: 13.54 GiB instead of 14.50, and decode reads fewer bytes per token through the fastest kernel path on RDNA4.

**Know the trade before you download.** Measured against a Q8_0 reference (300x512-chunk wikitext-2 paired KL divergence; HumanEval-164 and GSM8K-200 through the serving path, exact-verify speculation so the drafter cannot change outputs):

| | this file (pure) | [Unsloth UD-IQ4_XS](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) (recommended default) |
|---|---|---|
| Size | 13.54 GiB | 13.27 GiB |
| HumanEval decode, DFlash2 block 16 | **235.8 tok/s (257.3 peak)** | 208.1 tok/s |
| End-to-end, same ten prompts | **174.5 tok/s** | 156.2 tok/s |
| Plain decode | **37.4 tok/s** | 32.3 tok/s |
| PPL vs Q8_0 | +1.9% | +0.7% |
| Mean KLD vs Q8_0 | 0.028 | 0.018 |
| Top-1 agreement with Q8_0 | 92.0% | 94.1% |
| HumanEval pass@1 | 144/164 | 151/164 |
| GSM8K (first 200) | 171/200 | 177/200 |

**Pro:** about 12% more tokens per second on every request than the UD file, and the fastest Qwen3.8-27B target we have measured on this class of hardware.

**Con:** the quality cost is small but real and shows up on every instrument: KL divergence rises by half, top-1 agreement gives up two points, and it hands back seven HumanEval and six GSM8K problems against the UD file. If your workload is quality-sensitive, serve UD-IQ4_XS instead; it is what the lucebox quickstart recommends.

Full methodology, exactness verification, and the llama.cpp comparison: [the blog post](https://www.lucebox.com/blog/qwen38-r9700).

## Serve it

```bash
./dflash_server Qwen3.8-27B-IQ4_XS-pure.gguf \
  --draft qwen38-dflash2-q8_0.gguf \
  --draft-block-size 16 --max-ctx 131072 \
  --cache-type-k q8_0 --cache-type-v q8_0
```

The drafter is [z-lab's DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) converted with `server/scripts/convert_dflash_to_gguf.py` from the lucebox repo. Reproduce this file yourself from bartowski's IQ4_XS with:

```bash
llama-quantize --allow-requantize --pure \
  Qwen3.8-27B-IQ4_XS.gguf Qwen3.8-27B-IQ4_XS-pure.gguf IQ4_XS
```

## Vision: images in, with `Qwen3.8-27B-mmproj-Q8_0.gguf`

This repo also carries the image encoder (projector) for lucebox image input. It is [unsloth's BF16 projector](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) with its weight matrices recast to Q8_0 (rows that are not a multiple of 32 stay F16): 629 MB instead of 931 MB, and it encodes a 975-token image in 443 ms instead of 677 ms on the R9700, with the same scores.

```bash
luce_server Qwen3.8-27B-IQ4_XS-pure.gguf --target-device hip:0 \
  --draft qwen38-dflash2-q8_0.gguf --draft-device hip:0 \
  --mmproj Qwen3.8-27B-mmproj-Q8_0.gguf
```

Images go in OpenAI-style `image_url` parts of a chat message, up to four per request. Each image costs one token per 32x32 pixels, 64 to 1,024 tokens. Needs a lucebox build with image input ([PR #722](https://github.com/Luce-Org/lucebox/pull/722)); the full guide is `docs/image-input.md` in the repo.

Measured on one R9700 with this file, the Q8_0 projector and the DFlash2 drafter, thinking off:

| | this file + Q8_0 projector | UD-IQ4_XS + BF16 projector |
|---|---|---|
| AI2D (100 seeded questions) | **90** | 85 |
| ChartQA augmented / human (60 + 60) | **56 / 42** | 55 / 43 |
| Image prompt prefill, average | **0.56 s** | 0.71 s |
| One to four images per request | all correct | all correct |
| Image request decode | 36 tok/s | 31 to 35 tok/s |

Text requests are byte-identical with the projector loaded, at the same speed. Image requests decode without the drafter.

sha256 `47b8f0575185fcf09a6344fa46288f3506b4134cab77fa09ebb5a88c38d665e5`

Credit: [unsloth](https://huggingface.co/unsloth) for the projector; [bartowski](https://huggingface.co/bartowski) for the imatrix calibration this file inherits; [Qwen](https://huggingface.co/Qwen) for the model.

