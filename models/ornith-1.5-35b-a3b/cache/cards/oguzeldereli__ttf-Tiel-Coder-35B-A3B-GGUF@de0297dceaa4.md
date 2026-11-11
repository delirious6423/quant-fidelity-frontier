---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
library_name: gguf
tags:
  - ternary
  - 1.58-bit
  - gguf
  - llama.cpp
  - moe
  - code
  - ttf
---

# ttf-Tiel-Coder-35B-A3B-GGUF

Tiel-Coder-35B-A3B (Ornith-1.5-35B-A3B with the Sharp chat template, as in
[peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP](https://huggingface.co/peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF-MTP))
with every routed expert and dense projection **ternary** ({-1, 0, +1} times row and column scales), in one 8.74 GB
GGUF file. The vision tower and the MTP head are not included.

It was made in two steps:
1. **Conversion** from the BF16 Ornith-1.5-35B-A3B, layer by layer on one GPU:
   - ternary GPTQ on the centred input covariance (256 calibration windows of 2048 tokens);
   - row and column scales, the input mean folded into a bias;
   - a rank-16 ternary adapter on each linear.
2. **Recovery training**: 800 steps (about 26M tokens) of TTF ([ttf-ternary](https://github.com/oguzeldereli/ttf-ternary)),
   distilled from the BF16 model (cached top-64 teacher distribution plus a tail bucket) on a post-training mix
   (reasoning, code, math, tool use, chat) plus multilingual web text.

## Quality

llama.cpp `llama-server`, the embedded Sharp chat template with thinking off, greedy decoding, at most 512 new tokens.
The same harness and prompts for all three files:

| model | size | GSM8K (first 200) | HumanEval (164) | generated tokens per GSM8K item |
|---|---|---|---|---|
| **this model (TQ1_P)** | 8.74 GB | 0.800 ± 0.028 | 0.652 ± 0.037 | 152 |
| Tiel-Coder Q2_K_XL | 12.68 GB | 0.920 ± 0.019 | 0.854 ± 0.028 | 103 |
| Tiel-Coder IQ4_XS | 18.12 GB | 0.945 ± 0.016 | 0.896 ± 0.024 | 83 |

This file is 31% smaller than the Q2_K_XL quant and clearly behind it (−0.120 on GSM8K, −0.202 on HumanEval), and
its answers are about 50% longer. It is a short recovery run (800 steps); treat it as a research release, not as a
replacement for the 2-bit quants.

## What is in the file

| part | size | storage |
|---|---|---|
| ternary weights (34.0G trits) | 7.09 GB | TQ1_0, 1.69 bits per weight |
| token embedding + output head | 0.57 GB | Q4_K |
| expert adapters (rank 16) | 0.74 GB | exact: up-factors in Q4_0, down-factors in TQ1_0 |
| expert row / column scales and biases | 0.23 GB | F16 |
| router, norms, rest | 0.09 GB | BF16 / F32 |

The adapter up-factors are ternary times a row scale, which Q4_0 holds exactly once the column scale is moved into
the down-factor's block scale (the rank is padded to 32 with zeros).

An earlier upload (9.83 GB) stored the embedding and head in Q8_0 and the adapters and scales in F16 / F32. This
file has the same weights; on held-out text its perplexity is 3.941 ± 0.063 against 3.918 ± 0.062 for the earlier
file (the adapter repacking is exact; the difference comes from the Q4_K embedding and head and the F16 rounding of the
scales), and GSM8K / HumanEval moved from 0.805 / 0.659 to
0.800 / 0.652, within noise.

## Running it

The ternary tensors need llama.cpp with the TTF patches, built by [ttf-home](https://github.com/oguzeldereli/ttf-home)
(`qwen35moe` architecture, the `--q38` engine):

```bash
git clone https://github.com/oguzeldereli/ttf-home && cd ttf-home
pip install -U pip "setuptools>=68" && pip install -e . huggingface_hub
ttf engine build --q38

hf download oguzeldereli/ttf-Tiel-Coder-35B-A3B-GGUF --local-dir models/ttf-tiel
ttf chat models/ttf-tiel/ttf-Tiel-Coder-35B-A3B-TQ1_P.gguf
ttf serve models/ttf-tiel/ttf-Tiel-Coder-35B-A3B-TQ1_P.gguf --port 8080   # OpenAI-compatible API
```

The benchmarks above ran on an RTX 3090 Ti (24 GB) with every layer on the GPU (`-ngl 99`, 16k context, 4 parallel
slots). On smaller cards, llama.cpp's `--fit` keeps some experts in host memory.

## Files

| file | size |
|---|---|
| `ttf-Tiel-Coder-35B-A3B-TQ1_P.gguf` | 8.74 GB |
| `SHA256SUMS` | checksum |

## License

MIT, as the base model ([ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B)). The
inference patches in ttf-home have their own license.
