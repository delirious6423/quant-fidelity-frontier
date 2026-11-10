---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: mlx
tags:
- mlx
- kapitsa
- kquant
- quantized
- apple-silicon
---

# Qwen3.8-27B, MLX 4-bit KQuant

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in MLX 4-bit, built with **KQuant**: the same format and the same size as the standard 4-bit pack (15.13 GB of language-model weights) and closer to the original model. It runs on [Kapitsa](https://github.com/nace-ai/kapitsa), the local inference engine from [Nace.AI](https://www.nace.ai/) for Apple silicon Macs.

Every run and how KQuant builds a pack: [KQuant in the Kapitsa README](https://github.com/nace-ai/kapitsa#kquant). The other size: [Qwen3.5-9B KQuant](https://huggingface.co/nace-ai/Qwen3.5-9B-MLX-4bit-KQuant).

| | KL from bf16 (lower is closer) | GSM8K |
| --- | ---: | ---: |
| Standard | 0.054 | 0.950 |
| **KQuant** | **0.035** | 0.953 |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-4bit-KQuant/resolve/main/assets/kquant-dark.svg">
  <img alt="Horizontal bars for Qwen3.8-27B 4-bit, standard pack against KQuant: distance from the bf16 model 0.054 and 0.035 (lower is closer)." src="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-4bit-KQuant/resolve/main/assets/kquant-light.svg">
</picture>

## Run it

```bash
brew install nace-ai/tap/kapitsa
# chat page and API at http://127.0.0.1:8000
kapitsa serve --model 27b:kquant
```

This needs Kapitsa 0.2.34 or later (or a build from main until 0.2.34 is out).

or with mlx-lm:

```bash
pip install mlx-lm
mlx_lm.generate \
  --model nace-ai/Qwen3.8-27B-MLX-4bit-KQuant \
  --prompt "Write a haiku about Almaty."
```

## Quality

| Test | Standard pack | KQuant (this repo) |
| --- | ---: | ---: |
| KL from the bf16 model (lower is closer) | 0.054 | **0.035** |
| Agrees with the bf16 model on the next token | 91.2% | **92.9%** |
| GSM8K, all 1,319 questions | 0.950 | 0.953 |

Question by question on GSM8K, 13 only the standard pack gets right and 17 only KQuant: no difference we can measure (McNemar z 0.73).

KL, GSM8K and speed all compare against one standard pack: [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) at `6067b15`.

Quality was measured on text workloads; the vision tower is included but was not evaluated.

<details>
<summary>How measured</summary>

| | |
| --- | --- |
| Standard pack | [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) at `6067b15`, the pack `kapitsa serve --model 27b` serves |
| KL and agreement | next-token distributions over 32,704 tokens of WikiText-2, GSM8K chats and HumanEval code, against Qwen/Qwen3.8-27B @ 37561816 (bf16) |
| GSM8K | zero-shot, greedy, thinking off, the last number in the answer |
| What was scored | the text weights of both packs (15.13 GB each); the repos also carry the vision tower, which neither run exercised |
| Tokenizer | this repo ships the tokenizer of Qwen3.8-27B; every run here used the catalog pack's tokenizer, whose ids are identical on English, code, math, CJK, emoji and RTL text; Devanagari and Thai were not measured |
| Scored with | mlx-lm 0.32.2 on a Modal H100 (mlx with CUDA), on the text weights; every run is in the [results record](https://github.com/nace-ai/kapitsa/blob/main/bench/results/2026-11-10-m5-pro-kquant-27b/README.md) |

</details>

## Speed

| MacBook Pro M5 Pro, 48 GB | Standard pack (tok/s) | **KQuant (tok/s)** |
| --- | ---: | ---: |
| Writes, temperature 0.6 | 80.0 | 72.5 |
| Writes, greedy | 82.8 | 91.0 |
| Reads a 4K prompt | 719 | 735 |

These are measured rows (three runs per pack for each mode), not a speed claim. Sampled runs vary widely on this Mac: KQuant 76.7, 72.2, 68.5 tok/s at temperature 0.6, standard pack 77.6, 78.5, 83.8. Two of the three KQuant legs ran before `mtplx_runtime.json` was in the pack directory; the third leg has it, as this repo ships it.

<details>
<summary>How measured</summary>

| | |
| --- | --- |
| Engine | Kapitsa 0.2.32 (`e728524`) with the DFlash2 drafter [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) at `015e7956` for both packs; speeds with another drafter will differ |
| Writes | a 512-token answer, the mean of 6 runs per pack, half at temperature 0.6 and half greedy |
| Reads | a cold prompt of about 4K tokens, the mean of three runs per pack |
| Mac | MacBook Pro M5 Pro, 48 GB, macOS 26.5.2, on AC power |
| Measured | 10 November 2026 |

</details>

## What changes

| Part | Standard pack | KQuant (this repo) |
| --- | --- | --- |
| Scale per group of 64 weights | from min and max | searched over 11 clip ranges |
| Weighting | none | 128 calibration texts |
| Files, format, size | MLX affine 4-bit | the same |

## Credits

The MTP head (`mtp.safetensors`) and `mtplx_runtime.json` are from [Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed) at `1d5087d2` (MTPLX, apache-2.0), unchanged. Tokenizer and chat template as in [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Thanks to Youssofal.

## License

apache-2.0, as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
