---
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
language:
  - en
tags:
  - mlx
  - kquant
  - 5-bit
---

# Qwen3.8-27B · KQuant Q5

Also in [nace-ai/Qwen3.8-27B-MLX-KQuant/Q5](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-KQuant/tree/main/Q5), next to Q4.

**2.5× closer** to the original BF16 model than mlx-community 4-bit (KL 0.022 vs 0.054), with 28% more language-model weight (19.43 vs 15.13 GB).

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q5-KQuant/resolve/main/assets/kquant-dark.svg">
<img src="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q5-KQuant/resolve/main/assets/kquant-light.svg" alt="Qwen3.8-27B: KL to BF16 against language-model size in GB. Red diamonds: KQuant Q4, KQuant Q5. Grey dots: lmstudio-community 4-bit, mlx-community 4-bit, mlx-works oQ4e-mtp.">
</picture>

Red: KQuant; this repo is KQuant Q5. Grey: public packs. Lower = closer to BF16.

## Quick start

This is an MLX pack for Apple silicon Macs, 20.7 GB of files. Commands for this pack are still being tested; the [Qwen3.5-9B Q4 card](https://huggingface.co/nace-ai/Qwen3.5-9B-MLX-Q4-KQuant) shows the ones that ran there. The model thinks before it answers by default.

### Recommended settings

From the [Qwen/Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices):

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Thinking off | 0.7 | 0.8 | 20 | 0.0 | 1.5 |

Keep repetition_penalty at 1.0. Give answers room: up to 32,768 output tokens for most tasks with thinking on. A presence_penalty between 0 and 2 curbs endless repetition; high values can mix languages.

## About this pack

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) quantized for 5-bit MLX with **KQuant**: 19.43 GB of language-model weights.

- **Language-model weights:** 19.43 GB (19,425,410,048 bytes)
- **Vision:** The base model's vision tower, pinned by hash: 333 tensors, 921,460,192 bytes
- **Weight files:** 3 files, 20,661,452,485 bytes
- **Runtime:** mlx 0.32.4, mlx-lm 0.32.2, transformers 5.22.2, numpy 2.5.4
- **What was scored:** next-token distributions on original text and on chat and tool-call text, against the bf16 base model

## Quality

| Test | KQuant Q5 (this repo) | Tokens |
| --- | ---: | ---: |
| KL from the bf16 model, original text (lower is closer) | 0.0219 | 32,704 |
| KL from the bf16 model, chat and tool-call text (lower is closer) | 0.1340 | 65,472 |
| Top-1 agreement with the bf16 model, original text | 94.9% | 32,704 |

The other packs in the chart were scored in earlier runs of the same test, which did not record every hash, so the comparison is not strictly paired. This pack ships the base model's tokenizer files, byte for byte, with its chat template.

20 of 20 thinking-off trials placed the call and completed the round trip; none added an extra call (MLX 0.32.4 CUDA; mlx-lm 0.32.2). This pack's size and KL numbers come from its measurement run; the whole run, with its runtime, tokenizer hashes and KL legs, is in [results.json](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q5-KQuant/blob/main/results.json).

<details>
<summary>Exact values, sizes and revisions</summary>

| Pack | Language-model GB | Original-text KL | Pin |
|---|---:|---:|---|
| [KQuant Q5](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q5-KQuant) | 19.425 | 0.0219 | this revision |
| [KQuant Q4](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q4-KQuant) | 15.133 | 0.0349 | [c122d522](https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q4-KQuant/tree/c122d52252bd9b5db296d671cdc9c1f61f6e52f2) |
| [mlx-works oQ4e-mtp](https://huggingface.co/mlx-works/Qwen3.8-27B-oQ4e-mtp) | 16.050 | 0.0358 | [1c0609b2](https://huggingface.co/mlx-works/Qwen3.8-27B-oQ4e-mtp/tree/1c0609b20321c1faff8294be5ea88c656cedaa59) |
| [mlx-community 4-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) | 15.133 | 0.0538 | [b935172e](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit/tree/b935172e4bb5dcd0543774a34748fb92422e3027) |
| [lmstudio-community 4-bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) | 15.133 | 0.0544 | [6067b15c](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit/tree/6067b15cf581666a4aecf6af3afaba4bb5efc20c) |

Exact language-model bytes: KQuant Q5 19,425,410,048 B; KQuant Q4 15,132,802,048 B; mlx-works oQ4e-mtp 16,049,958,912 B; mlx-community 4-bit 15,132,802,048 B; lmstudio-community 4-bit 15,132,802,048 B. Decimal GB; vision excluded.

</details>

## Why KQuant

KQuant is Nace's quantization framework for MLX packs. It tunes each group of weights against how the model behaves on real text, not only against the numbers in its weights, and it writes an ordinary MLX pack: the same files, the same format, nothing new to install.

## Credits

The vision tower and the optional MTP head (`optiq/mtp.safetensors`, at its original path; none of our tests loaded it) are from [mlx-community/Qwen3.8-27B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-OptiQ-4bit/tree/a95c2908fa8ca27cbf29ed86da9a658d14093b63) at `a95c2908` (apache-2.0), unchanged. Tokenizer and chat template as in [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Thanks to mlx-community.

## License

apache-2.0, as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
