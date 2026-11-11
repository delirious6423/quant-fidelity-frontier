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

# Qwen3.8-27B, MLX Q4 KQuant

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in MLX Q4, built with **KQuant**: the same format as the lmstudio-community 4-bit pack and closer to the original model than that pack. It runs on [Kapitsa](https://github.com/nace-ai/kapitsa), the local inference engine from [Nace.AI](https://www.nace.ai/) for Apple silicon Macs.

Other sizes: [Qwen3.5-4B](https://huggingface.co/nace-ai/Qwen3.5-4B-MLX-Q4-KQuant), [Qwen3.5-9B](https://huggingface.co/nace-ai/Qwen3.5-9B-MLX-Q4-KQuant), [Qwen3.6-35B-A3B](https://huggingface.co/nace-ai/Qwen3.6-35B-A3B-MLX-Q4-KQuant), [Qwen3.8-Flash-Next](https://huggingface.co/nace-ai/Qwen3.8-Flash-Next-MLX-Q4-KQuant).


## Quickstart

### mlx-lm (Apple silicon)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the text model with [mlx-lm](https://github.com/ml-explore/mlx-lm):

```bash
uvx --from mlx-lm==0.32.2 mlx_lm.generate \
  --model nace-ai/Qwen3.8-27B-MLX-Q4-KQuant \
  --prompt 'Explain why the sky is blue.' --max-tokens 512 \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
  --chat-template-config '{"enable_thinking":false}'
```

This command runs non-thinking text generation, not image input. The mlx-lm CLI does not expose presence or repetition penalties.

### Kapitsa (Apple silicon)

Install [Kapitsa](https://github.com/nace-ai/kapitsa#install), then serve this pack locally:

```bash
kapitsa serve --model nace-ai/Qwen3.8-27B-MLX-Q4-KQuant --private
```

Open the local address printed by the server.

## How close it is

<picture>
<source media="(prefers-color-scheme: dark)" srcset="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q4-KQuant/resolve/main/assets/kquant-dark.svg">
<img src="https://huggingface.co/nace-ai/Qwen3.8-27B-MLX-Q4-KQuant/resolve/main/assets/kquant-light.svg" alt="Qwen3.8-27B: KL to BF16 against language-model size in GB. Red diamonds: KQuant. Grey circles: lmstudio-community 4-bit, mlx-community 4-bit and mlx-works oQ4e-mtp. Each point is labelled with its KL.">
</picture>

The red diamond is KQuant, this repo; grey circles are [lmstudio-community 4-bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit), [mlx-community 4-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) and [mlx-works oQ4e-mtp](https://huggingface.co/mlx-works/Qwen3.8-27B-oQ4e-mtp). Lower KL is closer to the original BF16 model. This is the Qwen3.8-27B panel of the KQuant launch chart. The mlx-works size includes its nested MTP weights.

<details>
<summary>Comparison data and exact file identities</summary>

### MLX public packs: original-text corpus

GB counts weight-file downloads including auxiliary weights, not the whole repository; configs and tokenizers are excluded. Lower KL and higher top-token agreement are closer to the reference model.

| Pack | GB | KL 95% CI | Top-token agreement |
| :--- | ---: | :--- | ---: |
| KQuant (ours) | 16.293 | 0.034936 [0.031795, 0.038471] | 92.87% |
| lmstudio-community MLX-4bit | 16.055 | 0.054384 [0.049780, 0.059621] | 91.21% |
| mlx-community 4bit | 16.055 | 0.053761 [0.049395, 0.058618] | 91.15% |
| mlx-works oQ4e-mtp | 16.972 | 0.035848 [0.033524, 0.038315] | 92.59% |

<details>
<summary>Comparison provenance</summary>

Measured 2026-11-11.

Engine: MLX.

- `nace-ai/Qwen3.8-27B-MLX-Q4-KQuant` at `c122d52252bd9b5db296d671cdc9c1f61f6e52f2`.
  File identities: unavailable in the report.
  Weight-file download (including auxiliary weights): 16293476273 bytes (16.293476273 GB).
  Scored language-model weights: 15132802048 bytes (15.132802048 GB), not a download-size measure.
- `lmstudio-community/Qwen3.8-27B-MLX-4bit` at `6067b15cf581666a4aecf6af3afaba4bb5efc20c`.
  File identities: unavailable in the report.
  Weight-file download (including auxiliary weights): 16054541599 bytes (16.054541599 GB).
  Scored language-model weights: 15132802048 bytes (15.132802048 GB), not a download-size measure.
- `mlx-community/Qwen3.8-27B-4bit` at `b935172e4bb5dcd0543774a34748fb92422e3027`.
  File identities: unavailable in the report.
  Weight-file download (including auxiliary weights): 16054541349 bytes (16.054541349 GB).
  Scored language-model weights: 15132802048 bytes (15.132802048 GB), not a download-size measure.
- `mlx-works/Qwen3.8-27B-oQ4e-mtp` at `1c0609b20321c1faff8294be5ea88c656cedaa59`.
  File identities: unavailable in the report.
  Weight-file download (including auxiliary weights): 16971702038 bytes (16.971702038 GB).
  Scored language-model weights: 16049958912 bytes (16.049958912 GB), not a download-size measure.

</details>

</details>

## Quality

| Test | lmstudio-community pack | KQuant (this repo) |
| --- | ---: | ---: |
| KL from the bf16 model (lower is closer) | 0.054 | **0.035** |
| Agrees with the bf16 model on the next token | 91.2% | **92.9%** |
| GSM8K, all 1,319 questions | 0.950 | 0.953 |

Question by question on GSM8K, 13 only the lmstudio-community pack gets right and 17 only KQuant: no difference we can measure (McNemar z 0.73).

KL, GSM8K and speed all compare against one lmstudio-community pack: [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) at `6067b15`.

Quality was measured on text workloads; the vision tower is included but was not evaluated.

<details>
<summary>How measured</summary>

| | |
| --- | --- |
| lmstudio-community pack | [lmstudio-community/Qwen3.8-27B-MLX-4bit](https://huggingface.co/lmstudio-community/Qwen3.8-27B-MLX-4bit) at `6067b15`, the pack `kapitsa serve --model 27b` serves |
| KL and agreement | next-token distributions over 32,704 tokens of WikiText-2, GSM8K chats and HumanEval code, against Qwen/Qwen3.8-27B @ 37561816 (bf16) |
| GSM8K | zero-shot, greedy, thinking off, the last number in the answer |
| What was scored | the text weights of both packs (15.13 GB each); the repos also carry the vision tower, which neither run exercised |
| Tokenizer | this repo ships the tokenizer of Qwen3.8-27B; every run here used the catalog pack's tokenizer, whose ids are identical on English, code, math, CJK, emoji and RTL text; Devanagari and Thai were not measured |
| Scored with | mlx-lm 0.32.2 on a Modal H100 (mlx with CUDA), on the text weights; every run is in the [results record](https://github.com/nace-ai/kapitsa/blob/main/bench/results/2026-11-10-m5-pro-kquant-27b/README.md) |

</details>

## Speed

| MacBook Pro M5 Pro, 48 GB | lmstudio-community pack (tok/s) | **KQuant (tok/s)** |
| --- | ---: | ---: |
| Writes, temperature 0.6 | 80.0 | 72.5 |
| Writes, greedy | 82.8 | 91.0 |
| Reads a 4K prompt | 719 | 735 |

These are measured rows (three runs per pack for each mode), not a speed claim. Sampled runs vary widely on this Mac: KQuant 76.7, 72.2, 68.5 tok/s at temperature 0.6, lmstudio-community pack 77.6, 78.5, 83.8. Two of the three KQuant legs ran before `mtplx_runtime.json` was in the pack directory; the third leg has it, as this repo ships it.

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

## Why KQuant

KQuant is Nace's quantization framework for MLX packs. An MLX pack ships only if it comes out closer to the original model than the lmstudio-community pack.

On this size KQuant takes the distance from the original bf16 model from 0.054 (lmstudio-community's pack) to 0.035 (KL, lower is closer) and moves GSM8K from 0.950 to 0.953, at the same size and in the same format. On this Mac it decodes slower when sampling and faster when greedy, so we claim no speed for it.

**How we checked it**

- **Token by token against the original.** The pack's next-token predictions are compared with the bf16 model's over 32K tokens of prose, math chats and code.
- **All 1,319 GSM8K questions**, compared with the lmstudio-community pack question by question, so a difference inside the noise is called noise.
- **Timed on a real M5 Pro 48 GB**, packs interleaved leg by leg. A leg is thrown out if a process that was not there at the start used the machine during it, and the results record lists every one we threw out.

**Why it is closer**

- **Calibrated on real text.** The pack is tuned to how the model behaves on real text, not only to the numbers in its weights.
- **A search per group of weights.** Every group gets its own best fit, instead of being rounded the same way as every other.

Use an MLX runtime with support for Qwen3.8-27B. The pinned mlx-lm command is above.

## Credits

The MTP head (`mtp.safetensors`) and `mtplx_runtime.json` are from [Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed](https://huggingface.co/Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed) at `1d5087d2` (MTPLX, apache-2.0), unchanged. Tokenizer and chat template as in [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Thanks to Youssofal.

## License

apache-2.0, as the base model [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
