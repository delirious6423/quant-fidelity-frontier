---
library_name: mlx
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
pipeline_tag: text-generation
base_model: google/gemma-4-26B-A4B-it
tags:
- mlx
- apple-silicon
- quantized
- awq
- imatrix
- mixed-precision
- gemma
---

# gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6

`google/gemma-4-26B-A4B-it` quantized to **4.674 bpw** for Apple Silicon (MLX). Built by elevating the top sensitivity tier from 4-bit to **6-bit AFFINE** on top of the [ship recipe (`hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq`)](https://huggingface.co/hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq).

| | Value |
| --- | --- |
| **Footprint** | 14 GB |
| **bpw** | 4.674 |
| **Recipe** | imatrix3plus + AWQ Option B + HIGH=6 (top 35% sensitivity → 6-bit) |
| **Min RAM** | 24 GB (32 GB recommended for ≥8K context) |
| **Tulu-3 PPL (mean of 3 seeds)** | **62.679** (vs ship 66.86, Δ −4.18) |

## Positioning in the 3-tier family

| Tier | Repo | Size | bpw | Tulu PPL | Specialty |
| --- | --- | --- | --- | --- | --- |
| **Standard** | [`hsng95/...-awq`](https://huggingface.co/hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq) | 12 GB | 3.916 | 66.86 | wikitext / TQA / GSM8K winner |
| **Quality (this)** | `hsng95/...-awq-high6` | 14 GB | 4.674 | 62.68 | **MMLU / ARC / KMMLU winner — most balanced** |
| **Flagship-KR** | [`hsng95/...-awq-high6-top40`](https://huggingface.co/hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6-top40) | 15 GB | 5.057 | 57.85 | Korean chat / HAERAE winner |

## Why HIGH=6?

The top sensitivity-ranked 35 % of tensors get elevated from 4-bit to 6-bit AFFINE. At 6-bit, quantization noise becomes small enough that the AWQ rescaling baked in from the 4-bit-noise calibration becomes a **near-no-op** rather than a mismatch — yielding **pure precision gain** instead of calibration drift.

Empirically the bit-tier ridge is non-monotonic and HIGH=6 sits on the optimum:

| HIGH bits | Tulu PPL | Δ vs ship |
| --- | --- | --- |
| 4 (ship) | 66.86 | — |
| 5 | 70.28 | **+3.42 (uncanny valley)** |
| **6** | **62.68** | **−4.18** |
| 8 | 64.13 | −2.73 |

## Multi-angle eval (vs ship)

Tested across 11 measurement axes including downstream tasks:

| Metric | Ship | HIGH=6 (this) | Winner |
| --- | --- | --- | --- |
| Tulu-3 PPL | 66.86 | **62.68** | HIGH=6 (−4.18) |
| Wikitext-2 PPL | **798** | 801 | ship (tie within noise) |
| MMLU (1860 samples) | 0.320 | **0.325** | HIGH=6 |
| ARC-Challenge | 0.345 | **0.365** | HIGH=6 |
| KMMLU | 0.207 | **0.216** | HIGH=6 |
| HellaSwag | **0.460** | 0.460 | tied |
| TruthfulQA | **0.395** | 0.380 | ship |
| GSM8K (no-shot CoT) | **0.267** | 0.033 | ship ⚠️ |
| HAERAE | 0.690 | 0.712 | HIGH=6 |

**Caveat**: GSM8K shows a sharp regression on no-shot chain-of-thought generation despite winning multi-choice MMLU/ARC. If you primarily target math with no-shot CoT, use the **ship** tier. For balanced knowledge + Korean + English chat, this tier is the best default.

## Usage (MLX)

```bash
pip install -U mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Write a short joke about saving RAM."},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256))
```

> ⚠️ Gemma 4 26B-A4B is instruction-tuned. **Always use `apply_chat_template`** — raw prompts produce garbage on every Gemma 4 quant (3-bit, 4-bit, AWQ alike).

## Sampling

Per the upstream Gemma 4 recommendation:

* `temperature=1.0`
* `top_p=0.95`
* `top_k=64`

## Quantization details

* **Source**: `google/gemma-4-26B-A4B-it` (bf16)
* **imatrix corpus**: 38 sequences, ~68 % Korean / 37 % conversational (lumen `imatrix3plus`)
* **AWQ**: Option B (`mean_sq` proxy), `mlp_down` absorption groups filtered out, embed_tokens at AFFINE 8-bit
* **Bit allocation**: top 35 % sensitivity → **6-bit AFFINE**, middle 30 % → 4-bit, bottom 35 % → 3-bit
* **Group size**: 64
* **Build tool**: `lumen-rs/scripts/quant/streaming_awq_quantize_strict.py --high-bits 6`

## Acknowledgments

Built on top of [`mlx-community/gemma-4-26b-a4b-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-4bit) (DWQ source) for imatrix capture, and the upstream [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) for bf16 source weights.

## License

Apache 2.0 (Gemma 4 license terms apply, see [Gemma license](https://ai.google.dev/gemma/docs/gemma_4_license)).
