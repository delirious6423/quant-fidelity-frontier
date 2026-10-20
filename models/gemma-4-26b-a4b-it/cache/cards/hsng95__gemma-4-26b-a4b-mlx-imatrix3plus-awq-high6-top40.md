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
- korean
language:
- ko
- en
---

# gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6-top40

`google/gemma-4-26B-A4B-it` quantized to **5.057 bpw** for Apple Silicon (MLX) — the **Korean-chat flagship** of the lumen Gemma 4 family. Built by combining HIGH=6 (top sensitivity tier → 6-bit) with **top4_fraction=0.40** (40 % of tensors elevated to top tier instead of the default 35 %).

| | Value |
| --- | --- |
| **Footprint** | 15 GB |
| **bpw** | 5.057 |
| **Recipe** | imatrix3plus + AWQ Option B + HIGH=6 + top4_fraction=0.40 |
| **Min RAM** | 32 GB (36 GB recommended for ≥8K context) |
| **Tulu-3 PPL (mean of 3 seeds)** | **57.85** (vs ship 66.86, Δ **−9.01**) |

## Positioning in the 3-tier family

| Tier | Repo | Size | bpw | Tulu PPL | Specialty |
| --- | --- | --- | --- | --- | --- |
| **Standard** | [`hsng95/...-awq`](https://huggingface.co/hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq) | 12 GB | 3.916 | 66.86 | wikitext / TQA / GSM8K winner |
| **Quality** | [`hsng95/...-awq-high6`](https://huggingface.co/hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6) | 14 GB | 4.674 | 62.68 | MMLU / ARC / KMMLU winner — most balanced |
| **Flagship-KR (this)** | `hsng95/...-awq-high6-top40` | 15 GB | 5.057 | **57.85** | **HAERAE / Korean chat / tulu PPL winner** |

## External benchmark context

This is the **lowest-PPL MLX-loadable Gemma 4 26B-A4B quantization currently available**:

| External MLX build | Tulu PPL (seed 7) |
| --- | --- |
| `nightmedia/gemma-4-26B-A4B-it-mxfp4-mlx` | 81.74 |
| **this model** (seed 7) | **43.79** (**−37.95**) |

NVIDIA NVFP4 (TensorRT-only) and `cyankiwi/...-AWQ-4bit` (compressed-tensors, vLLM-only) formats are not MLX-loadable and therefore not directly comparable on Apple Silicon.

## Multi-angle eval (vs ship & Quality tier)

| Metric | Ship | HIGH=6 | **This** | Winner |
| --- | --- | --- | --- | --- |
| Tulu-3 PPL | 66.86 | 62.68 | **57.85** | **this (−9.01)** |
| Wikitext-2 PPL | **798** | 801 | 801 | ship (cross-corpus inconsistent) |
| HAERAE (Korean) | 0.690 | 0.712 | **0.752** | **this** |
| MMLU | 0.320 | **0.325** | 0.310 | HIGH=6 |
| ARC-Challenge | 0.345 | **0.365** | 0.330 | HIGH=6 |
| HellaSwag | **0.460** | 0.460 | 0.410 | ship / HIGH=6 |
| TruthfulQA | **0.395** | 0.380 | 0.365 | ship |
| GSM8K (no-shot CoT) | **0.267** | 0.033 | 0.167 | ship |

**When to choose this tier**: Korean conversational workloads, instruction-following on `tulu`-like calibration distributions, long-form Korean generation. It maximises HAERAE and tulu PPL at the cost of broad-MMLU and English-CoT-math (where ship/HIGH=6 remain better).

## Why top4_fraction=0.40 on HIGH=6?

After confirming HIGH=6 as the optimal top tier (−4.18 ppl), we re-swept top4_fraction under the new tier. Result: sharp non-monotonic peak at **0.40**:

| top4_fraction | Tulu PPL | Δ vs ship |
| --- | --- | --- |
| 0.30 | 71.64 | +4.78 |
| 0.35 (HIGH=6 baseline) | 62.68 | −4.18 |
| **0.40** | **57.85** | **−9.01** |
| 0.42 | 58.81 | −8.05 (tied within σ) |
| 0.45 | 64.90 | −1.96 |
| 0.50 | 71.41 | +4.55 |

The 35 → 40 % threshold contains a discrete tensor cluster that delivers a quality jump independent of the bit tier itself. AWQ re-search at this top4 setting gave 58/60 bit-identical scales to ship → confirms AWQ-recipe coupling is robust to ±10 % top4 shift on the HIGH=6 base.

## Usage (MLX)

```bash
pip install -U mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6-top40")

messages = [
    {"role": "system", "content": "당신은 도움이 되는 한국어 비서입니다."},
    {"role": "user", "content": "양자화가 무엇인지 한 문단으로 설명해줘."},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
print(generate(model, tokenizer, prompt=prompt, max_tokens=512))
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
* **Bit allocation**: top **40 %** sensitivity → 6-bit AFFINE, middle 25 % → 4-bit, bottom 35 % → 3-bit
* **Group size**: 64
* **Build tool**: `lumen-rs/scripts/quant/streaming_awq_quantize_strict.py --high-bits 6 --top4-fraction 0.40`

## Acknowledgments

Built on top of [`mlx-community/gemma-4-26b-a4b-4bit`](https://huggingface.co/mlx-community/gemma-4-26b-a4b-4bit) (DWQ source) for imatrix capture, and the upstream [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) for bf16 source weights.

## License

Apache 2.0 (Gemma 4 license terms apply, see [Gemma license](https://ai.google.dev/gemma/docs/gemma_4_license)).
