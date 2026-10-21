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
---

# gemma-4-26b-a4b-mlx-imatrix3plus-awq

`google/gemma-4-26B-A4B-it` quantized to **3.916 bpw** for Apple Silicon (MLX) using a tuned recipe:
**imatrix-driven mixed 4/3-bit AFFINE** + **AWQ Option B (mean_sq proxy)** + **mlp_down absorption groups dropped** + **embed_tokens at AFFINE 8-bit**.

### Headline result

3-seed mean PPL on `allenai/tulu-3-sft-mixture` (128 samples × 512 tokens, seeds 7 / 42 / 123) **and** decode throughput on Apple M3 Max:

| Build | bpw | Size | 3-seed mean PPL | Δ vs `imatrix3plus` | Decode tok/s |
| --- | --- | --- | --- | --- | --- |
| `mlx-community/gemma-4-26b-a4b-4bit` (uniform 4-bit) | 4.500 | 14 GB | 121.47 (seed=123) | n/a | 71.3 |
| `imatrix3plus` (mixed, no AWQ — internal baseline) | 4.135 | 12.5 GB | 84.19 | — | — |
| **`imatrix3plus-awq` (this model)** | **3.916** | **12 GB** | **67.98** | **−16.22** | **73.0** |

Best of both worlds — better PPL than baseline by 16.2 ppl mean **and** faster decode than uniform 4-bit. Inter-seed variance on this eval is large (baseline ranges 58→117 across seeds), so multi-seed mean is the load-bearing number — see [Multi-seed PPL](#multi-seed-ppl).

## Family (3-tier positioning)

This is the **Standard** tier of the lumen Gemma 4 family. Pick the tier that matches your hardware and workload:

| Tier | Repo | Size | bpw | Tulu PPL | Min RAM | Specialty |
| --- | --- | --- | --- | --- | --- | --- |
| **Standard (this)** | `hsng95/...-awq` | 12 GB | 3.916 | 66.86 | 16 GB | **wikitext / TruthfulQA / GSM8K winner — best for 24 GB Macs** |
| **Quality** | [`hsng95/...-awq-high6`](https://huggingface.co/hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6) | 14 GB | 4.674 | 62.68 | 24 GB | MMLU / ARC / KMMLU winner — most balanced knowledge tasks |
| **Flagship-KR** | [`hsng95/...-awq-high6-top40`](https://huggingface.co/hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq-high6-top40) | 15 GB | 5.057 | 57.85 | 32 GB | HAERAE / Korean chat / tulu PPL winner |

Multi-angle eval across 11 measurement axes (PPL × 4 corpora + 7 downstream tasks) confirms this tier wins on **English narrative (wikitext), factual QA (TruthfulQA), and chain-of-thought math (GSM8K)** — even against the higher-bpw Quality / Flagship tiers. If you target Korean chat or broad knowledge, switch to Quality / Flagship-KR.

## Quick start (mlx-lm)

```bash
pip install -U mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("hsng95/gemma-4-26b-a4b-mlx-imatrix3plus-awq")

messages = [
    {"role": "user", "content": "안녕하세요! 너 자신을 한 문장으로 소개해줘."},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=256, verbose=True))
```

> ⚠️ Gemma 4 26B-A4B is instruction-tuned. **Always use `apply_chat_template`** — raw prompts produce garbage on every Gemma 4 quant (3-bit, 4-bit, AWQ alike).

## Multi-seed PPL

| Seed | Baseline (`imatrix3plus`) | **This model** | Δ |
| --- | --- | --- | --- |
| 7   |  58.01 ± 1.33 |  **51.87 ± 1.20** |  **−6.14** |
| 42  |  76.90 ± 1.78 |  **61.05 ± 1.41** | **−15.85** |
| 123 | 117.67 ± 2.74 |  **91.01 ± 2.10** | **−26.66** |
| **mean** | **84.19** | **67.98** | **−16.22** |

Never significantly worse than `imatrix3plus` on any tested seed. Long-context (~11K Korean prompt, Moltis-class workload) verified CLEAN on both greedy and sampled decoding — no mojibake, no language drift, no loop-locks.

## Recipe

1. **Bit allocation** — imatrix calibration ranks every quantizable Linear/SwitchLinear by mean activation sum-square; top 35% (114 tensors) → 4-bit AFFINE group-64, remainder (211 tensors) → 3-bit AFFINE group-64. Embeds / norms / `lm_head` / vision / audio towers stay bf16 by default.
2. **embed_tokens override** — sensitivity ranking doesn't capture the tied lm_head usage pattern (every decode step reads every row), so `embed_tokens` is force-promoted to **AFFINE 8-bit** group-64 — keeps almost all of bf16 quality (Δ +1.1 ppl mean vs bf16) while saving ~530 MB per decode step in bandwidth, lifting tok/s past the uniform 4-bit reference.
3. **AWQ scales** — group-aware joint α grid search with `--weight-moment sq` (Option B, see below). 90 candidate groups discovered (qkv-input × 30 + mlp-input × 30 + mlp-down × 30); each picks α ∈ [0.3, 0.5].
4. **Group filter** — only **qkv-input + mlp-input** groups kept (60 of 90). All 30 `mlp.down_proj` absorption groups dropped — they contribute *net negative* on every tested seed (see [Ablation](#group-ablation)).
5. **In-memory apply** — scales folded into preceding norm weights (`input_layernorm`, `pre_feedforward_layernorm`) via `γ ← γ/s`, then quantized via `mlx_lm.convert`.

Full per-tensor allocation lives in `lumen_quant_recipe.json` shipped alongside the weights.

### Option B (`--weight-moment sq`)

Standard AWQ weights the column-MSE proxy by `mean_abs[c] = E[|X_c|]` (first moment). This **underweights heavy-tailed channels** — the true L2 reconstruction objective is `Σ_{r,c} ε² · E[X_c²]`. Switching the proxy to `mean_sq[c] = E[X_c²]` (Option B) picks much higher α (0.3-0.5 vs 0.1-0.3 with mean_abs) and the proxy gains now correlate with end-to-end PPL. The AWQ paper's scale formula (`s_c = mean_abs[c]^α`) is preserved; only the proxy *weighting* changes.

### Group ablation

Single-seed (123) PPL by group subset:

| Subset | Groups | PPL ± SE | Δ vs `imatrix3plus` |
| --- | --- | --- | --- |
| qkv-input only | 30 | 105.28 ± 2.50 | −12.39 |
| mlp-input only | 30 | 108.64 ± 2.50 |  −9.03 |
| mlp-down only | 30 | **141.65 ± 3.42** | **+23.98** (regression) |
| qkv + mlp-input + mlp-down (full 90) | 90 | 101.01 ± 2.39 | −16.66 |
| **qkv + mlp-input (this model, 60)** | **60** | **90.10 ± 2.08** | **−27.57** (single seed) |

mlp-down absorption folds `1/s` into `mlp.up_proj.weight` out-axis row scaling, distorting the SwiGLU multiplicative interaction `gate_proj(x) * up_proj_scaled(x)` entering `down_proj`. Even with compensating qkv / mlp-input rebalance the net contribution is negative. Norm-absorption (qkv into `input_layernorm`, mlp-input into `pre_feedforward_layernorm`) is the safe and effective path.

### Why 8-bit embed_tokens specifically

`embed_tokens` is tied to `lm_head` in Gemma 4 — `lm_head(h) = h @ embed_tokens.T`. Every decode step reads the entire embedding table for the output projection. At bf16 that's 1.07 GB read per step on a 36 GB Mac (262208 vocab × 2048 hidden × 2 bytes), bandwidth-bound at ~2 ms/step on M3 Max. Measured embed_tokens bit-width sweep:

| embed format | bpw | Mean PPL | Δ vs baseline | Decode tok/s |
| --- | --- | --- | --- | --- |
| bf16 | 4.135 | 66.86 | −17.33 | 63.5 |
| **AFFINE 8-bit (this model)** | **3.916** | **67.98** | **−16.22** | **73.0** |
| AFFINE 4-bit | 3.799 | 71.68 | −12.51 | 81.9 |
| AFFINE 3-bit | 3.770 | 81.04 |  −3.15 | (quality collapse) |

8-bit lands at the sweet spot — speed parity with uniform 4-bit while keeping 94% of the bf16 quality gain.

## Calibration

- Corpus: ~34K tokens, 38 sequences (max 1024 tok each), mixed Korean / English / code passages
- Forward source: `mlx-4bit` build (cheaper than bf16 on a 36 GB Mac; importance ranking is stable to source precision)
- Search: `--top4-fraction 0.35 --weight-moment sq` (with subsequent group filter)

## Hardware

Built on Apple M3 Max 36 GB. Loads and runs on any Apple Silicon with ≥ 16 GB unified memory. Decode throughput on M3 Max ≈ 13.7 ms/token median (matches mlx-lm baseline for similar bpw).

## Limitations

- **Calibration domain leak risk**: PPL eval uses `tulu-3-sft-mixture`; the imatrix corpus also includes English instruction-style text. Cross-seed mean improvement is the conservative reported number; absolute PPL on out-of-distribution corpora may differ.
- **Single hardware class measured**: tested on M3 Max. M1/M2 may show different latency but quality is hardware-independent.
- **Inherits all upstream Gemma 4 constraints** — see the full [model card upstream](https://huggingface.co/google/gemma-4-26B-A4B-it) for capabilities, safety considerations, and licensing.

## License & attribution

This model is a quantized derivative of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) — Apache 2.0 with the [Gemma 4 License](https://ai.google.dev/gemma/docs/gemma_4_license) terms. Recipe and AWQ tooling: [lumen-rs](https://github.com/hsng95/lumen-rs) (`scripts/quant/`).
