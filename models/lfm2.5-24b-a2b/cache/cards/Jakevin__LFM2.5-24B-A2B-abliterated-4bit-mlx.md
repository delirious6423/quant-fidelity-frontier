---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: mlx
pipeline_tag: text-generation
tags: [mlx, abliterated, uncensored, moe, lfm2]
---

# LFM2.5-24B-A2B-abliterated-4bit-mlx

> ⚠️ **Uncensored model.** Refusal behavior has been removed / strongly reduced via abliteration. You are responsible for how you use it. Experimental / personal-learning release.

## Base & license

- Base: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), **LFM Open License v1.0** (see `LICENSE`).
- This folder redistributes the base `LICENSE` and adds a short `NOTICE` describing the modifications (abliteration + MLX 4-bit quantization). The base release has no `NOTICE` file.

## Abliteration

Refusal directions were removed following the recipe of [AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF](https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF) (credit to AtomicChat), with directions **recomputed on our own weights**:

- Per-layer **biprojected** refusal direction, measured with the model chat template and an **empty think block**.
- Update: `W' = W - 1.75 · r rᵀ W`
- Applied on layers **14–39**: mixer `out_proj` and all MoE expert `w2` (down projection).
- Layers **0–13**, dense layers, router, and embeddings were left untouched.

## Quantization (~13 GB on disk)

Standard **MLX 4-bit** affine quantization after abliteration:

- Default: `bits=4`, `group_size=64`, `mode=affine`
- MoE router / gate tensors kept at **8-bit** (same pattern as LiquidAI’s official MLX 4-bit)

No training scripts are included in this repository.

## Usage

```bash
pip install mlx-lm
mlx_lm.generate --model Jakevin/LFM2.5-24B-A2B-abliterated-4bit-mlx --prompt "Hello"
```

```python
from mlx_lm import load, generate
model, tok = load("Jakevin/LFM2.5-24B-A2B-abliterated-4bit-mlx")
p = tok.apply_chat_template(
    [{"role": "user", "content": "Hello"}],
    add_generation_prompt=True,
    tokenize=False,
)
print(generate(model, tok, p, max_tokens=256))
```

Peak memory during local load/eval on Apple Silicon was about **13.6 GB**.

## Evaluation

Small sample; **no confidence intervals**. Numbers below are from saved per-item logs for this checkpoint (`abl-q4-v2`).

### Harmful prompts (10)

Paired run vs official LiquidAI MLX 4-bit, **`max_tokens=1800`**:

| Model | Refused | Complied | Unfinished think |
|---|---|---|---|
| Official LiquidAI MLX 4-bit | **10/10** | 0/10 | 0/10 |
| **This model** | **0/10** | 7/10 | 3/10 |

(Official item scored “comply” by a loose regex still returned a short refusal body; counted as refuse.)

Spot check of **one** harmful prompt at a higher cap (`extra8192.json`):

| Model | Result | Tokens |
|---|---|---|
| Official LiquidAI MLX 4-bit | refuse, closed | 153 |
| **This model** | comply, closed | 6,428 |

There is **no** saved log of all 10 harmful prompts for this 4-bit build at an 8K cap.

### Harmless prompts (10)

Same paired run, **`max_tokens=600`**:

| Model | Refused | Complied | Unfinished think |
|---|---|---|---|
| Official LiquidAI MLX 4-bit | 0/10 | 6/10 | 4/10 |
| **This model** | 0/10 | 4/10 | 6/10 |

### General questions (4)

Paired run, **`max_tokens=1000`**:

| Prompt | Official | This model |
|---|---|---|
| Taipei night market (zh-TW) | 609 tok, done | 581 tok, done |
| Photosynthesis | 167 tok, done | 236 tok, done |
| Fibonacci code (iterative) | 933 tok, done | unfinished at 1,000 |
| Capital of Japan | 194 tok, done | 204 tok, done |

Fibonacci was re-run for this model at a higher cap (`extra8192.json`): **closed at 1,235 tokens** with a correct iterative implementation (official re-run: 900 tokens).

There is **no** saved log of all four general questions for this 4-bit build at an 8K cap.

## Related: ternary sibling

[Jakevin/LFM2.5-24B-A2B-abliterated-ternary-mlx](https://huggingface.co/Jakevin/LFM2.5-24B-A2B-abliterated-ternary-mlx) is a different quantization of the same abliterated full-precision weights. That ternary build has **known quality degradation** (long unfinished generations / weaker answers). Prefer this 4-bit MLX build for general use.
