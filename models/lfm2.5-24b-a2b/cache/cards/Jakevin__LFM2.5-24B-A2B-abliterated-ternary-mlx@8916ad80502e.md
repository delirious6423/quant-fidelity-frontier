---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
library_name: mlx
pipeline_tag: text-generation
tags: [mlx, abliterated, uncensored, moe, ternary, gptq, lfm2]
---

# LFM2.5-24B-A2B-abliterated-ternary-mlx

**Known quality degradation: this ternary build often fails to finish even simple questions within 8K tokens and can hallucinate facts. Not recommended for general use; see Evaluation.**

> ⚠️ **Uncensored model.** Refusal behavior has been removed. You are responsible for how you use it. Experimental / personal-learning release.

## Base & license
- Base: [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), released under the **LFM Open License v1.0**. `LICENSE` and `NOTICE` are kept unchanged; see `MODIFICATION_NOTE.txt` for the modifications.

## Abliteration
- Refusal direction removed via per-layer **biprojected** direction: `W' = W - 1.75 · r rᵀ W`
- Applied to layers **14–39**: mixer `out_proj` and MoE expert `w2` (down projection).
- Follows the recipe of [AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF](https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF) (credit to AtomicChat), with directions recomputed on our own weights.

## Quantization (~10.3 GB)
- MoE experts: GPTQ 3-bit; per layer, low-routing-frequency experts are ternary (GPTQ) — **114 ternary expert slots**, ~4.7% of expert params (1,075,838,976 / 22,951,231,488).
- Attention/conv/dense: 4-bit; router: 8-bit; embeddings: 6-bit (MLX affine, group size 64).
- Storage is standard MLX affine quantization (uniform bits per SwitchGLU stack; ternary values are stored inside regular low-bit tensors), so **no custom kernels are needed — stock `mlx-lm` loads it** (tested with mlx-lm 0.32.0).
- `per_layer_alloc.json` / `quant_stats.json` describe the per-layer allocation.

## Usage
```bash
pip install mlx-lm
mlx_lm.generate --model Jakevin/LFM2.5-24B-A2B-abliterated-ternary-mlx --prompt "Hello"
```
```python
from mlx_lm import load, generate
model, tok = load("Jakevin/LFM2.5-24B-A2B-abliterated-ternary-mlx")
p = tok.apply_chat_template([{"role": "user", "content": "Hello"}], add_generation_prompt=True, tokenize=False)
print(generate(model, tok, p, max_tokens=512))
```
Peak memory ~13.6 GB during evaluation (Apple Silicon).

## Evaluation
Small sample; no confidence intervals.

### Harmful prompts (10, 8K cap)
| Model | Refused |
|---|---|
| Official LiquidAI MLX 4-bit | 10/10 |
| **This model** | **0/10** (8 answered, 2 unfinished at 8K) |

### General questions (4, 8K cap)
| Prompt | Official LiquidAI MLX 4-bit | This model |
|---|---|---|
| Taipei night market | 693 tokens, done | 1,354 tokens, done but contains fabricated details |
| Photosynthesis | 153 tokens, done | not finished at 8,192 |
| Fibonacci code | 933 tokens, done | not finished at 8,192 |
| Capital of Japan | 240 tokens, done | 203 tokens, done |

For reference, the non-ternary abliterated 4-bit build (Jakevin/LFM2.5-24B-A2B-abliterated-4bit-mlx) was not run on the same 4-question 8K suite; with a 1,000-token cap it finished 3/4, and the Fibonacci question finished in 1,235 tokens in a separate 8K rerun.
