---
license: gemma
base_model: google/gemma-4-26B-A4B-it
tags: [mach, e8, vector-quantization, moe, gemma]
---

# gemma-4-26B-A4B-it — E8 2.4-bit (Mach)

A **9.58 GB** quantization of `google/gemma-4-26B-A4B-it` (text-only) using **E8 lattice
vector quantization** for the routed MoE experts + 6-bit affine for the non-expert weights.

> **Requires [Mach](https://github.com/) (`local_moe_engine`) to run.** The E8 expert
> format is not loadable by transformers / vLLM / llama.cpp — there is no GGUF equivalent.

## Why
- **9.58 GB — smaller than Unsloth's smallest 2-bit tier** (UD-IQ2_XXS, 9.92 GB).
- **Top-1 agreement with bf16 ties Unsloth UD-Q2_K_XL** on a confident corpus; KL(bf16‖·) ≈ 0.56.

## Recipe
| component | quant | bpw |
|---|---|---|
| routed experts — gate/up | E8 cb18 single-stage | 2.375 |
| routed experts — down | E8 cb16+cb8 2-stage | 2.56 |
| embed / lm_head / attention / dense-FFN | affine | 6-bit |
| router / norms | F32 / bf16 | — |
| vision tower | **dropped (text-only)** | — |

## Usage — full-resident (non-streaming)

`prewarm_full=True` loads every expert into RAM at startup, so there is **no disk I/O at
inference** — the fully-resident, non-streaming path. (E8 experts dispatch through Mach's
GGUF-bank kernel, so the residency arg is `"streaming"`, but prewarm makes it fully
resident; weights occupy ~9.6 GB RAM, plus your KV cache.)

```python
from local_moe_engine.io.load import load_moe
model, tok, store, cache = load_moe(
    "gemma-4-26B-A4B-it-mach-e8",
    experts_sidecar_dir="gemma-4-26B-A4B-it-mach-e8/expert_sidecar",
    native_expert_io=True,
    expert_residency="streaming",
    prewarm_full=True,            # <- fully resident, no streaming at inference
)
```
