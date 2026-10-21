---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
tags: [gguf, lfm2, moe, abliterated, uncensored, llama.cpp]
---

# LFM2.5-24B-A2B-abliterated-mixed-GGUF

> **Warning: uncensored model.** Refusal behaviour has been largely removed. It may produce harmful, offensive or illegal content. You are responsible for how you use it and for complying with the LFM Open License v1.0 and applicable law.

Abliterated [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) as a **mixed-precision** GGUF (this is **not** a ternary build).

## Files
| File | Size | Notes |
|---|---|---|
| `LFM2.5-24B-A2B-abliterated-mixed-IQ3_M.gguf` | 10.67 GB | mixed IQ3_M / Q4_K / Q8_0 |
| `lfm25-abl-v2.imatrix` | 56 MB | importance matrix computed on the abliterated weights |

## Quantization recipe (verified from the GGUF tensor types)
- Routed experts (`ffn_{gate,up,down}_exps`): IQ3_M quant mix (stored as IQ3_S tensors; 3 `ffn_down_exps` tensors kept at Q4_K by llama.cpp's IQ3_M rules), using an imatrix **recomputed on the abliterated weights**.
- Attention (`attn_q/k/v/output`), short-conv (`in_proj`/`out_proj`) and dense FFN layers: Q4_K.
- Token embedding (tied with output): Q8_0.
- Norms, conv kernels, router: F32.

## Abliteration
Follows the recipe of [AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF](https://huggingface.co/AtomicChat/LFM2.5-24B-A2B-Uncensored-GGUF) (credit to AtomicChat), recomputed on our own activations: a per-layer biprojected refusal direction `r`, applied as `W' = W - 1.75 · r rᵀ W` to the mixer `out_proj` and all expert `w2` (down) weights on layers 14–39.

## Usage
Requires llama.cpp with `lfm2moe` support (tested with build b9810).
```
llama-cli -m LFM2.5-24B-A2B-abliterated-mixed-IQ3_M.gguf -c 8192 -ngl 99 --jinja
```

## Evaluation
**A systematic refusal/quality evaluation was NOT completed for this GGUF** (the eval harness failed to capture model output). Only manual checks were done (Apple M4 Pro, llama.cpp b9810, temp 0.2, `--reasoning off`, the model still emits a thinking block):
- Smoke test: "What is the capital of Japan?" → correct answer ("Tokyo"), thinking block closed, ~83 tok/s generation, peak RSS ~10.9 GB.
- Manual check: **2/2 harmful prompts not refused** (2 prompts from our held-out harmful set, 8192-token cap; both finished with complete answers of ~900–1,000 words, ~82 tok/s).

Refusal removal is inherited from the abliterated v2 weights: the MLX 4-bit build of the same weights refused 0/10 harmful prompts at a 1,800-token cap. It was not separately measured on this GGUF beyond the 2-prompt check above. Samples are tiny; no confidence intervals. Quantization-related quality loss has not been measured.

## Related
- [Jakevin/LFM2.5-24B-A2B-abliterated-4bit-mlx](https://huggingface.co/Jakevin/LFM2.5-24B-A2B-abliterated-4bit-mlx)
- [Jakevin/LFM2.5-24B-A2B-abliterated-ternary-mlx](https://huggingface.co/Jakevin/LFM2.5-24B-A2B-abliterated-ternary-mlx)

## License
LFM Open License v1.0 — see `LICENSE` and `NOTICE`.
