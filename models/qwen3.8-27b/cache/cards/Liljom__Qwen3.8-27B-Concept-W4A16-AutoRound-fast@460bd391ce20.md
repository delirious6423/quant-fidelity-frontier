---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
language:
  - en
tags:
  - int4
  - w4a16
  - compressed-tensors
  - auto-round
  - vllm
  - hyperqwen
  - weight-edit
  - qwen3
---

# Qwen3.8-27B-Concept-W4A16-AutoRound-fast

The [`ConceptCountry/qwen-concept-27b`](https://huggingface.co/ConceptCountry/qwen-concept-27b) weight edit,
carried into an existing 4-bit quant **without re-quantizing**. The base is the **AutoRound-fast** W4A16 build
that [HyperQwen](https://github.com/syv-ai/HyperQwen) serves on a single RTX 3090. Every tensor except the 9 edited
matrices is byte-identical to that base.

**Eval findings (full write-up, all four builds compared):** https://claude.ai/artifact/SscXkuciQGhHpgwhWqoGBz

## Lineage
1. [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), the original BF16 model.
2. [`dbirks/Qwen3.8-27B-W4A16-AutoRound`](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound): Intel AutoRound
   int4 W4A16, group 128, symmetric, compressed-tensors.
3. **AutoRound-fast**: HyperQwen's single-user "fast" variant. Shards 1–6 come unchanged from (2). The int4-GPTQ
   `lm_head`, the int4 MTP draft module and the 40,960-row draft head come from
   [`syvai/qwen3.8-27b-3090-fast-variant`](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant). HyperQwen
   assembles it with `prepare/fetch_fast_variant.py` into `models/Qwen3.8-27B-W4A16-AutoRound-fast`.
4. **This repo**: (3) plus the Concept edit.

## What was changed (notice of modification)
- Only `model.language_model.layers.{32,34,36,38,40,42,44,46,48}.mlp.down_proj.weight_packed` differ from (3).
  Scales, shapes, config, tokenizer, MTP and every other tensor are byte-identical.
- The edit (`bf16/EDIT_SPEC.json` upstream: mode `add`, alpha 0.1, writer `down`, those 9 layers) is about 0.5 % of
  ‖W‖, which is smaller than one int4 step. Re-rounding `W + Δ` with the quant's scales loses it, and a fresh RTN
  re-quant would throw away AutoRound's tuned rounding.
- Method ("delta flips"): keep the quant's scales `s` and integers `q_p` and add only the flips the edit causes
  under round-to-nearest:

      q_new = clamp(q_p + round(W_concept / s) − round(W_base / s), −8, 7)

  The base weights' fractional parts act as dither, so on average the integer change reproduces Δ.
- Gate: preserved fraction p = ⟨ΔQ, Δ⟩ / ‖Δ‖² and rank-1 gain g must be in [0.7, 1.3] for every matrix.
  Measured: p = 0.94–1.00 on all 9 layers. 0.30–0.64 % of the ints flipped, at most 0.007 % clamped.

## Eval findings (short)
Our own private eval suite, one RTX 3090, vLLM with MTP speculative decoding. AR = the unedited AutoRound-fast base.
- **Reasoning effort medium** (4 hard tasks × 4 seeds, MTP k = 4): **−27 % reasoning tokens and −27 % wall time vs AR**,
  same decode speed, same MTP acceptance (0.658 vs 0.651), same KV cache. Hard passes 12 vs 13 of 16; the gap is
  one call on a bug that fails on some seed in every build.
- **Reasoning effort low** (full 94-call suite, 2 runs vs 1): −7 to −11 % thinking on hard tasks, −23 % on
  structured output, about 10 % faster overall. Structured output 40/40 in every run; no quality change we could
  measure (run-to-run spread is as big as the gap).
- This is well short of the upstream card's 27–54 % at effort low, and we saw no quality gain.
- We serve this build ourselves because it costs nothing in quality or speed.

Full tables, per-task results and caveats (n = 16 per build at medium, temperature 1.0): https://claude.ai/artifact/SscXkuciQGhHpgwhWqoGBz

## How to run
Drop-in replacement for the AutoRound-fast dir in [HyperQwen](https://github.com/syv-ai/HyperQwen). Download it
next to the base, then point HyperQwen's `MODEL` at this directory. It needs HyperQwen's vLLM patches (MTP draft
vocabulary among them), as the AutoRound-fast base does. Plain vLLM without those patches is untested.

## Limitations
- Small private eval (16 scored calls per build at medium, 94 per run at low), temperature 1.0: single-task
  differences of one call are not findings.
- Not tested at effort xhigh, the setting the edit authors report on.
- Vision tower untouched but not evaluated.

## Credits, license, citation
Apache-2.0, as all upstreams. Credit: the Qwen team (base model); ConceptCountry (the edit); dbirks (AutoRound
quant); syv-ai / HyperQwen (fast variant and serving stack).

```bibtex
@article{cheng2023optimize, title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs}, author={Cheng, Wenhua and others}, journal={arXiv:2309.05516}, year={2023}}
```
