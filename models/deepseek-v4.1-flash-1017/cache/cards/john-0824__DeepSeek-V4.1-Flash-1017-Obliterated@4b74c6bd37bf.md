---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
tags:
- abliteration
- refusal-removal
- deepseek
- sglang
---

# DeepSeek-V4.1-Flash-1017-Obliterated

Refusal-abliterated (weight surgery) version of **deepseek-ai/DeepSeek-V4.1-Flash-1017**.
The refusal direction was located with a write-surface probe (sglang forward hooks on
`self_attn`/`mlp` outputs; 4,096 harmful + 4,096 harmless WildGuardMix prompts;
diff-of-means + randomized-SVD top-8 per layer per surface) and projected out of the
residual-writer tensors with per-group Frobenius-norm preservation:

- `self_attn.o_b_proj` (attention output) — layers 2-39 — FP8
- `mlp.shared_experts.down_proj` — layers 2-39 — FP8
- `mlp.experts.{e}.down_proj` (routed) — layers 2-39 — FP4

Edited tensors were re-quantized to the **original stock formats** (FP8 e4m3 with 32x32
e8m0 block scales; FP4 e2m1 nibble-packed with per-32 e8m0 scales), so the checkpoint
serves exactly like the base model. Unedited tensors are byte-identical to the release.

## Results

| Metric | Base | This model |
|---|---|---|
| Refusal rate — held-out 200-prompt final test (greedy, 128 tok) | **59.0%** | **0.0%** |
| Refusal rate — full 74,031-prompt WildGuardMix harmful corpus | — | **0.19%** (139/72,183 valid) |
| Harmless false-refusal (200-prompt control) | 0.0% | 0.0% |
| MMLU 0-shot (57 subjects x 10, 570 questions) | 82.28% | **82.81%** |

Measurement symmetry: before/after use the same seeded prompt sets, decoding, scorer, and
serving stack (sglang, TP2, engram tables on host, DSPARK speculative decoding).

Surgery quality under re-quantization (fraction of the intended delta preserved):
o_b_proj 97.7%, shared expert down 98.3%, routed expert down 30.1% (FP4 grid coarser than
the edit — see the reports below).

## Serving

Identical to the base model, e.g.:

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
sglang serve --trust-remote-code --model-path <this repo> --tp 2 \
  --mem-fraction-static 0.8 --speculative-algorithm DSPARK \
  --speculative-dspark-block-size 5 --cuda-graph-max-bs-decode 64
```

## Files

- `obliteration_patch_report.json` — per-layer/per-group surgery statistics (delta size,
  re-quantization survival, self-check roundtrip errors, rewritten shard list)
- `obliteration_probe_report.json` — per-layer refusal-signal ranking from the probe

## Note

A small share of responses (2.5%) are scored "invalid" by the refusal scorer
(verbose/truncated long-form answers); inspection shows they are compliance, not refusal.
A handful of residual responses use hedging phrasing ("I can't physically X, but here's
the full walkthrough") while still answering.

License: MIT (same as the base model).
