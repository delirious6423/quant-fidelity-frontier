---
license: mit
library_name: transformers
pipeline_tag: text-generation
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
tags:
  - deepseek-v4
  - abliterated
  - heretic
  - fp8
  - reasoning
  - tool-use
---

# DeepSeek-V4-Flash-0731 HERETIC Abliterated FP8

A native-FP8 behavioral derivative of [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731).

This release removes a three-mode refusal subspace from selected attention output projections while preserving the official architecture, tokenizer/encoder, FP8 format, reasoning modes, tool-call format, routed experts, and shared experts.

> **Release gate:** GREEN — baked artifact load, behavior, tool-call, and corrected reasoning smoke verified.

## Tool attribution

This checkpoint was produced with [Heretic](https://github.com/p-e-w/heretic) v1.4.0 by Philipp Emanuel Weidmann and contributors, using base commit `7675b90d648154cdfefa597372cc477df9848eab`. Heretic is their project and is licensed `AGPL-3.0-or-later`; it is not owned or authored by this model publisher.

The release-specific work here is the DeepSeek V4/mHC two-node adaptation, separate `chat` / `think-high` / `think-max` direction capture, rank-3 subspace construction, native-FP8 attention-only bake, and the evaluation/package receipts. Model weights retain the upstream DeepSeek MIT license. Heretic-derived source modifications, when distributed, remain under AGPL-3.0-or-later. See `HERETIC_ATTRIBUTION.md`.

## What changed

- Built three separate refusal directions from `chat`, `think-high`, and `think-max` activations.
- Orthonormalized them into a rank-3 refusal subspace per layer instead of averaging low-cosine directions.
- Edited attention output projections for backbone layers `10–42`.
- Applied the deepest backbone subspace to the three attached MTP/DSpark attention projections.
- Used `lambda=1.5` selected by a runtime sweep under a `>=0.99` tool-call gate.
- Dequantized and re-quantized only edited FP8 tensors back to the original `F8_E4M3` + `F8_E8M0` representation.

## What did not change

- No reasoning mode was disabled or shortened.
- No lower-bit conversion was applied.
- No routed-expert or shared-expert weight was edited.
- No tokenizer, official V4 encoder, model config, context setting, or tool schema was replaced.
- The official source checkpoint remained read-only throughout capture, search, bake, and evaluation.

## Mechanical receipts

- Edited tensors: `72` (`36` weight + scale pairs)
- Untouched indexed tensors: `72,245`
- Overlay size: approximately `1.21 GB`
- Cross-node deterministic overlay SHA-256: `601c440978a973d6c0a7dbf00903bf5bbe0c278d4e43d68947ff65ed631b1bd3`
- Rewritten index SHA-256: `f06fc8d7a4872ba6e8ccc84d6b48859123e9beeae2ad9f8bdb9cf82f260ec114`
- Worst quantized intended-effect retention: `90.57%`
- Clamp fraction: `5.17e-6`
- Dangling shards across both independently produced package trees: `0`

## Behavioral evaluation

Deterministic evaluation used the official DeepSeek V4 encoder, temperature `0`, seed `0`, identical prompts, and the same two-node vLLM runtime for baseline and derivative.

### Refusal and tool use

| Mode | Official baseline (n=small sweep) | Baked model (n=full) |
|---|---:|---:|
| chat | 93.75% | 29.17% (`n=48`) |
| think-high | 75.00% | 33.33% (`n=24`) |
| think-max | 87.50% | 29.17% (`n=24`) |

- Empty-answer rate: `0%`
- Tool-call compliance: `100%` (`6/6`)
- Correct-tool rate: `100%` (`6/6`)
- Tool parse failures: `0`

The refusal detector is deliberately broad and may over-count soft disclaimers. These figures are evaluation receipts, not a claim that every safety behavior has been removed.

### Reasoning regression smoke

A 12-item deterministic arithmetic, symbolic, sequence, and code-output smoke was run in all three modes. This is a regression smoke, not a substitute for MATH, HumanEval, or a broad capability benchmark.

- Official: chat `10/12`, think-high `12/12`, think-max `12/12`
- Baked: chat `10/12`, think-high `12/12`, think-max `12/12`

Full details and result hashes are in `EVAL_RESULTS.md`.

## Runtime

Use a DeepSeek-V4-compatible runtime and the official encoder. This checkpoint retains the official V4/DSpark architecture and native FP8 layout. Generic runtimes without DeepSeek V4 support may not load it correctly.

The model was exercised on two NVIDIA DGX Spark nodes with tensor parallelism `2`, the official V4 encoder, and a DeepSeek-V4-capable vLLM build.

## Reproducibility

The release includes:

- refusal direction capture and diagnostics
- rank-3 orthonormal subspace construction receipt
- runtime lambda sweep
- FP8 fixed-exponent projection/requantization code
- bake report
- behavior and quality evaluation receipts
- package hashes

## Limitations

- This is an unofficial community derivative and is not affiliated with or endorsed by DeepSeek.
- Abliteration reduces some refusal behavior and can increase compliance with unsafe requests. This checkpoint is not safety-certified; downstream deployers remain responsible for evaluation, access controls, monitoring, and compliance with applicable law and platform policy.
- The reasoning check is intentionally small; stronger benchmark coverage is welcome.
- Refusal behavior is prompt- and decoder-sensitive.
- The three MTP/DSpark stages inherit the deepest measured backbone subspace because the capture run did not instantiate speculative decoding.
- This is a behavioral weight edit. Evaluate it for your own use case before deployment.

## License and attribution

The upstream repository and model weights are MIT licensed. This derivative retains the MIT license and original DeepSeek copyright notice.

Base model: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)

## Citation

```bibtex
@misc{deepseekai2026deepseekv4,
  title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  author={DeepSeek-AI},
  year={2026}
}
```
