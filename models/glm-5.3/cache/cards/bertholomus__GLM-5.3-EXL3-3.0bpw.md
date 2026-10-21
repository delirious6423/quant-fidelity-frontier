---
license: other
license_name: glm-5.3
license_link: LICENSE
base_model: zai-org/GLM-5.3
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
language:
- en
- zh
tags:
- exl3
- exllamav3
- glm_moe_dsa
- glm-5.3
- tensorfold
- dgx-spark
- gb10
- tensor-parallel
---

# GLM-5.3 — EXL3 3.0 bpw (BertholomusAI)

**EXL3 3.0 bpw quantization of [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (the full 753B
`glm_moe_dsa` model, not Flash), made by BertholomusAI from Z.AI's BF16 weights. Model and weights © 2026 Z.AI,
under the [GLM-5.3 License](LICENSE).** This repository holds quantized weights only; the original model card,
benchmarks and prompt format are on Z.AI's [model page](https://huggingface.co/zai-org/GLM-5.3).

Built to run on four NVIDIA DGX Spark (GB10) nodes, tensor-parallel 4, with
[TensorFold](https://github.com/ashhart/TensorFold) (by ashhart) and our `glm_moe_dsa` engine branch.
Recipe, launch scripts and benchmarks: **[bertholomus/glm-5.3-tensorfold-tp4-4xgb10](https://github.com/bertholomus/glm-5.3-tensorfold-tp4-4xgb10)**.

## Quantization

| | |
|---|---|
| Source | `zai-org/GLM-5.3` BF16, revision `9d2398f4`, 282 shards, every shard SHA-256-checked against the Hub |
| Converter | [exllamav3](https://github.com/turboderp-org/exllamav3) 1.5.3 (`d3739fd`) by turboderp, on one GB10 |
| Command | `convert.py -b 3.0 -hq -mb 4 -hb 6 -cb mul1 -cr 250 -cc 2048` |
| Routed experts (layers 3–77, 256 each) | 3 bits, codebook mul1 |
| Attention, shared experts, indexer `wq_b` (`-hq`) | 5 bits |
| MTP layer (`-mb`) | 4 bits |
| `lm_head` (`-hb`) | 6 bits |
| Embeddings, norms, indexer `wk` / `weights_proj` | bf16 |
| Whole model | 3.04 bpw average, 39 shards, 273 GiB |

## Quality (KL divergence vs BF16)

exllamav3 `eval/model_diff.py`, wikitext-2 test split, 32 rows × 2,048 tokens, BF16 activations:

| | value |
|---|---|
| KL(P_BF16 ‖ P_quant) | **0.109** |
| KL(P_quant ‖ P_BF16) | 0.123 |
| Top-1 agreement with BF16 | **90.2 %** |
| Perplexity, quant / BF16 | 2.911 / 2.750 |

Longer panel (4 × 16,384 tokens, exercises DSA's top-2,048 selection): KL 0.119.

## Serving (measured, 4× GB10, TP4, release build)

- **4 concurrent streams:** 82 tok/s aggregate (code), 71–73 (chat), time to first token ~0.5 s, at up to 65,536
  tokens a stream (replicated cache).
- **4 streams sharing one 1,048,576-token pool** (context parallelism over the four ranks): ~64 tok/s aggregate (code),
  56 (chat), first token 0.2–1.1 s.
- **Single stream:** ~37–43 tok/s with MTP-3 drafts.
- **1M context:** needle found at 1,039,064 tokens.
- **Exact:** every concurrent reply is bit-identical to its solo run; drafted output equals serial decoding.
- **Long prompts are slow:** ~400 tok/s prefill on the 1M configuration (~76 s to the first token at 32K).

Full numbers, both configurations and how to reproduce them are in the recipe repository.

## Before you run this

- **For most people, [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) is the better everyday model.** It is
  faster and has native vision. We made this quant for one reason: quality, as close to the full model as four Sparks
  allow.
- **No vision** in this first release.
- **No DFlash.** The only DFlash drafter for the full GLM-5.3 is licensed CC BY-NC-ND 4.0, so this recipe drafts with
  GLM-5.3's own MTP layer. That costs speed, not quality: the full model checks every token.

## Credits

- **Z.AI** — GLM-5.3. These weights are a quantized derivative of theirs and remain under the GLM-5.3 License,
  Copyright (c) 2026 Z.AI (`LICENSE`, unchanged).
- **turboderp** — EXL3 and exllamav3 (MIT): the format and the converter.
- **ashhart** — TensorFold (Apache-2.0): the serving engine.
- **BertholomusAI** (Albert Lee, [bertholomus](https://huggingface.co/bertholomus)) — the quantization, the
  `glm_moe_dsa` TP4 engine branch and the measurements.

Not affiliated with or endorsed by Z.AI, NVIDIA, the TensorFold authors or turboderp.
