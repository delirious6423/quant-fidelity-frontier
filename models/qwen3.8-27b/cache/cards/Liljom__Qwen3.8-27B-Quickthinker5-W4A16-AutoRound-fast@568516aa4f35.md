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

# Qwen3.8-27B-Quickthinker5-W4A16-AutoRound-fast

The [`scar-ai/Qwen3.8-27B-Quickthinker`](https://huggingface.co/scar-ai/Qwen3.8-27B-Quickthinker) edit at
**alpha 5** (upstream ships alpha 7; its write-up calls 5 the safer setting), carried into the **AutoRound-fast**
W4A16 build that [HyperQwen](https://github.com/syv-ai/HyperQwen) serves on a single RTX 3090. We call it AR+QT5.

> **Research artifact, not a recommendation.** In our eval it cut thinking by 23 % but failed our hardest task
> more often, and it costs MTP acceptance and KV cache. We did not adopt it.
>
> **Eval findings (full write-up, all four builds compared):** https://claude.ai/artifact/SscXkuciQGhHpgwhWqoGBz

## Lineage
1. [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), the original BF16 model.
2. [`dbirks/Qwen3.8-27B-W4A16-AutoRound`](https://huggingface.co/dbirks/Qwen3.8-27B-W4A16-AutoRound): Intel AutoRound
   int4 W4A16, group 128, symmetric, compressed-tensors.
3. **AutoRound-fast**: HyperQwen's single-user "fast" variant. Shards 1–6 come unchanged from (2). The int4-GPTQ
   `lm_head`, the int4 MTP draft module and the 40,960-row draft head come from
   [`syvai/qwen3.8-27b-3090-fast-variant`](https://huggingface.co/syvai/qwen3.8-27b-3090-fast-variant). HyperQwen
   assembles it with `prepare/fetch_fast_variant.py` into `models/Qwen3.8-27B-W4A16-AutoRound-fast`.
4. **This repo**: (3) plus Quickthinker alpha 5 (patch from upstream revision `dd9e07ab`).

## What was changed (notice of modification)
- **L52 and L56 `mlp.down_proj`: stored as BF16 `.weight`**, not int4. Their int4 `weight_packed/scale/shape` are
  removed, the shard and index are rewritten, and both modules are added to `config.json`
  `quantization_config.ignore`. The edit sets single columns far outside the int4 range (L52 column 5247 norm
  0.69 → ~51 at alpha 5), so it cannot be carried in int4. L52's column is `bf16(base + 5/7 · (alpha7 − base))`;
  L56 (column 7546) is upstream's value. Cost: about +0.24 GiB VRAM, −2.7 % KV cache.
- **`lm_head`, 12 hedge-token rows** (Double, Could, Another, Wait, Maybe, Perhaps, Hold, Actually, Alternative,
  Hmm, Alternatively, Potential): delta flips on the existing int4 GPTQ head,
  `q_new = clamp(q_p + round(W_new/s) − round(W_base/s), −8, 7)` with the head's own scales. Gate: preserved
  fraction p = 0.975 (6,692 flips, 21 clamps). Only those rows' bytes change.
- **`mtp.draft_lm_head`**: the same 12 rows synced from the new `lm_head` (the draft head is a slice of it).
- Everything else is byte-identical to (3). The side file `quantization_config.json` is unchanged and unused
  (vLLM reads `config.json`).

## Eval findings (short)
Our own private eval suite, one RTX 3090, vLLM with MTP (k = 4), reasoning effort medium, 4 hard tasks × 4 seeds
(16 calls per build). AR = the unedited AutoRound-fast base.

| | AR | AR+QT5 (this) |
|---|---|---|
| Reasoning tokens, sum | 164,805 | 126,638 (−23 %) |
| Wall time, sum | 2,810 s | 2,280 s (−19 %) |
| Hard passes of 16 / mean score | 13 / 0.868 | 12 / 0.794 |
| Hardest task (H3b) passes of 4 / score | 3 / 0.750 | 2 / 0.500 |
| Empty answer, not capped | 0 | 2 |
| MTP accept rate | 0.651 | 0.622 |
| KV cache (tokens) | 225,046 | 219,065 |

- The thinking cut is real, but the two empty H3b answers (the model stopped inside its reasoning after 23k–25k
  tokens) did not occur in any non-QT5 build. The cause was not determined.
- The Concept edit alone cut reasoning 27 % with no quality or MTP cost, so we serve that instead
  ([`Liljom/Qwen3.8-27B-Concept-W4A16-AutoRound-fast`](https://huggingface.co/Liljom/Qwen3.8-27B-Concept-W4A16-AutoRound-fast)).

Full tables, per-task results and caveats: https://claude.ai/artifact/SscXkuciQGhHpgwhWqoGBz

## How to run
Drop-in replacement for the AutoRound-fast dir in [HyperQwen](https://github.com/syv-ai/HyperQwen): point
`MODEL` at this directory. It needs HyperQwen's vLLM patches (MTP draft vocabulary among them), as the base
does. Plain vLLM without those patches is untested.

## Limitations
- 16 scored calls per build at temperature 1.0: a one-call difference on a task is not a finding.
- Not tested at effort xhigh; upstream evaluated on GSM8K / MATH-500 / MMLU-Pro only.

## Credits, license, citation
Apache-2.0, as all upstreams. Credit: the Qwen team (base model); Reflex Labs (Quickthinker); dbirks (AutoRound
quant); syv-ai / HyperQwen (fast variant and serving stack).

```bibtex
@misc{reflexlabs2026quickthinker,
  title  = {Qwen3.8-27B-Quickthinker: 43\% less thinking from a 0.00027\% weight edit},
  author = {{Reflex Labs}},
  year   = {2026},
  url    = {https://huggingface.co/scar-ai/Qwen3.8-27B-Quickthinker}
}
@article{cheng2023optimize, title={Optimize Weight Rounding via Signed Gradient Descent for the Quantization of LLMs}, author={Cheng, Wenhua and others}, journal={arXiv:2309.05516}, year={2023}}
```
