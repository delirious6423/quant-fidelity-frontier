---
license: other
base_model: zai-org/GLM-5.3
pipeline_tag: text-generation
tags:
  - glm
  - glm-5.3
  - moe
  - exl3
---

> [!TIP]
> **[Support this work →](https://donate.sybilsolutions.ai)** · [X](https://x.com/0xsero) · [GitHub](https://github.com/0xsero) · [REAP paper](https://arxiv.org/abs/2510.13999) · [Cerebras REAP](https://huggingface.co/collections/cerebras/cerebras-reap)

# GLM-5.3 — EXL3 3.0bpw (unpruned base)

**The full GLM-5.3 (all 256 experts) quantized to ~3-bit — 293 GB. This is the top of the series and the reference every pruned cut is measured against.**

## What this is

GLM-5.3 is a 753B mixture-of-experts model: each token is routed to 8 of 256 expert sub-networks per layer, so only ~40B parameters are active at once. This repo is that full model quantized to **EXL3 3.0 bpw** (turboderp’s exllamav3 trellis format) — no experts removed. Attention, the shared expert, and the router are kept at full-precision BF16; only the expert weights are compressed.

## Which size should I pick?

Every row is the same model at a different quality/size trade-off — fewer experts is smaller and faster but further from the original:

| Variant | Experts kept | Size | KL vs BF16 (lower = closer) |
|---|---|---|---|
| **base (unpruned)** (this) | 256 (all) | 293 GB | 0.089 |
| [661B](https://huggingface.co/0xSero/GLM-5.3-661B-EXL3-3.0bpw) | 224 / 256 | 258 GB | 0.195 |
| [615B](https://huggingface.co/0xSero/GLM-5.3-615B-EXL3-3.0bpw) | 208 / 256 | 240 GB | 0.283 |
| [569B](https://huggingface.co/0xSero/GLM-5.3-569B-EXL3-3.0bpw) | 192 / 256 | 223 GB | 0.361 |
| [533B](https://huggingface.co/0xSero/GLM-5.3-533B-EXL3-3.0bpw) | 180 / 256 | 210 GB | 0.428 |
| [500B](https://huggingface.co/0xSero/GLM-5.3-500B-EXL3-3.0bpw) | 168 / 256 | 197 GB | 0.511 |

This is the reference cut. Any smaller variant below trades a little fidelity for a lot less VRAM.

## How close to the original is it?

**KL divergence vs full BF16: 0.089 nats**, measured token-by-token over a sealed 25-prompt panel across the full 154k-token vocabulary. KL divergence is the standard “how differently do these two models predict” score: **0 = identical**, lower = closer. For scale: the *unpruned* base scores 0.089, and this cut still reproduces the full model’s single most-likely next token about 91% of the time.

All cuts are measured the same way in the [fidelity study](https://huggingface.co/datasets/0xSero/glm-5.3-reap-fidelity-study).

## Why these experts (and not the popular ones)

Naive pruning keeps the experts that fire *most often* — which quietly deletes the specialists that a single domain (code, a rare language, structured output) leans on. This series instead scores each expert by its **largest share of any one domain’s routed work** (“max-over-domain”), so every domain keeps its specialists even when they’re globally rare. On the same panel this beats frequency-based pruning by a wide margin — the head-to-head is in the [fidelity study](https://huggingface.co/datasets/0xSero/glm-5.3-reap-fidelity-study).

## Running it

```bash
pip install exllamav3        # or run via TabbyAPI
# download this repo, then tensor-split it across your GPUs in TabbyAPI/exllamav3
```
Built for Blackwell / any CUDA GPUs with enough combined VRAM (4x 96 GB).

## Credits

- **[Z.AI / zai-org](https://huggingface.co/zai-org)** — [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3), the base model.
- **[Cerebras Research](https://github.com/CerebrasResearch/reap)** — REAP ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999)).
- **[turboderp / exllamav3](https://github.com/turboderp-org/exllamav3)** — the EXL3 format and runtime.

Observations: [`glm-5.3-reap-observations-v1`](https://huggingface.co/datasets/0xSero/glm-5.3-reap-observations-v1) · Fidelity study: [`glm-5.3-reap-fidelity-study`](https://huggingface.co/datasets/0xSero/glm-5.3-reap-fidelity-study) · Built on 8× NVIDIA RTX PRO 6000 Blackwell.

## License
Inherits the [GLM-5.3 license](https://huggingface.co/zai-org/GLM-5.3).
