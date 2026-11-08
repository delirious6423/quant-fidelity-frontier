---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
library_name: transformers
tags:
- reap
- expert-pruning
- moe
- deepseek_v41
---

# DeepSeek-V4.1-Flash-REAP-50

DeepSeek-V4.1-Flash with 50% of its routed experts removed by REAP
(Router-weighted Expert Activation Pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999)).
Each MoE layer keeps 192 of its 384 routed experts; the shared expert, attention, embeddings
and every other weight are unchanged, and no weights were retrained after pruning.

This is an independent derivative of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision
`1bae0e108899`), made by jackmaxwil. It is not affiliated with or endorsed by DeepSeek.
The base model is MIT-licensed and so is this checkpoint.

Sibling release: [jackmaxwil/DeepSeek-V4.1-Flash-REAP-75](https://huggingface.co/jackmaxwil/DeepSeek-V4.1-Flash-REAP-75).
Calibration statistics, survivor lists and the eval data: [jackmaxwil/DeepSeek-V4.1-Flash-REAP-stats](https://huggingface.co/datasets/jackmaxwil/DeepSeek-V4.1-Flash-REAP-stats).

## What changed

- `n_routed_experts` is 192 in `config.json` and `inference/config.json`; `num_experts_per_tok` stays 6.
- The router weight and bias of every MoE layer are sliced to the surviving experts, and the
  surviving experts are renumbered 0..191 in their original order.
- Expert weights keep the base model's FP4 format. `keep-prune.json` records the source index
  and survivor-list hashes; `SHA256SUMS` covers every file.

Run it exactly as the base model (see `inference/`); only the expert count differs.

## How the experts were chosen

REAP scores each expert by its router gate weight times the norm of its output, averaged over
the tokens routed to it, and keeps the top 192 per layer. Calibration ran on 2,946,031
tokens (24 sessions of up to 16,384-token records) of English and code, weighted toward agentic
coding, terminal and tool use. Sources: nvidia/Open-SWE-Traces, nebius/SWE-rebench-V2-PRs, nvidia/Nemotron-Terminal-Corpus, Agent-Ark/Toucan-1.5M, CIRCL/vulnerability-scores, common-pile/stackexchange_filtered, HuggingFaceFW/finepdfs, corbt/enron-emails, allenai/WildChat-4.8M, HuggingFaceFW/finewiki. No Chinese text was used.

## Evaluation

A held-out eval split from the same sources (108 sessions, 437,350 scored positions) was run
through the full model and through the full model with the pruned experts masked out of the
router. The table compares the pruned model with the full model as teacher: KL over the
teacher's top 2,048 logits, top-1 agreement, and the change in NLL of the reference next token
(negative means the pruned model gave the reference token higher probability).

| Domain | Scored positions | Top-1 agreement | KL(teacher ‖ pruned) | NLL change (nats) |
|---|---:|---:|---:|---:|
| **all** | 437,350 | 0.803 | 0.810 | -0.138 |
| agentic-python | 63,310 | 0.718 | 1.331 | -0.358 |
| agentic-rust | 49,149 | 0.697 | 1.414 | -0.404 |
| bash | 50,094 | 0.921 | 0.270 | -0.155 |
| business | 32,823 | 0.839 | 0.259 | +0.138 |
| business-email | 15,382 | 0.791 | 0.546 | +0.313 |
| code-diffs | 64,558 | 0.939 | 0.107 | +0.062 |
| cyber | 30,806 | 0.828 | 0.582 | +0.360 |
| english-chat | 15,439 | 0.773 | 0.366 | +0.076 |
| english-prose | 18,873 | 0.834 | 0.550 | +0.354 |
| networking | 32,674 | 0.896 | 0.156 | +0.087 |
| tool-use | 64,242 | 0.662 | 1.931 | -0.707 |

All four candidates scored in the same run:

| Variant | Experts kept | Top-1 agreement | KL | NLL change |
|---|---:|---:|---:|---:|
| REAP-33 (not published) | 256 / 384 | 0.842 | 0.511 | +0.025 |
| REAP-50 | 192 / 384 | 0.803 | 0.810 | -0.138 |
| REAP-67 (not published) | 128 / 384 | 0.755 | 1.114 | -0.109 |
| REAP-75 | 96 / 384 | 0.720 | 1.349 | -0.131 |

Read these numbers with care:

- They come from router masking inside the full model, which is what this checkpoint computes.
  The same export code was checked on REAP-75: that checkpoint, run on its own over 5 eval
  sessions (1,329 positions), matched its masked-router numbers (KL 1.378 vs 1.385, top-1
  agreement 0.676 vs 0.672). REAP-50 itself is checksum-verified only, and generation has not
  been tested yet.
- The negative NLL changes on agentic and tool-use text mean the pruned model fits the
  reference traces (written by other models) a little better while moving away from the full
  model's own distribution. They are not evidence that pruning improves the model.
- Only text was calibrated and evaluated. The vision path is untouched and untested.
- There is no post-pruning fine-tuning. Expect more degradation on domains outside the
  calibration mix, and on Chinese in particular.
