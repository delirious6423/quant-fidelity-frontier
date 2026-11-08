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

# DeepSeek-V4.1-Flash-REAP-75

DeepSeek-V4.1-Flash with 75% of its routed experts removed by REAP
(Router-weighted Expert Activation Pruning, [arXiv:2510.13999](https://arxiv.org/abs/2510.13999)).
Each MoE layer keeps 96 of its 384 routed experts; the shared expert, attention, embeddings
and every other weight are unchanged, and no weights were retrained after pruning.

This is an independent derivative of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision
`1bae0e108899`), made by jackmaxwil. It is not affiliated with or endorsed by DeepSeek.
The base model is MIT-licensed and so is this checkpoint.

Sibling release: [jackmaxwil/DeepSeek-V4.1-Flash-REAP-50](https://huggingface.co/jackmaxwil/DeepSeek-V4.1-Flash-REAP-50).
Calibration statistics, survivor lists and the eval data: [jackmaxwil/DeepSeek-V4.1-Flash-REAP-stats](https://huggingface.co/datasets/jackmaxwil/DeepSeek-V4.1-Flash-REAP-stats).

## What changed

- `n_routed_experts` is 96 in `config.json` and `inference/config.json`; `num_experts_per_tok` stays 6.
- The router weight and bias of every MoE layer are sliced to the surviving experts, and the
  surviving experts are renumbered 0..95 in their original order.
- Expert weights keep the base model's FP4 format. `keep-prune.json` records the source index
  and survivor-list hashes; `SHA256SUMS` covers every file.

Run it exactly as the base model (see `inference/`); only the expert count differs.

## How the experts were chosen

REAP scores each expert by its router gate weight times the norm of its output, averaged over
the tokens routed to it, and keeps the top 96 per layer. Calibration ran on 2,946,031
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
| **all** | 437,350 | 0.720 | 1.349 | -0.131 |
| agentic-python | 63,310 | 0.635 | 2.055 | -0.799 |
| agentic-rust | 49,149 | 0.625 | 2.055 | -0.640 |
| bash | 50,094 | 0.882 | 0.447 | -0.140 |
| business | 32,823 | 0.693 | 0.747 | +0.535 |
| business-email | 15,382 | 0.696 | 1.011 | +0.602 |
| code-diffs | 64,558 | 0.889 | 0.305 | +0.232 |
| cyber | 30,806 | 0.658 | 1.615 | +1.532 |
| english-chat | 15,439 | 0.632 | 0.962 | +0.460 |
| english-prose | 18,873 | 0.757 | 0.968 | +0.719 |
| networking | 32,674 | 0.820 | 0.411 | +0.319 |
| tool-use | 64,242 | 0.587 | 2.807 | -1.378 |

All four candidates scored in the same run:

| Variant | Experts kept | Top-1 agreement | KL | NLL change |
|---|---:|---:|---:|---:|
| REAP-33 (not published) | 256 / 384 | 0.842 | 0.511 | +0.025 |
| REAP-50 | 192 / 384 | 0.803 | 0.810 | -0.138 |
| REAP-67 (not published) | 128 / 384 | 0.755 | 1.114 | -0.109 |
| REAP-75 | 96 / 384 | 0.720 | 1.349 | -0.131 |

Read these numbers with care:

- They come from router masking inside the full model. This exported checkpoint was then run
  on its own (on CPU, through the reference code with dequantize-then-matmul stand-ins for the
  FP8/FP4 kernels) over 5 of the eval sessions, 1,329 positions, and scored against the same
  teacher logits. It matched the masked-router numbers for those positions: KL 1.378 vs 1.385,
  NLL change -0.244 vs -0.225, top-1 agreement 0.676 vs 0.672. That check is teacher-forced
  scoring; free-running generation has not been tested yet.
- The negative NLL changes on agentic and tool-use text mean the pruned model fits the
  reference traces (written by other models) a little better while moving away from the full
  model's own distribution. They are not evidence that pruning improves the model.
- Only text was calibrated and evaluated. The vision path is untouched and untested.
- There is no post-pruning fine-tuning. Expect more degradation on domains outside the
  calibration mix, and on Chinese in particular.
