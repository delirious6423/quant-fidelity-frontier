---
license: mit
license_name: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags: [exl3, deepseek, tensorfold, gb10, dgx-spark]
---

# DeepSeek-V4.1-Flash-1017 — EXL3 2.5 bpw

EXL3 quant of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision `1bae0e10`) by BertholomusAI, sized to serve on two GB10 (DGX Spark class) nodes. Model and weights © DeepSeek under the MIT License.

## Bits
| Part | Format |
|---|---|
| Routed experts | EXL3 2.5 bpw |
| Attention, shared experts, KV compressor, indexer queries, Engram key/value projection | EXL3 4 bpw (hq) |
| Indexer keys | EXL3 8 bpw |
| Output head | EXL3 6 bpw |
| MTP / DSpark drafter: routed experts and input projection | EXL3 4 bpw |
| MTP / DSpark drafter: attention and shared experts | EXL3 6 bpw |
| Embeddings, router gates, norms, the drafter's markov head, vision tower and aligner | 16-bit, unchanged |
| Overall | 2.52 bpw over the quantized weights, about 185 GB |

Converter: ExLlamaV3 1.6.0 with Fabiano Anemone's ([@anoane](https://github.com/anoane)) DeepSeek-V4 support,
[turboderp-org/exllamav3 pull request 449](https://github.com/turboderp-org/exllamav3/pull/449) (MIT), and four small
changes of ours for this checkpoint: the model's state carried from module to module for each calibration row, the
indexer keys and the Engram key/value projection quantized, a loader fallback, and a routed expert that no
calibration token reached quantized without calibration. Codebook mul1, calibration 192 rows × 2048 tokens.

The Engram tables are not in this repository: serve them from the original checkpoint's shards 47 and 48, as the
recipe does.

## Served
Served on two GB10 (DGX Spark class) nodes with TensorFold tensor parallel 2 over RDMA: 1,048,576 tokens a request in
a shared window of 2,097,152 positions, 8 concurrent streams, images, DSpark drafting. Gates passed on that lane: exact chat
71/71, the six API gates, 16 replies as a burst and staggered == solo, vision (6/6, 9/9, and four at a time 18/18 ==
solo), forced and automatic tool calls 16/16, structured output 11/11 with bursts == solo, kept-prompt resume, and
1M-context needles 3/3 in a 1,039,836-token prompt (the second and third asks resumed 1,038,336 tokens from the kept
state). Four 159,925-token prompts at once gave the same replies as each one alone from a cold start (5/5), every needle found.

Speed on that lane (tok/s; greedy, DSpark drafting, median of 3 unless noted; the recipe's
[results/served-v073/rows/](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/tree/v0.7.3/results/served-v073/rows)):

| Row | tok/s |
|---|---|
| Decode, one stream: code / prose / structured (512 tokens) | 106.0 / 60.5 / 118.3 |
| 2 and 4 streams at once, aggregate (median of 9) | 95.3 and 134.0 |
| 4 streams sustained 90 s, aggregate | 151.8 |
| 8 streams sustained 90 s, aggregate | 168.3 |
| Prompt 8K / 32K / 64K / 128K | 2,439 / 2,568 / 2,550 / 2,523 |
| Prompt 93,335 tokens | 2,364 |
| Decode after a 128K prompt (three fixed prompts) | 99.2 |

Recipe (the launch line, the gates and how to ready this download):
[bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10), v0.7.3.
This repository ships no chat template. The recipe's `tools/serving_1017.py` adds ours and six tensors under the
names and types the engine reads (the MTP markov head, the Engram q/k as F32).

## Quality
Next-token distributions against DeepSeek's original 1017 checkpoint (its own FP8 / FP4 weights, run in 16-bit by
ExLlamaV3's reference forward), teacher-forced over the full vocabulary: KL(original ‖ quant) and top-1 agreement on
our 122-row set (294,481 tokens: wiki text, math, code, prose, creative writing, tool calls, and five long rows of
13K-32K tokens), with 95 % bootstrap intervals. Our 4.0 bpw quant
([bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw](https://huggingface.co/bertholomus/DeepSeek-V4.1-Flash-1017-EXL3-4.0bpw))
on the same rows, for scale:

| Quant | short KL | short top-1 | long KL | long top-1 | wiki KL | wiki top-1 |
|---|---|---|---|---|---|---|
| 2.5 bpw (this) | 0.1305 [0.1209, 0.1406] | 92.01 % [91.10, 92.85] | 0.2876 [0.0652, 0.6093] | 87.07 % [76.31, 94.65] | 0.1558 [0.1428, 0.1695] | 87.41 % [86.05, 88.83] |
| 4.0 bpw | 0.0331 [0.0312, 0.0351] | 96.17 % [95.76, 96.55] | 0.1202 [0.0113, 0.2817] | 92.31 % [84.19, 97.47] | 0.0374 [0.0341, 0.0410] | 93.90 % [93.07, 94.74] |

By kind (KL / top-1), 2.5 bpw: code 0.0494 / 95.22 %, math 0.1425 / 91.51 %, prose 0.1254 / 91.77 %, creative
0.0623 / 91.09 %, tool calls 0.5948 / 79.43 %. The 4.0 bpw quant: code 0.0097 / 97.80 %, math 0.0391 / 95.74 %, prose
0.0248 / 96.50 %, creative 0.0109 / 96.02 %, tool calls 0.1951 / 87.77 %. Tool-call turns lose the most at 2.5 bpw; for agent work
that needs the closest match, the 4.0 bpw quant (four nodes) is the better pick. The full report, by depth too and
with the paired differences:
[results/served-v070/kl-report.md](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10/blob/v0.7.3/results/served-v070/kl-report.md)
in the recipe.

## Credits
- DeepSeek for DeepSeek-V4.1-Flash-1017.
- turboderp for ExLlamaV3 and the EXL3 format.
- Fabiano Anemone ([@anoane](https://github.com/anoane)) for ExLlamaV3's DeepSeek-V4 support (pull request 449).
- TensorFold (ashhart) for the engine this was served on.

Not affiliated with or endorsed by DeepSeek.
