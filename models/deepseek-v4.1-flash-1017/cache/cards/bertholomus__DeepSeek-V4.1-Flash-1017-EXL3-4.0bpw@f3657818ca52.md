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

# DeepSeek-V4.1-Flash-1017 — EXL3 4.0 bpw

EXL3 quant of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision `1bae0e10`) by BertholomusAI. Model and weights © DeepSeek under the MIT License.

## Bits
| Part | Format |
|---|---|
| Routed experts | EXL3 4.0 bpw |
| Attention, shared experts, KV compressor, indexer | EXL3 6 bpw (hq) |
| Output head | EXL3 6 bpw |
| MTP / DSpark drafter layers | EXL3 4 bpw |
| Embeddings, vision tower | 16-bit, unchanged |
| Overall | 4.02 bpw over the quantized weights, about 275 GB |

Converter: ExLlamaV3 1.6.0 (codebook mul1), calibration 128 rows × 2048 tokens.

The Engram tables are not in this repository: serve them from the original checkpoint's shards 47 and 48, as the
recipe does.

## Served
Served on four GB10 (DGX Spark class) nodes with TensorFold tensor parallel 4 over RDMA. Gates passed on that lane: exact chat 71/71, tool calls, drafted == serial, 16-request bursts == solo, vision, 1M-context needles 3/3.

| Row (HTTP, tok/s) | TP4, 4 × GB10 |
|---|---|
| decode, 1 stream: code / prose / structured | 136 / 79 / 187 |
| 2 streams / 4 streams | 102 / 142 |
| 4 streams sustained | 183 |
| prefill 8K / 128K | 2,793 / 2,679 |
| decode after a 128K prompt | 140 |

Recipe (the launch line, the gates and how to ready this download):
[bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp4-4xgb10), v0.2.0.
This repository ships no chat template. The recipe's `tools/serving_1017.py` adds ours and six tensors under the
names and types the engine reads (the MTP markov head, the Engram q/k as F32).

## Quality
KL divergence against the native 1017 model: not measured yet; this card will be updated.

## Credits
- DeepSeek for DeepSeek-V4.1-Flash-1017.
- turboderp for ExLlamaV3 and the EXL3 format.
- TensorFold (ashhart) for the engine this was served on.

Not affiliated with or endorsed by DeepSeek.
