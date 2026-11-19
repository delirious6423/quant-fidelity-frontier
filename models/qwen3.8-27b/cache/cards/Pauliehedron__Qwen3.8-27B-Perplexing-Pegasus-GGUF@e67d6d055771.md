---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - quantized
  - llama.cpp
  - imatrix
  - qwen3.8
language:
  - en
  - zh
---
# Pauliehedron "Meter Stick" Qwen3.8-27B · Perplexing-Pegasus

The "Perplexing-Pegasus" merge of Qwen 3.8 27B in a
uniform codec layout (identical tensor-block structure, Q8_0 MTP head), so
rungs differ only in quant format. Based on the
[AveLabs/Qwen3.8-27B-Perplexing-Pegasus-GGUF](https://huggingface.co/AveLabs/Qwen3.8-27B-Perplexing-Pegasus-GGUF)
merge. Pegasus is a variant of Qwen 3.8 27B — the vanilla Pauliehedron
Qwen 3.8 27B builds are the reference — and these rungs are quantized on
the same meter-stick shapes so it can be compared like-for-like.

## Files

| File | Quant | Note |
|------|-------|------|
| `Qwen3.8-27B-Perplexing-Pegasus-IQ4_NL-uniform-imx.gguf` | IQ4_NL | 4-bit imatrix quant |
| `Qwen3.8-27B-Perplexing-Pegasus-IQ4_XS-uniform-imx.gguf` | IQ4_XS | 4-bit imatrix quant |
| `Qwen3.8-27B-Perplexing-Pegasus-Q4_K_M-uniform-imx.gguf` | Q4_K_M | 4-bit imatrix K-quant |
| `Qwen3.8-27B-Perplexing-Pegasus-Q5_K_M-uniform-imx.gguf` | Q5_K_M | 5-bit imatrix K-quant |
| `Qwen3.8-27B-Perplexing-Pegasus-Q6_K-uniform-imx.gguf` | Q6_K | 6-bit imatrix K-quant |

## Container

All rungs share one uniform container (`gmlx validate`): 866 tensors, 65
blocks, 360 F32 small tensors (norms/rope/freqs), and the Q8_0 MTP head
(blk.64, 8 tensors — identical on every rung). The per-rung codec mix:

| Rung | main body | tail |
|------|-----------|------|
| IQ4_NL | IQ4_NL ×433 | Q5_K ×64 + Q6_K ×1 |
| IQ4_XS | IQ4_XS ×433 | Q5_K ×64 + Q6_K ×1 |
| Q4_K_M | Q4_K ×433 | Q6_K ×65 |
| Q5_K_M | Q5_K ×433 | Q6_K ×65 |
| Q6_K | Q6_K ×498 | — |

## Rebuild

Base: [AveLabs/Qwen3.8-27B-Perplexing-Pegasus-GGUF](https://huggingface.co/AveLabs/Qwen3.8-27B-Perplexing-Pegasus-GGUF)
(BF16 shards) → `convert_hf_to_gguf.py` (f16) → `llama-imatrix`
(importance-matrix calibration) → `llama-quantize --imatrix` per quant.
The calibration imatrix bin is numeric and contains no source text; the
calibration corpus used to produce that bin is retained locally and not
included here, so byte-exact re-derivation additionally needs that corpus.

## Measured (our host)

M1 Studio (Apple M1 Max, 32-core GPU, 64 GB unified memory). Speeds are
local measurements with [llama.cpp](https://github.com/ggml-org/llama.cpp)
/ [gmlx (asher/gmlx)](https://github.com/asher/gmlx) on this host — your
results will differ by hardware, context depth, and server load.

KLD cells: `llama-perplexity --kl-divergence`, corpus
[wiki.test.raw](https://huggingface.co/datasets/Salesforce/wikitext) (WikiText `wikitext-2-raw-v1` test split), n_ctx 8192 · 8 chunks, reference = the family's f16 logits (AveLabs
BF16 keeper). These cells are the reference curve the Pauliehedron uniform
quantizations of Qwen 3.8 27B are measured against. Speed measurements are
in progress.

| Quant | KLD (mean ± σ) |
|------|---------|
| `IQ4_NL` | 0.054082 ± 0.004097 |
| `Q4_K_M` | 0.043857 ± 0.003303 |
| `Q5_K_M` | 0.021354 ± 0.002800 |
| `IQ4_XS` | 0.055525 ± 0.004106 |
| `Q6_K` | 0.009171 ± 0.001557 |

JevBench (JEV public sample) systemone accuracy, same 260-cell basis,
route-a/b @ed50fe0 perm-a (gmlx wire). The 30B-switch rows are the
2026-10-24 full-set @8K run.

| Family / rung | systemone acc |
|---|---|
| **Pegasus IQ4_NL (this repo)** | **0.7654** |
| Pauliehedron Qwen3.8-27B meter (IQ4_NL) | 0.7538 |
| granite-switch-4.1-30b Q4_K_M / Q6_K / Q3_K_M | 0.7269 / 0.7231 / 0.7000 |
| granite-switch-4.1-8b Q6_K / IQ4_NL / Q4_K_M | 0.6885 / 0.6615 / 0.6538 |

The remaining Pegasus rungs' JevBench cells are in progress (same battery).

## License

Derived from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
(Apache-2.0). Redistribution permitted with attribution; the base's license
terms require distributing a copy of the License with derivative works, so
the Apache-2.0 text is included in this repo as `LICENSE`. The evaluation
corpus is the public Salesforce/WikiText dataset — see the
[dataset card](https://huggingface.co/datasets/Salesforce/wikitext) for its terms; the
corpus itself is not redistributed here.