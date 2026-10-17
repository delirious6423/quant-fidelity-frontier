---
language:
- en
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- jang
- jangh
- quantized
- apple-silicon
- moe
- vision
- video
- reasoning
- thinking
- agent
- tool-use
- mtp
- gptq
- imatrix
- qwen4_exp
---

<p align="center">
  <a href="https://mlx.studio"><img src="https://huggingface.co/JANGQ-AI/Naive-N0.5-Flash-JANGH2/resolve/main/vmlx-app.png" alt="MLX Studio / vMLX — run JANG models on Apple Silicon" width="820"></a>
</p>
<h3 align="center">Run JANG models in <a href="https://mlx.studio">MLX Studio</a> / <a href="https://vmlx.net">vMLX</a></h3>
<p align="center"><a href="https://huggingface.co/JANGQ-AI"><img src="https://huggingface.co/JANGQ-AI/Naive-N0.5-Flash-JANGH2/resolve/main/jangq-logo.png" alt="JANGQ-AI" width="300"></a></p>

> ⚠️ **Runtime: vMLX (Python) support for this bundle is not in a released build yet.** It was built and validated on
> a vMLX development build that runs `qwen4_exp` JANGH bundles with 4- and 6-bit expert widths. Released vMLX builds
> cannot load it.

# JANGQ-AI/Qwen3.8-Flash-Next-JANGH4

**Qwen3.8-Flash-Next at near-reference quality for 128 GB Macs.** About 74 GiB in memory; the 53.6 GiB n-gram
embedding table is read from the SSD on demand. Every reasoning effort closes as often as our 6-bit release, MTP is
as good as the bf16 MTP, and image and video input are calibrated.

A JANGH bundle of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): a 125B-parameter MoE
(512 routed experts, top-10 + a shared expert), gated-delta-net and sparse-attention layers, one MTP layer, image +
video input and a 51B-entry n-gram embedding table.

| part | format |
|---|---|
| routed experts (48 layers) | **JANGH 4-bit**; the down projection is **6-bit in 38 layers**, picked per projection by measured error per byte |
| attention (gated delta net, sparse attention, indexer), shared experts, `lm_head`, token embeddings, n-gram projections | affine **8-bit**, calibrated (importance matrix + AWQ) |
| MTP layer | affine 6-bit, calibrated from its own capture |
| vision tower | affine 8-bit, calibrated on images and video |
| routers, norms, mixers | kept in bf16 / fp32 |
| n-gram embedding table | affine 8-bit (group 32), on SSD, 16 rows × 180 bytes per token |

## Read this first: what "agreement" can reach
Scores are against the full bf16 model, run layer by layer from disk. Some next tokens are nearly tied, so any change
to the weights flips a few of them. We measured that on the **unquantized** model: relative noise of 0.001 on a single
tensor. That run is the **ceiling** in the tables below; the distance to the ceiling, not to 100%, is what the
quantization costs.

## Fidelity vs the bf16 model
510,954 teacher-forced positions on three held-out sets never used for calibration. Top-128 renormalized KL, scored
on the bundle exactly as stored.

| held-out set | positions | | median KL ↓ | mean KL ↓ | p90 / p95 / p99 ↓ | top-1 ↑ | top-5 ↑ | top-10 ↑ |
|---|---|---|---|---|---|---|---|---|
| general (215 seqs, 11 domains) | 268,727 | ceiling | 0.0005 | 0.037 | 0.037 / 0.112 / 0.776 | 96.47% | 99.81% | 99.91% |
| | | **JANGH4** | **0.0017** | **0.079** | **0.113 / 0.314 / 1.593** | **94.36%** | **99.61%** | **99.79%** |
| reasoning traces (low / medium / xhigh) | 42,481 | ceiling | 0.0000 | 0.015 | 0.006 / 0.013 / 0.290 | 98.03% | 99.90% | 99.96% |
| | | **JANGH4** | **0.0002** | **0.041** | **0.034 / 0.067 / 0.867** | **95.93%** | **99.77%** | **99.86%** |
| tool conversations (96) | 199,746 | ceiling | 0.0013 | 0.090 | 0.186 / 0.432 / 1.579 | 93.03% | 99.29% | 99.62% |
| | | **JANGH4** | **0.0042** | **0.222** | **0.568 / 1.275 / 3.522** | **88.66%** | **98.10%** | **98.91%** |

General set by domain, median KL · top-1 agreement:

| domain | positions | ceiling | **JANGH4** |
|---|---|---|---|
| coding | 39,155 | 0.0001 · 98.8% | **0.0005 · 97.1%** |
| math | 9,067 | 0.0001 · 99.0% | **0.0006 · 97.4%** |
| office / finance | 13,159 | 0.0002 · 97.9% | **0.0006 · 96.5%** |
| reasoning | 14,004 | 0.0000 · 96.9% | **0.0001 · 94.7%** |
| science | 4,156 | 0.0006 · 98.2% | **0.0035 · 95.8%** |
| long context | 79,784 | 0.0008 · 96.9% | **0.0023 · 95.2%** |
| networking | 6,458 | 0.0013 · 95.2% | **0.0029 · 93.5%** |
| cybersecurity | 15,683 | 0.0008 · 95.5% | **0.0030 · 92.9%** |
| creative | 10,029 | 0.0022 · 96.7% | **0.0098 · 93.6%** |
| terminal | 42,987 | 0.0011 · 94.2% | **0.0034 · 91.8%** |
| agentic | 34,245 | 0.0002 · 94.6% | **0.0007 · 91.6%** |

## Decisions: does it choose like the bf16 model?
| decision points (same choice as bf16) | ceiling | **JANGH4** |
|---|---|---|
| `</think>` closes, reasoning and general sets | 85 / 85 | **84 / 85** |
| "call a tool or answer?" at tool-call points, flipped vs bf16 | 9 / 112 | **16 / 112** |
| "call a tool or answer?" at answer points, flipped vs bf16 | 3 / 151 | **3 / 151** |

## Reasoning efforts (served, same 272 prompts as our 6-bit JANG_6S release)
Math, office/finance, networking, science, cybersecurity, coding and agentic prompts, each at every effort.

| effort | closes on its own: JANG_6S → **JANGH4** | exact-answer accuracy: JANG_6S → **JANGH4** | median tokens: JANG_6S → **JANGH4** |
|---|---|---|---|
| low | 95.7% → **95.7%** | 55.6% → **55.6%** | 1,077 → **1,107** |
| medium | 97.1% → **98.6%** | 55.6% → **58.3%** | 1,234 → **1,274** |
| xhigh | 97.0% → **97.0%** | 66.7% → **66.7%** | 930 → **828** |
| thinking off | 97.1% → **95.6%** | 72.2% → **72.2%** | 837 → **802** |

The efforts behave like the reference: xhigh spends far more tokens on hard prompts (p90 12,928 tokens vs 2,912 at
low; JANG_6S: 11,386 vs 3,165), and those long traces close at the reference's rate.

## MTP (speculative decoding)
| | **JANGH4 MTP (6-bit, calibrated)** | bf16 MTP |
|---|---|---|
| greedy acceptance (MTP guess = next token), 510,954 positions | **67.91%** | 67.85% |
| top-1 agreement with the bf16 MTP · median KL | **96.28% · 0.0006** | — |

The MTP proposal head ships as a calibrated 4-bit refit of this bundle's 8-bit `lm_head` (`mtp_draft/`, described
by `vmlx_mtp_proposal_head.json` at the bundle root).

## Image and video (served)
| check | **JANGH4** |
|---|---|
| invoice image: read every line item and compute the total ($1,994.75) | ✅ |
| video: counter value at start and end, background color change | ✅ |
| insurance premium word problem at low / medium / xhigh and thinking off | ✅ all four |

## Speed and memory (vMLX Python, M5 Max 128 GB, A/B/A against JANG_4M)
| | **JANGH4** (run A · run A′) | JANG_4M (affine 4-bit, same model) |
|---|---|---|
| decode, 256 tokens greedy (tok/s) | **49.97 · 48.66** | 50.89 |
| prefill, 4,096-token prompt (tok/s) | **1,628 · 1,353** ¹ | 1,186 |
| weights in memory / peak | 72.3 / 75.7 GiB | 71.0 / 74.3 GiB |

Same runtime build, one model loaded at a time, order JANGH4 → JANG_4M → JANGH4; each value is the median of the
warm runs (the first, cold run of each set is dropped). MTP off. Decode is on par with the plain affine 4-bit quant
(the 8-bit attention is the largest share of the bytes read per token); long-prompt prefill is **14-37% faster**.
¹ Run A′'s two warm prefill runs were 1,595 and 1,112 tok/s.

## JANGH
JANGH is our expert format: a fixed blockwise Hadamard-32 transform with no random signs and a closed-form codebook
per bit width (odd cubic at 2-4 bits, linear at 6 and 8), with one fp16 scale per row, stored bit-for-bit in MLX's own
packing layout so a runtime reuses MLX's kernel structure for decode and prefill. Rounding is GPTQ on each expert's own
input covariance.

For compatibility with runtimes already built against it, the on-disk identifiers keep their original names:
`config.json` → `jangtq` block (`version: 2`), per-module `"mode": "jangtq2"`, tensors `*.tq2_packed` /
`*.tq2_scales`, `jang_config.format = "jangtq2"`. They are not loadable by, and must not be routed to, JANGTQ v1
loaders.

## What's in the bundle
- **Vision + video**: Qwen3-VL tower, image and video processor configs.
- **MTP**: 1 layer plus the calibrated proposal head.
- **Thinking + agentic**:
  - Thinking is ON by default (the template opens `<think>`); reasoning is kept in history.
  - Reasoning efforts `low` / `medium` / `xhigh` (default `xhigh`) via the `reasoning_effort` chat-template kwarg.
  - Thinking off (`enable_thinking: false`) renders `<think>\n\n</think>\n\n`.
- **Tool calls**: XML function calls (`<tool_call>` / `<function=name>` / `<parameter=arg>…</parameter>`), declared as
  `tool_parser: qwen`; reasoning parser `qwen3`.
- **Expert widths differ per projection** inside a layer: gate/up 4-bit in all 48 layers; down 6-bit in layers 5,
  7-33, 35, 36, 38, 40-46 and 4-bit in the other 10. Each width is declared per module in `config.json`
  (`quantization`) and in `jang_config.expert_bits`; codebooks for 2/3/4/6 are declared in `config.json` → `jangtq`.
- **Self-describing**: per-module `quantization` map, the JANGH format block, capabilities, chat / effort / sampling /
  MTP / context stamps, 128 explicit n-gram table shard rules. Raw evaluation results are in `evaluation/`.
- Every shard is alignment-safe (zero-copy memory mapping). 65 shards, 3,072 tensors, 125.6 GiB on disk.

## Serving contract
- Sampling, thinking: `temperature=1.0, top_p=0.95, top_k=20`; thinking off: `temperature=0.7, top_p=0.8, top_k=20,
  presence_penalty=1.5`. No repetition penalty.
- EOS: `[248046, 248044]` · context: 262,144 native (YaRN-extensible to 1M)
- Memory: ~74 GiB of weights stay in memory (wire that much, not the folder size); the n-gram table is read from the
  SSD and is never loaded into memory.
- Continuous batching on the development build: keep at most 6 concurrent sequences (at 7+ the routed-expert path
  switches to its long-prompt kernels and slows down).

## Build details
- Source: `Qwen/Qwen3.8-Flash-Next` (bf16)
- Calibration: 604,179 tokens rendered with the model's own chat template: images and video (100k, 48 sequences),
  reasoning traces at every effort (82k), cybersecurity (78k), coding (76k), agentic and tool
  conversations (75k), terminal (70k), long context (37k), math (25k), science (25k), networking (17k), office and
  finance (12k), creative (7k). The held-out sets are disjoint from calibration.
- Experts: JANGH, GPTQ on per-expert centered covariances with act-order, all 48 MoE layers; widths placed per
  projection by measured held-out error per byte; unit-gain row scales (A/B'd against plain GPTQ scales; the
  difference was within noise)
- Everything else: importance-matrix weighted affine fit + AWQ from the same capture; vision tower from a dedicated
  image/video capture; MTP from its own capture

Quantized and validated by **Jinho Jang** — eric@jangq.ai
