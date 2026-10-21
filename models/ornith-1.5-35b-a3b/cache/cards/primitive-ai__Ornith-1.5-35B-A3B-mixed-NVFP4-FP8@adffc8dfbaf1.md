---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- vllm
- compressed-tensors
- nvfp4
- fp8
- mixed-precision
- quantized
- moe
- multimodal
- blackwell
thumbnail: https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8/resolve/main/assets/banner.png
---

<p align="center">
  <img src="assets/banner.png" alt="Primitive — Ornith-1.5-35B-A3B mixed NVFP4/FP8" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/size-22.6%20GiB-F98424?style=flat-square" alt="size 22.6 GiB">
  <img src="https://img.shields.io/badge/vs%20BF16-3.0×%20smaller-F98424?style=flat-square" alt="vs BF16 3.0x smaller">
  <img src="https://img.shields.io/badge/knowledge-91.7%20vs%2092.0%20BF16-3DD13D?style=flat-square" alt="knowledge 91.7 versus BF16 92.0">
  <img src="https://img.shields.io/badge/tool%20calling-69.0%20vs%2077.0%20BF16-lightgrey?style=flat-square" alt="tool calling 69.0 versus BF16 77.0">
  <img src="https://img.shields.io/badge/throughput-1.72×%20@%20conc%2032-F98424?style=flat-square" alt="throughput 1.72x at concurrency 32">
  <img src="https://img.shields.io/badge/vision-untouched-F98424?style=flat-square" alt="vision tower untouched">
  <a href="https://primitive.com"><img src="https://img.shields.io/badge/web-primitive.com-lightgrey?style=flat-square" alt="primitive.com"></a>
</p>

<h3 align="center">A third of the size, level with BF16 on knowledge. Read the tool-calling note.</h3>

<p align="center">
Mixed-precision quantization of <a href="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B">ornith-ai/Ornith-1.5-35B-A3B</a> to <b>22.6 GiB</b><br>
(<b>3.0× smaller than BF16</b>) on <b>stock vLLM</b>, with the vision tower and the<br>
in-checkpoint MTP block left intact.
</p>

---

## Why this quant

* 🗜️ **3.0× smaller than BF16, 39% smaller than the official FP8.** 22.6 GiB against 67.0 GiB and
  36.7 GiB.
* 🎯 **Closest to BF16 on knowledge of any quant of this model measured.** 91.7 against BF16's 92.0
  on a 1,170-item suite, against the official FP8's 91.2 and NVFP4's 90.9. The FP8 ties BF16 on this
  suite too; this build is nearer.
* 🛠️ **Tool calling is this build's weaker axis, and there is a sibling build for that.** 71.8 over
  **six** runs against BF16's 75.1 over four: real but marginal, and not distinguishable from either
  official quant. Pooled over both suites this build scores **88.7**, ahead of the official NVFP4
  (88.3) and behind the FP8 (89.0). If tool calling is your workload, take
  [our agentic build](https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-agentic-NVFP4-FP8):
  same size, **74.0** on that suite with a third of the run-to-run spread.
* ⚡ **1.72× BF16 throughput at concurrency 32, 1.22× single-stream.** 1,454 tok/s against 845.
* 👁️ **The vision tower is untouched.** Vision, the projector and norms stay BF16: a VLM loses its
  visual grounding before its words.
* 🚀 **The MTP block is preserved**, so speculative decoding remains available. The tooling drops it
  by default, and dropping it is silent.
* 🖥️ **BF16 `lm_head`.** It costs a little size and speed, and it is what keeps the Ampere
  (Marlin W·A16) path loadable; a quantized head crashed that path on a sibling architecture.
* 🔧 **Stock vLLM, `compressed-tensors`, no patches.**

---

## Serve it

```bash
vllm serve primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8 --max-model-len 32768
```

Text and image inputs both work. For long-context serving, `--kv-cache-dtype fp8` buys a large
increase in KV capacity at negligible throughput cost; it is left off here so the numbers below
describe the checkpoint rather than a serving flag.

---

## Measured

**1,370 items across fourteen public benchmarks.** A 1,170-item knowledge suite and a 200-item
tool-calling suite, under one fixed protocol for every row: `temperature 0.6` / `top_p 0.95` /
`top_k 20`, thinking forced on, a 16,384-token budget, concurrency 32 plus a 60-item single-stream
subset, all rows back to back in one sitting on one RTX PRO 6000 Blackwell.

| build | size | overall | knowledge | call | abstain | runs k/a | finished | out/answer | tok/s @ 32 | per-token latency |
|---|---|---|---|---|---|---|---|---|---|---|
| BF16 (reference) | 67.0 G | 89.5 | 92.0 | 78.6 | 61.3 | 1/4 | 99.2% | 708 | 845 | 37.9 ms |
| [ornith-ai FP8](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8) | 36.7 G | 88.8 | 91.2 | 79.1 | 58.1 | 1/4 | 99.4% | 679 | 1239 | 25.8 ms |
| **this repo** | **22.6 G** | **88.7** | 91.7 | 74.4 | 60.0 | 3/6 | 99.3% | 760 | 1469 | 21.8 ms |
| [ornith-ai NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) | 21.8 G | 88.3 | 90.9 | 76.1 | 61.3 | 1/4 | 99.6% | 686 | 1681 | 19.0 ms |
| *this repo, knowledge half re-measured* | — | — | 91.9 | — | — | 1/1 | 99.1% | 772 | 1473 | 21.7 ms |

**`overall` is one number over both suites**: the 1,170 knowledge and 200 tool-calling items pooled
as 1,370, weighted **85.4%** and **14.6%** by item count. Repeat runs of one checkpoint moved it by
about a point, so treat a gap below **1.0** as a tie.

<details>
<summary>Where the tool-calling half goes: call vs abstain</summary>

| build | agentic | call | abstain |
|---|---|---|---|
| BF16 (reference) | 75.1 | 78.6 | 61.3 |
| [ornith-ai FP8](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-FP8) | 74.9 | 79.1 | 58.1 |
| **this repo** | **71.8** | 74.4 | 60.0 |
| [ornith-ai NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-NVFP4) | 73.1 | 76.1 | 61.3 |

`call` is accuracy on the 160 rows that require a call; `abstain` is the 40 whose correct action is
to call **nothing**. Benchmarks: BFCL v4 (30, including `irrelevance`), xLAM/APIGen (45), ToolACE
(45), Glaive v2 (40), nvidia When2Call (40, the abstention rows). Tool schemas go in the system prompt
and the model answers with a JSON array of calls, the same way for every model. An item counts only
if every expected call is present with acceptable arguments and no call is invented.

**Abstention is the weak axis for every model we have measured** (52–82%), so a build can look strong
on `overall` and still over-call.

</details>

The last row is a drift control: the same checkpoint re-measured after two hours of other work came
back within **1.3% on throughput and 0.4 points on knowledge**.

**Every `agentic` figure above is a mean of repeated runs, and on this model that is essential.**
Quantized builds of Ornith carry two to three times BF16's run-to-run spread on the 200-item
tool-calling suite: six runs of this checkpoint span **67.5 to 75.5 (sd 3.1)**, against BF16's 1.4
and the official quants' 1.2-1.7. An earlier version of this card reported a single run (69.0) as
though it were the number, and drew a conclusion from it that the data does not support. Corrected:

| build | tool calling | n | sd |
|---|---|---|---|
| BF16 | 75.1 | 4 | 1.4 |
| ornith-ai FP8 | 74.9 | 4 | 1.7 |
| ornith-ai NVFP4 | 73.1 | 4 | 1.2 |
| **this repo** | **71.8** | 6 | 3.1 |

This build is **3.4 points below BF16** on tool calling (t=2.3, so real but marginal) and **not
statistically distinguishable from either official quant** -- the NVFP4 by 1.4 points at t=1.0. On the
pooled score it leads that NVFP4 (88.7 vs 88.3). Read differences under about 2 points here as noise.

🛠️ **If tool calling is your workload, use
[our agentic build](https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-agentic-NVFP4-FP8)
instead.** Same size and format, calibrated for tool use: **74.0** on this suite, a tie with BF16's
75.1, with a third of this build's run-to-run spread and 0.2 points less general knowledge.

**Read this honestly: the official NVFP4 build is faster and smaller than this one.** 15.6% more
aggregate throughput, 27% more single-stream, and 0.75 GiB less on disk. It gets there partly by
quantizing the 508 M-parameter `lm_head`, which is hit on every decoded token; this build keeps that
head at BF16. So the trade is roughly: their build if you want maximum speed on Blackwell, this one
if you want the closest accuracy to BF16 and an output head that keeps the Ampere fallback path
loadable. All four accuracy figures sit within about a point of each other, which is this suite's
run-to-run spread, so treat that column as a tie and choose on the others.

---

## Comparable with our other models

Accuracy numbers move for reasons that have nothing to do with the model: a shorter token budget, a
different `temperature`, or whether the model was allowed to reason at all. So every number in this
table, on this card and on our other cards, comes from one fixed protocol.

The same 1,370 items: a 1,170-item knowledge suite (MMLU-Pro, ARC-Challenge, HellaSwag, WinoGrande,
CommonsenseQA, BoolQ, OpenBookQA, GSM8K, MATH-500) and a 200-item tool-calling suite (BFCL v4,
xLAM/APIGen, ToolACE, Glaive v2, nvidia When2Call). `temperature 0.6`, `top_p 0.95`, `top_k 20`,
thinking forced on, a 16,384-token budget, no reasoning parser, scoring the last `ANSWER:` in the
reply. Concurrency 32 on one RTX PRO 6000 Blackwell, each model's rows in one sitting. Auto-scored,
no LLM judge. Both halves are means of at least three runs per build.


| model | shape | size | overall | knowledge | call | abstain | finished | out | tok/s @ 32 |
|---|---|---|---|---|---|---|---|---|---|
| [Laguna-XS-2.1](https://huggingface.co/primitive-ai/Laguna-XS-2.1-mixed-NVFP4-MXFP8) | 31 B MoE | 19.3 GiB | 81.7 | 83.8 | 68.4 | 73.5 | 98.9% | 1097 tok | 1523 |
| [Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/primitive-ai/Nemotron-3.5-Lightning-30B-A3B-mixed-INT4-INT8) | 30 B MoE+Mamba | 19.2 GiB | 87.1 | 87.9 | 85.4 | 70.5 | 97.9% | 1429 tok | 2204 |
| **Ornith-1.5-35B-A3B** (this repo) | 35 B MoE | 22.6 GiB | **88.7** | 91.7 | 74.4 | 60.0 | 99.3% | 760 tok | 1469 |
| [Muse-Glimmer-30B](https://huggingface.co/primitive-ai/Muse-Glimmer-30B-mixed-NVFP4) | 30 B MoE | 20.4 GiB | 86.6 | 88.8 | 78.6 | 54.5 | 99.7% | 800 tok | 1176 |
| [Qwen3.8-27B](https://huggingface.co/primitive-ai/Qwen3.8-27B-mixed-NVFP4-FP8) | 27 B dense | 20.7 GiB | 88.8 | 90.4 | 85.5 | 54.5 | 99.7% | 651 tok | 908 |
| [Laguna-S-2.1](https://huggingface.co/primitive-ai/Laguna-S-2.1-mixed-NVFP4-MXFP8) | 110 B MoE | 64.0 GiB | 84.3 | 87.1 | 64.6 | 81.0 | 97.3% | 995 tok | 670 |
| [Qwen3.8-Flash-Next](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8) | 180 B MoE (6 B active) | 183.7 GB | 90.3 | 92.2 | 84.8 | 56.7 | 99.5% | 686 tok | 816 |

**Read `overall` with `finished`.** `overall` scores an answer that overran the token budget as wrong,
but it cannot say whether the model needed the room or failed to stop; `finished` and `out` separate
those. A gap under **1.0** is a tie. The `tok/s` column comes from each model's own sitting and drifts
a few percent between sittings, so read it as a bracket.

**`call` and `abstain` are the tool-calling suite's two halves, reported separately.** `call` is
accuracy on the 160 items that require a tool call; `abstain` is the 40 whose correct action is to
call nothing. They used to be pooled into one `agentic` number, and the pooling misled: a model with
ordinary call accuracy and unusual abstention discipline outscored models that are better at
actually making calls. Weight them by your own workload's mix.

<details>
<summary>Per-benchmark detail, both halves</summary>

| benchmark | Laguna-XS-2.1 | Nemotron-3.5-Lightning-30B-A3B | **Ornith-1.5-35B-A3B** | Muse-Glimmer-30B | Qwen3.8-27B | Laguna-S-2.1 | Qwen3.8-Flash-Next |
|---|---|---|---|---|---|---|---|
| **knowledge** | | | | | | | |
| mmlu_pro | 79.0 | 82.0 | 89.5 | 89.0 | 89.5 | 81.0 | 89.0 |
| math_500 | 76.0 | 80.0 | 88.0 | 74.0 | 82.0 | 83.0 | 86.0 |
| gsm8k | 97.0 | 96.0 | 99.0 | 98.0 | 98.0 | 99.0 | 97.0 |
| arc_challenge | 94.0 | 97.3 | 98.7 | 98.0 | 98.0 | 96.0 | 98.7 |
| hellaswag | 70.7 | 78.0 | 86.7 | 83.3 | 82.0 | 76.7 | 88.7 |
| winogrande | 84.7 | 89.3 | 89.3 | 90.7 | 89.3 | 86.0 | 95.3 |
| commonsenseqa | 80.8 | 83.3 | 86.7 | 82.5 | 84.2 | 85.8 | 86.7 |
| boolq | 90.0 | 92.0 | 92.0 | 90.0 | 91.0 | 92.0 | 94.0 |
| openbookqa | 94.0 | 99.0 | 98.0 | 97.0 | 97.0 | 94.0 | 96.0 |
| **tool calling** | | | | | | | |
| bfcl_v4 | 83.3 | 96.7 | 76.7 | 86.7 | 96.7 | 90.0 | 96.7 |
| xlam | 64.4 | 73.3 | 55.6 | 66.7 | 77.8 | 48.9 | 82.2 |
| toolace | 37.8 | 71.1 | 71.1 | 64.4 | 68.9 | 48.9 | 73.3 |
| glaive | 87.5 | 97.5 | 92.5 | 100.0 | 100.0 | 85.0 | 100.0 |
| when2call | 75.0 | 77.5 | 52.5 | 55.0 | 55.0 | 75.0 | 55.0 |

</details>

## What's quantized to what

| tensors | format |
|---|---|
| routed expert projections (all 40 layers × 256 experts) | NVFP4 (group 16) |
| full-attention `q/k/v/o_proj`; linear-attention `in_proj_qkv`/`in_proj_z`/`out_proj`; shared expert | FP8 E4M3 (channel) |
| `lm_head`, embeddings, router, `shared_expert_gate`, the linear-attention state path (`in_proj_a`/`in_proj_b`/`conv1d`/`A_log`/`dt_bias`), norms, vision tower, MTP block | BF16 |

Weights are quantized with GPTQ against a text-only calibration set; the vision tower is not
quantized, so image inputs are unaffected. Activations for the NVFP4 group are quantized dynamically
at runtime by vLLM.

---

<p align="center">
  <br>
  <sub>
    <a href="https://primitive.com"><b>primitive</b></a> ·
    <a href="https://huggingface.co/primitive-ai">more models</a> ·
    inference economics for production LLM systems
  </sub>
</p>
