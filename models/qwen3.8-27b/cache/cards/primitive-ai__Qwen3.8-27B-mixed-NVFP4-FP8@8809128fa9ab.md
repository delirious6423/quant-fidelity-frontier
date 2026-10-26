---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: transformers
tags:
- vllm
- compressed-tensors
- nvfp4
- fp8
- mixed-precision
- quantized
- speculative-decoding
- blackwell
- a100
thumbnail: https://huggingface.co/primitive-ai/Qwen3.8-27B-mixed-NVFP4-FP8/resolve/main/assets/banner.png
---

<p align="center">
  <img src="assets/banner.png" alt="Primitive — Qwen3.8-27B mixed NVFP4/FP8" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/size-~21%20GiB-F98424?style=flat-square" alt="21 GiB">
  <img src="https://img.shields.io/badge/vs%20BF16-2.5×%20smaller-F98424?style=flat-square" alt="2.5x smaller than BF16">
  <img src="https://img.shields.io/badge/accuracy-88.8%20vs%2089.3%20BF16-3DD13D?style=flat-square" alt="accuracy 88.8 versus BF16 89.3">
  <img src="https://img.shields.io/badge/throughput-2.0×%20@%20conc%2032-F98424?style=flat-square" alt="2x throughput at concurrency 32">
  <img src="https://img.shields.io/badge/runs%20on-Blackwell%20·%20A100-F98424?style=flat-square" alt="Runs on Blackwell and A100">
  <a href="https://primitive.com"><img src="https://img.shields.io/badge/web-primitive.com-lightgrey?style=flat-square" alt="primitive.com"></a>
</p>

<h3 align="center">Half the memory. Twice the throughput. The same answers.</h3>

<p align="center">
Mixed-precision quantization of <a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen/Qwen3.8-27B</a> down to
<b>~21 GiB</b>, <b>2.5× smaller than BF16</b>, that serves on <b>stock vLLM</b><br>
and still loads on an A100.
</p>

> **Revision 2026-10-26: `in_proj_qkv` and `in_proj_z` now share one global scale per layer.** vLLM fuses the two
> into one module and reads a single scale; layers 8–63 had stored one each, so one tensor of each pair loaded at the wrong
> scale. They were re-quantized from BF16 with the same quantizer, and nothing else changed. Drift from BF16 on 400
> traces BF16 wrote fell from +109.9 to +30.2 ×10⁻³ nats per token, and top-1 agreement rose from 89.0% to 94.3%.
> Rows not marked *fixed* were measured on the earlier revision.

---

## Why this quant

* 🗜️ **2.5× smaller than BF16, 28% smaller than Qwen's own FP8.** 20.7 GiB against
  51.7 GiB BF16 and 28.7 GiB FP8.
* 🎯 **Accuracy ties BF16 and eight of the nine builds measured.** 88.8 vs 89.3 pooled over 1,370
  items. Repeating one unchanged checkpoint moves that number by ~1.1 points, which is wider than
  the whole field's spread.
* ⚡ **2.0× throughput at concurrency 32, 2.5× at concurrency 1** versus BF16.
* 💸 **At the cheap end of the field on output tokens.** 651 per answer against a 625–818 spread,
  at a 99.7% completion rate. Both go straight to your serving bill.
* 🖥️ **Verified end to end on Ampere as well as Blackwell.** Native FP4/FP8 kernels on Blackwell,
  Marlin W·A16 fallback on A100, where the BF16 `lm_head` is what makes that path load.
* 🚀 **Speculative decoding survives quantization.** The in-checkpoint MTP head is
  preserved: up to **+101%** with a DSpark drafter.
* 🔧 **No calibration data, no custom runtime.** Weights-only round-to-nearest,
  `compressed-tensors`, stock `vllm serve`.

---

## Serve it

```bash
vllm serve primitive-ai/Qwen3.8-27B-mixed-NVFP4-FP8 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

---

## Measured

**1,370 items across fourteen public benchmarks.** A 1,170-item knowledge suite and a 200-item
tool-calling suite, under one fixed protocol for every row: `temperature 0.6` / `top_p 0.95` /
`top_k 20`, thinking forced on, a 16,384-token budget, concurrency 32, all rows back to back in one
sitting on one RTX PRO 6000 Blackwell.

| build | size | overall | knowledge | call | abstain | runs k/a | finished | out/answer | tok/s @ 32 | per-token latency | tok/s @ 1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [Qwen BF16](https://huggingface.co/Qwen/Qwen3.8-27B) | 51.7 G | 89.3 | 91.0 | 85.6 | 55.0 | 1/1 | 99.1% | 818 | 483 | 66.2 ms | 26.3 |
| [Qwen FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 28.7 G | 89.6 | 91.4 | 85.0 | 55.0 | 1/1 | 99.3% | 807 | 752 | 42.6 ms | 45.6 |
| [rdtand PrismaAQUA 5.5bit](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm) | 22.0 G | 89.4 | 91.2 | 85.6 | 52.5 | 1/1 | 99.6% | 793 | 927 | 34.5 ms | 58.3 |
| [unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) † | 21.8 G | 89.6 | 91.4 | 85.0 | 57.5 | 1/1 | 99.6% | 750 | 974 | 32.9 ms | 59.6 |
| **this repo, fixed** (separate sitting, vLLM 0.32.0) | **20.7 G** | **89.7** | 91.4 | 85.4 | 58.3 | 1/3 | 99.4% | 828 | — | — | — |
| this repo, earlier revision | 20.7 G | 88.8 | 90.4 | 85.5 | 54.5 | 12/5 | 99.7% | 651 | 908 | 35.3 ms | 66.4 |
| [RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) † | 20.4 G | 89.5 | 91.3 | 85.0 | 52.5 | 3/1 | 99.3% | 782 | 985 | 32.5 ms | 63.7 |
| [cyankiwi AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) | 19.6 G | 89.8 | 91.3 | 85.6 | 62.5 | 1/1 | 99.4% | 718 | 956 | 33.5 ms | **69.3** |
| [sakamakismile MTP-NVFP4](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) | 19.1 G | 89.1 | 90.8 | 84.4 | 57.5 | 1/1 | 99.8% | 625 | 965 | 33.1 ms | 67.8 |
| [gittensor NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) † | **17.5 G** | 90.1 | 91.6 | 86.2 | 60.0 | 1/1 | 99.7% | 809 | 1151 | 27.8 ms | **69.3** |
| *this repo, its single leg from the sitting above* | — | 88.3 | 89.9 | 85.6 | 52.5 | 1/1 | 99.7% | 651 | 908 | 35.3 ms | — |

**`overall` is one number over both suites**: the 1,170 knowledge and 200 tool-calling items pooled
as 1,370, weighted **85.4%** and **14.6%** by item count. Repeat runs of one checkpoint moved it by
about a point, so treat a gap below **1.0** as a tie.

<details>
<summary>Where the tool-calling half goes: call vs abstain</summary>

| build | agentic | call | abstain |
|---|---|---|---|
| [Qwen BF16](https://huggingface.co/Qwen/Qwen3.8-27B) | 79.5 | 85.6 | 55.0 |
| [Qwen FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) | 79.0 | 85.0 | 55.0 |
| [rdtand PrismaAQUA 5.5bit](https://huggingface.co/rdtand/Qwen3.8-27B-PrismaAQUA-5.5bit-vllm) | 79.0 | 85.6 | 52.5 |
| [unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) † | 79.5 | 85.0 | 57.5 |
| **this repo** | **79.3** | 85.5 | 54.5 |
| [RadixArk NVFP4](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) † | 78.5 | 85.0 | 52.5 |
| [cyankiwi AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.8-27B-AWQ-INT4) | 81.0 | 85.6 | 62.5 |
| [sakamakismile MTP-NVFP4](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4) | 79.0 | 84.4 | 57.5 |
| [gittensor NVFP4-RTX5090](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) † | 81.0 | 86.2 | 60.0 |

`call` is accuracy on the 160 rows that require a call; `abstain` is the 40 whose correct action is
to call **nothing**. Benchmarks: BFCL v4 (30, including `irrelevance`), xLAM/APIGen (45), ToolACE
(45), Glaive v2 (40), nvidia When2Call (40, the abstention rows). Tool schemas go in the system prompt
and the model answers with a JSON array of calls, the same way for every model. An item counts only
if every expected call is present with acceptable arguments and no call is invented.

**Abstention is the weak axis for every model we have measured** (52–82%), so a build can look strong
on `overall` and still over-call.

</details>

**† These three checkpoints ship a `kv_cache_scheme` that enables an FP8 KV cache automatically.**
That is a real memory advantage, and it also lifts their throughput against builds running a BF16 KV
cache, so the speed columns are not purely a comparison of weights.

**Read the `runs` column before the ordering.** This build's figures are means of twelve knowledge
runs; every community build except RadixArk is a single draw. Re-running RadixArk interleaved with
this build on one host settled its 92.0 to **91.3**, and left the two 0.35 apart on knowledge. Since
one unchanged checkpoint moves `overall` by ~1.1 points, the 1.3 points separating this whole field
is not a ranking. Decide on size, output tokens and hardware instead.

**A100 40 GB.** Everything runs Marlin W·A16 there, and a build whose `lm_head` is quantized will
not load on that path at all, which is why this one leaves it at BF16:

| build | knowledge | call | abstain | finished |
|---|---|---|---|---|
| this repo, earlier revision | 90.5 | 84.8 | 60.8 | 99.8% |
| **this repo, fixed** | **90.6** | 86.2 | 56.7 | 99.4% |

Both rows were measured with the same harness on the same A100, one after the other: knowledge is two runs for the
earlier revision and one for the fixed build, tool calling three runs each. Throughput there was 790 tok/s at concurrency 32 and 53.9 single-stream, measured on the earlier revision.

Repeat measurements of this exact checkpoint span **89.7 to 91.0** on the knowledge suite across
**twelve runs** in five sittings, sd 0.39, and its tool-calling half moves about a point over five.
Throughput moved −4.5% between two of those sittings, which is why every row of a table here is
measured back to back rather than assembled from separate runs.

---

## Speculative decoding

<p align="center">
  <img src="assets/speculative.png" alt="Throughput gain over no drafting: MTP +59% at concurrency 1 and +61% at 16; DSpark drafter +101% at concurrency 1 and +83% at 16" width="100%">
</p>

A separate measurement run from the table above. The baselines differ, so the
two are never plotted together.

| speculative config | conc 1 | conc 16 |
|---|---|---|
| none | 59.4 tok/s | 396.6 tok/s |
| MTP (`{"method":"mtp","num_speculative_tokens":2}`) | +59% | +61% |
| DSpark drafter (`RadixArk/Qwen3.8-27B-DSpark`, n=4) | **+101%** | **+83%** |

> **Gotcha:** the RadixArk drafter's `config.json` needs
> `architectures: ["Qwen3DSparkModel"]`. As shipped it says `DSparkDraftModel`,
> which vLLM routes to the DeepSeek-V4 class and crashes on `hc_mult`.

---

## Reasoning effort is a real cost lever

Qwen3.8's chat template exposes `reasoning_effort`, and its default is `xhigh`. Measured on this
checkpoint, same 1,170 items, same sitting:

| `reasoning_effort` | acc | finished | out/answer | tok/s @ conc 32 | per-token latency |
|---|---|---|---|---|---|
| `xhigh` (the default) | **90.5** | 99.7% | 613 | 894 | 35.8 ms |
| `medium` | 89.7 | 99.8% | 587 | 901 | 35.5 ms |
| `low` | 89.3 | 99.9% | **528** | **1035** | **30.9 ms** |

```bash
curl localhost:8000/v1/chat/completions -d '{
  "model": "...", "messages": [...],
  "chat_template_kwargs": {"reasoning_effort": "low"}
}'
```

**`low` costs 1.2 accuracy points and returns about 25% cheaper answers.** 14% fewer output tokens
*and* 14% lower per-token latency, which compound. On a throughput-bound deployment where a point of
accuracy is worth less than a quarter of the serving bill, that is the setting to run.

**`medium` is not worth using.** It saves 4% of tokens for 0.8 points, the worst trade of the three.

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
| [Ornith-1.5-35B-A3B](https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8) | 35 B MoE | 22.6 GiB | 88.7 | 91.7 | 74.4 | 60.0 | 99.3% | 760 tok | 1469 |
| [Muse-Glimmer-30B](https://huggingface.co/primitive-ai/Muse-Glimmer-30B-mixed-NVFP4) | 30 B MoE | 20.4 GiB | 86.6 | 88.8 | 78.6 | 54.5 | 99.7% | 800 tok | 1176 |
| **Qwen3.8-27B** (this repo) | 27 B dense | 20.7 GiB | **88.8** | 90.4 | 85.5 | 54.5 | 99.7% | 651 tok | 908 |
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

| benchmark | Laguna-XS-2.1 | Nemotron-3.5-Lightning-30B-A3B | Ornith-1.5-35B-A3B | Muse-Glimmer-30B | **Qwen3.8-27B** | Laguna-S-2.1 | Qwen3.8-Flash-Next |
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
| all 64 layers' MLP `gate/up/down_proj`; linear-attn `in_proj_qkv`/`in_proj_z` on layers 8–63 | NVFP4 (group 16) |
| full-attn `q/k/v/o_proj`; linear-attn `in_proj_qkv`/`in_proj_z` on layers 0–7; `linear_attn.out_proj` | FP8 E4M3 (channel) |
| `lm_head`, linear-attn `in_proj_a`/`in_proj_b` gates, conv1d, norms, embeddings, vision tower, MTP head | BF16 |

Weights-only round-to-nearest, no calibration.

---

<p align="center">
  <br>
  <img src="assets/primitive-logo.png" alt="Primitive" width="34"><br>
  <sub>
    <a href="https://primitive.com"><b>primitive</b></a> ·
    <a href="https://huggingface.co/primitive-ai">more models</a> ·
    inference economics for production LLM systems
  </sub>
</p>
