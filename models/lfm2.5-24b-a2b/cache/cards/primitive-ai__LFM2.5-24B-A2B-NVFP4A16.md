---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: vllm
language:
- en
- ar
- zh
- fr
- de
- ja
- ko
- es
- pt
- it
- vi
- th
- id
- hi
- ru
- pl
tags:
- vllm
- compressed-tensors
- nvfp4
- quantized
- moe
- lfm2.5
- liquid
- reasoning
- agentic
- tool-calling
- single-gpu
---

<p align="center">
  <img src="https://img.shields.io/badge/size-14.95%20GB-F98424?style=flat-square" alt="size 14.95 GB">
  <img src="https://img.shields.io/badge/vs%20BF16-3.2×%20smaller-F98424?style=flat-square" alt="3.2x smaller than BF16">
  <img src="https://img.shields.io/badge/drift%20from%20BF16-below%20the%20FP8%20release-3DD13D?style=flat-square" alt="drift from BF16 below the FP8 release">
  <img src="https://img.shields.io/badge/accuracy-level%20with%20BF16-3DD13D?style=flat-square" alt="accuracy level with BF16">
  <img src="https://img.shields.io/badge/KV%20pool-3.68M%20tokens-F98424?style=flat-square" alt="KV pool 3.68M tokens">
  <img src="https://img.shields.io/badge/tested%20on-RTX%20PRO%206000%20Blackwell-lightgrey?style=flat-square" alt="tested on RTX PRO 6000 Blackwell">
  <a href="https://primitive.com"><img src="https://img.shields.io/badge/web-primitive.com-lightgrey?style=flat-square" alt="primitive.com"></a>
</p>

<h3 align="center">4-bit expert weights, 16-bit everywhere else: closer to BF16 than Liquid's own FP8, at 60% of its size.</h3>

<p align="center">
NVFP4 weight-only build of <a href="https://huggingface.co/LiquidAI/LFM2.5-24B-A2B">LiquidAI/LFM2.5-24B-A2B</a> at
<b>14.95 GB</b>, on one GPU.<br>
The routed experts store NVFP4 weights and compute in 16-bit (W4A16). The short-convolution blocks, attention, the two
dense layers, embeddings and routers stay BF16.
</p>

> **Accuracy suites cannot separate these builds.** On our 1,370 items every build in the table below, BF16 included,
> scores within 0.7 points of every other. So the table also scores each build against BF16's own
> output: 400 BF16 reasoning traces, the extra negative log-likelihood each quant assigns them per token, and how often
> its most likely next token is BF16's. That measurement is deterministic, and on it the builds range from
> +8.6 to +66.8.

---

## Why this quant

* 🎯 **Closest to BF16 of the builds we measured.** +8.6×10⁻³ nats per token above BF16 on BF16's own
  traces, with top-1 agreement 96.1%. Liquid's FP8 release scores +10.0 and
  96.0%; the published W4A4 NVFP4 builds score +20.6 to +66.8.
* 🗜️ **14.95 GB against 47.94 GB in BF16 and 24.75 GB for the FP8 release.** On one
  96 GB card that leaves a 3,682,347-token KV pool at 32K context, against 2,077,736 for BF16.
* ⚡ **218 tok/s single-stream and 1,395 at concurrency 32**, against 204 and 1,131 for the FP8 release and 179 and 861 for BF16. The W4A4 builds run
  1,639 to 1,923 tok/s at 32: their expert activations are 4-bit too, which is faster on FP4 tensor cores.
  On the same expert weights we measured that switch at +20.1 instead of +8.6 in drift from BF16.
* 📏 **Level with BF16 on the accuracy suites.** 83.9 overall against BF16's 83.9;
  knowledge 85.4 against 85.6, tool calling 74.8 against 74.4.
* 🧩 **Stock vLLM 0.31.0, no patches.** `compressed-tensors`, `nvfp4-pack-quantized`, weights only.

---

## Serve it

```bash
hf download primitive-ai/LFM2.5-24B-A2B-NVFP4A16 --local-dir ./LFM2.5-24B-A2B-NVFP4A16

docker run --gpus all --ipc=host -p 8000:8000 -v $PWD:/models \
  vllm/vllm-openai:v0.31.0 \
  --model /models/LFM2.5-24B-A2B-NVFP4A16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --enable-auto-tool-choice --tool-call-parser lfm2 --reasoning-parser qwen3
```

The 0.31.0 image (`sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3`) serves `Lfm2MoeForCausalLM` as released, and the experts load
on vLLM's Marlin NVFP4 kernel. Boot takes about 172 s on an idle card. The model always thinks before it answers, so give
agentic turns a generous `max_tokens`. Liquid recommends `temperature 0.1`, `top_k 50`, `repetition_penalty 1.05`; our
measurements use the fixed protocol below for every build instead, so they compare builds with each other, not with
Liquid's published numbers.

---

## Measured

One RTX PRO 6000 Blackwell, 96 GB, one card, no tensor parallelism. The 1,170-item knowledge suite and the 200-item
tool-calling suite, `temperature 0.6` / `top_p 0.95` / `top_k 20`, thinking on, a 16,384-token budget, concurrency
32, auto-scored with no LLM judge, BF16 KV cache for every row except the one marked †. Throughput is 8K in / 512 out, prefix-cache free, two
seeds per cell, every build benched on the same card one after another with the second GPU idle.

| build | size | drift from BF16 (×10⁻³ nats/token) | top-1 agreement | knowledge | tool-calling | call | abstain | finished | tok/s @1 | tok/s @32 |
|---|---|---|---|---|---|---|---|---|---|---|
| [LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), BF16 release | 47.94 GB | 0 (reference) | 100% | 85.6 | 74.4 | 82.2 | 43.0 | 98.8% | 179.3 | 861 |
| [LFM2.5-24B-A2B-FP8](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-FP8), FP8 release | 24.75 GB | +10.0 | 96.0% | 85.3 | 74.9 | 83.1 | 42.0 | 99.1% | 204.1 | 1131 |
| [infinity-ai-institute NVFP4](https://huggingface.co/infinity-ai-institute/LFM2.5-24B-A2B-NVFP4), W4A4 | 14.95 GB | +20.6 | 94.5% | 85.2 | 74.9 | 82.8 | 43.5 | 99.0% | 219.7 | 1639 |
| [cloudnathan5 NVFP4-W4A4](https://huggingface.co/cloudnathan5/LFM2.5-24B-A2B-NVFP4-W4A4), W4A4 | 13.87 GB | +66.8 | 91.2% | 84.8 | 74.9 | 82.5 | 44.5 | 99.2% | 249.2 | 1848 |
| [myroslavtryhubets NVFP4A16](https://huggingface.co/myroslavtryhubets/LFM2.5-24B-A2B-NVFP4A16), W4A16 | 13.87 GB | +32.3 | 93.6% | 85.4 | 75.9 | 83.4 | 46.0 | 99.0% | 238.7 | 1395 |
| [CoolJohn-AI NVFP4-AutoRound](https://huggingface.co/CoolJohn-AI/LFM2.5-24B-A2B-NVFP4-AutoRound), W4A4 † | 14.24 GB | +31.7 | 93.3% | 85.1 | 74.6 | 82.5 | 43.0 | 98.9% | 241.9 | 1923 |
| **this repo, NVFP4A16** | **14.95 GB** | **+8.6** | **96.1%** | 85.4 | 74.8 | 82.4 | 44.5 | 98.8% | 218.2 | 1395 |

Drift and agreement come from 400 BF16 traces of the knowledge suite (521,362 completion tokens) scored by each
build in one deterministic pass; repeating a pass gives the same number to six decimal places, and the standard error
over traces is 0.3 to 1.2. Tool-calling is the mean of 5 to 10 runs per build, with a within-build standard
deviation of 0.5 to 2.1, so that column is one band. Knowledge is two runs per build. † served with
the vLLM loader patch from its own repository, as its card instructs; its config also sets an FP8 KV cache,
which vLLM applies, so that row runs FP8 KV and has the larger pool.

What this table does not cover:

* **Languages other than English.** The model supports sixteen; both suites are English.
* **Context past 32K.** The model's window is 131,072 tokens; we did not test the 4-bit experts there.
* **Other GPUs.** Only the RTX PRO 6000 Blackwell (sm_120) was tested.

---

## Comparable with our other models

Accuracy numbers move for reasons that have nothing to do with the model: a shorter token budget, a
different `temperature`, or whether the model was allowed to reason at all. So every number in this
table, on this card and on our other cards, comes from the one fixed protocol described above, the
same 1,370 items, auto-scored, no LLM judge.

| model | shape | size | overall | knowledge | call | abstain | finished | out/answer |
|---|---|---|---|---|---|---|---|---|
| [Laguna-XS-2.1](https://huggingface.co/primitive-ai/Laguna-XS-2.1-mixed-NVFP4-MXFP8) | 31 B MoE | 19.3 GiB | 81.7 | 83.8 | 68.4 | 73.5 | 98.9% | 1097 |
| [Nemotron-3.5-Lightning-30B-A3B](https://huggingface.co/primitive-ai/Nemotron-3.5-Lightning-30B-A3B-mixed-INT4-INT8) | 30 B MoE+Mamba | 19.2 GiB | 87.1 | 87.9 | 85.4 | 70.5 | 97.9% | 1429 |
| [Ornith-1.5-35B-A3B](https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8) | 35 B MoE | 22.6 GiB | 88.7 | 91.7 | 74.4 | 60.0 | 99.3% | 760 |
| [Muse-Glimmer-30B](https://huggingface.co/primitive-ai/Muse-Glimmer-30B-mixed-NVFP4) | 30 B MoE | 20.4 GiB | 86.6 | 88.8 | 78.6 | 54.5 | 99.7% | 800 |
| [Qwen3.8-27B](https://huggingface.co/primitive-ai/Qwen3.8-27B-mixed-NVFP4-FP8) | 27 B dense | 20.7 GiB | 88.8 | 90.4 | 85.5 | 54.5 | 99.7% | 651 |
| [Granite-4.2-30B](https://huggingface.co/primitive-ai/granite-4.2-30b-mixed-NVFP4-FP8) | 30 B dense | 18.1 GB | 85.5 | 86.2 | 85.8 | 60.8 | 98.5% | 1502 |
| [Nex-N2.5-mini NVFP4](https://huggingface.co/primitive-ai/Nex-N2.5-mini-NVFP4) | 35 B MoE, 3 B active | 23.91 GB | 88.5 | 90.5 | 81.9 | 57.5 | 99.3% | 524 |
| [Nex-N2.5-mini mixed](https://huggingface.co/primitive-ai/Nex-N2.5-mini-mixed-NVFP4-FP8) | 35 B MoE, 3 B active | 26.04 GB | 88.5 | 90.6 | 80.9 | 58.7 | 99.2% | 504 |
| [Nex-N2.5-mini FP8](https://huggingface.co/primitive-ai/Nex-N2.5-mini-FP8) | 35 B MoE, 3 B active | 38.13 GB | 88.7 | 90.9 | 79.4 | 60.0 | 99.5% | 545 |
| [K2-Horizon-MoVA-36B-A4B NVFP4](https://huggingface.co/primitive-ai/K2-Horizon-MoVA-36B-A4B-NVFP4) | 37 B MoE+MoVA, 4 B active | 36.7 GB | 84.1 | 86.5 | 71.8 | 60.4 | 95.6% | 1234 |
| [K2-Horizon-MoVA-36B-A4B mixed](https://huggingface.co/primitive-ai/K2-Horizon-MoVA-36B-A4B-mixed-NVFP4-FP8) | 37 B MoE+MoVA, 4 B active | 44.5 GB | 84.9 | 87.3 | 73.5 | 58.9 | 96.3% | 1118 |
| [Laguna-S-2.1](https://huggingface.co/primitive-ai/Laguna-S-2.1-mixed-NVFP4-MXFP8) | 110 B MoE | 64.0 GiB | 84.3 | 87.1 | 64.6 | 81.0 | 97.3% | 995 |
| [Qwen3.8-Flash-Next](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8) | 180 B MoE, 6 B active | 183.7 GB | 90.3 | 92.2 | 84.8 | 56.7 | 99.5% | 686 |
| [MiMo-V2.6-Flash-RL NVFP4](https://huggingface.co/primitive-ai/MiMo-V2.6-Flash-RL-NVFP4) | 309 B MoE, 15 B active | 182.40 GB | 89.3 | 91.1 | 83.0 | 61.0 | 99.0% | 664 |
| [Kolibri-1 FP8](https://huggingface.co/Aleph-Alpha/Kolibri-1) | 78 B MoE, 3.5 B active | 78.83 GB | 85.0 | 86.1 | 86.1 | 48.0 | 97.0% | 1824 |
| [Kolibri-1 NVFP4](https://huggingface.co/primitive-ai/Kolibri-1-NVFP4) | 78 B MoE, 3.5 B active | 47.70 GB | 84.9 | 86.2 | 84.9 | 48.0 | 96.4% | 1947 |
| [Kolibri-1 mixed](https://huggingface.co/primitive-ai/Kolibri-1-mixed-NVFP4-FP8) | 78 B MoE, 3.5 B active | 45.81 GB | 83.9 | 85.0 | 85.5 | 47.5 | 96.6% | 1848 |
| **LFM2.5-24B-A2B NVFP4A16** (this repo) | 24 B MoE, 2.45 B active | **14.95 GB** | **83.9** | 85.4 | 82.4 | 44.5 | 98.8% | 1603 |

`overall` pools the two suites as 1,370 items, weighted 85.4% knowledge and 14.6% tool calling by
item count. **Read it with `finished`**: `overall` scores an answer that overran the token budget as
wrong, and cannot say whether the model needed the room or failed to stop. A gap under **1.0** in
`overall` is a tie. Sizes are as each card reports them, which mixes GB and GiB.

---

## What's quantized to what

LFM2.5-24B-A2B is 23.97 B parameters, 2.45 B active per token, and 95.7% of them are routed experts.

| | params | share |
|---|---|---|
| routed experts, 38 layers × 64 × (w1+w2+w3) | 22.95 B | 95.7% |
| short-convolution blocks, 30 layers | 0.50 B | 2.1% |
| embeddings, tied with the LM head, vocab 128000 | 0.26 B | 1.1% |
| dense feed-forward, layers 0 and 1 | 0.14 B | 0.6% |
| attention, 10 layers | 0.10 B | 0.4% |
| routers, norms, expert bias | 0.005 B | 0.02% |

| tensors | count | format |
|---|---|---|
| routed experts on all 38 MoE layers | 7,296 modules | NVFP4 weights (E2M1, E4M3 scale per 16, FP32 global), 16-bit activations |
| everything else | 314 tensors | BF16, byte-identical to the release |

`compressed-tensors`, format `nvfp4-pack-quantized`, one config group with `input_activations: null`. w1 and w3 of
each expert share one global scale, so the fused projection vLLM builds keeps the exact values. The expert weights are
calibrated, not rounded to nearest.

---

## Licence

[LFM Open License v1.0](LICENSE), the base model's licence; a copy is in this repository. This is a modified version
of LiquidAI/LFM2.5-24B-A2B: the routed-expert weights were quantized to NVFP4 and `config.json` carries a
`quantization_config`. All other weights are unchanged, and the model and its copyright belong to Liquid AI, Inc.
Commercial use is licensed only for legal entities below US$10 million in annual revenue (section 5 of the licence).

---

<p align="center">
  <br>
  <img src="https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8/resolve/main/assets/primitive-logo.png" alt="Primitive" width="34"><br>
  <sub>
    <a href="https://primitive.com"><b>primitive</b></a> ·
    <a href="https://huggingface.co/primitive-ai">more models</a> ·
    inference economics for production LLM systems
  </sub>
</p>
