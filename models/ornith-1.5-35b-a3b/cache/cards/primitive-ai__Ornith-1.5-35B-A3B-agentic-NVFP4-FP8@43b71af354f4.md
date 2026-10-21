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
- tool-calling
- agentic
- blackwell
thumbnail: https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-agentic-NVFP4-FP8/resolve/main/assets/banner.png
---

<p align="center">
  <img src="assets/banner.png" alt="Primitive — Ornith-1.5-35B-A3B agentic NVFP4/FP8" width="100%">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/size-22.6%20GiB-F98424?style=flat-square" alt="size 22.6 GiB">
  <img src="https://img.shields.io/badge/tool%20calling-74.0%20vs%20BF16's%2075.1-3DD13D?style=flat-square" alt="tool calling 74.0 versus BF16 75.1">
  <img src="https://img.shields.io/badge/run--to--run%20sd-1.1%20vs%203.1%20(2.8×%20steadier)-3DD13D?style=flat-square" alt="run to run sd 1.1 versus 3.1">
  <img src="https://img.shields.io/badge/knowledge-91.5%20vs%2091.7-lightgrey?style=flat-square" alt="knowledge 91.5 versus 91.7">
  <a href="https://primitive.com"><img src="https://img.shields.io/badge/web-primitive.com-lightgrey?style=flat-square" alt="primitive.com"></a>
</p>

<h3 align="center">The steady tool-calling build: BF16-level agentic accuracy that reproduces run to run.</h3>

<p align="center">
A second quantization of <a href="https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B">ornith-ai/Ornith-1.5-35B-A3B</a>,
identical in size and format to<br>our <a href="https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8">general-purpose build</a>
but calibrated for tool use.<br><b>Pick this one for agents. Pick the other one for question answering.</b>
</p>

---

## Which of the two should you use

| | call | abstain | knowledge | pooled | tool-calling sd |
|---|---|---|---|---|---|
| **this repo** | **77.5** (n=6) | 58.3 (n=6) | 91.5 (n=3) | **89.0** | **1.3** |
| [our general build](https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8) | 74.4 (n=4) | 60.0 (n=4) | **91.7** | 88.7 | 3.1 |
| BF16 reference | 78.6 (n=4) | 61.3 (n=4) | 92.0 | 89.5 | 1.4 |

* ⚖️ **Level with BF16 on tool calling** (74.0 against 75.1), and **not provably better than our
  general build** either (+2.2 points against sds of 1.3 and 3.1). We are not claiming that gap.
* 📉 **What it demonstrably is, is steadier.** Tool-calling sd **1.3 vs 3.1**. Six runs of the general
  build span 67.5–75.5; eight of this one span 72.5–76.0.
* 📚 **Knowledge costs 0.2 points** (91.5 against 91.7), and on the pooled score this build is **0.3
  ahead** (89.0 against 88.7).
* 🎯 **The gain is in calling, not declining.** 77.5 on the 160 items that need a call, against the
  general build's 73.3. On the 40 abstention items the two are 58.3 and 56.7, inside a spread of about
  5 points, so there is no claim to make there.

---

## Serve it

```bash
vllm serve primitive-ai/Ornith-1.5-35B-A3B-agentic-NVFP4-FP8 --max-model-len 32768
```

Verified to load on **Blackwell** (native NVFP4/FP8) and on **Ampere** (A100, weight-only Marlin
path: the BF16 `lm_head` is what keeps that path loadable).

---

## Measured

**1,370 items across fourteen public benchmarks.** A 1,170-item knowledge suite over nine of them
and a 200-item tool-calling suite over five more (BFCL v4, xLAM/APIGen, ToolACE, Glaive v2, nvidia
When2Call). One fixed protocol for every row: `temperature 0.6` / `top_p 0.95` / `top_k 20`, thinking
forced on, a 16,384-token budget, concurrency 32, on one RTX PRO 6000 Blackwell.

| build | size | call | abstain | knowledge | pooled |
|---|---|---|---|---|---|
| BF16 reference | 67.0 G | 78.6 (n=4) | 61.3 (n=4) | 92.0 | 89.5 |
| ornith-ai FP8 | 36.7 G | 79.1 (n=4) | 58.1 (n=4) | 91.2 | 89.0 |
| **this repo** | **22.6 G** | **77.5** (n=6) | 58.3 (n=6) | 91.5 (n=3) | **89.0** |
| ornith-ai NVFP4 | 21.8 G | 76.1 (n=4) | 61.3 (n=4) | 90.9 | 88.3 |
| [our general build](https://huggingface.co/primitive-ai/Ornith-1.5-35B-A3B-mixed-NVFP4-FP8) | 22.6 G | 74.4 (n=4) | 60.0 (n=4) | 91.7 | 88.7 |

**None of the four quants above is distinguishable from another on tool calling.** The column spans
2.3 points against per-build sds of 0.6 to 1.3. Every figure is a mean with its n, which matters more
on this model than on any other we have measured: a single run of the general build can land anywhere
between 67.5 and 75.5, so differences under about 2 points are not differences.


---

## What's quantized to what

| tensors | format |
|---|---|
| all 40 layers' routed experts (`gate/up/down_proj`) | NVFP4 (group 16) |
| `self_attn` q/k/v/o, `linear_attn` in/out projections, shared-expert projections | FP8 E4M3 |
| `lm_head`, embeddings, vision tower, MTP block, router gates, norms | BF16 |

Rounding is chosen by GPTQ against a tool-call and agent-trajectory calibration set. Rows sharing a
prompt with any eval item were removed first, so the numbers above are measurements, not fits.

---

<p align="center">
  <br>
  <sub>
    <a href="https://primitive.com"><b>primitive</b></a> ·
    <a href="https://huggingface.co/primitive-ai">more models</a> ·
    inference economics for production LLM systems
  </sub>
</p>

`call` is accuracy on the 160 suite items that require a tool call; `abstain` is the 40 whose correct action is to call nothing. The two are reported separately because pooling them let abstention discipline masquerade as call accuracy. Split means cover the runs whose per-half records survive; the pooled tool-calling means and their n are unchanged.
