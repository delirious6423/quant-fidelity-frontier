---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
tags:
  - exl3
  - quantized
  - deepseek
  - deepseek-v4.1
  - mixture-of-experts
  - dgx-spark
---

# DeepSeek-V4.1-Flash EXL3 2.8 bpw (q28-v2)

An EXL3 quantization of [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
at **2.80 bits per weight**, built to serve on **two NVIDIA DGX Sparks at TP = 2**. Every routed expert matrix gets
its own width (2, 3, 4 or 5 bits), allocated from measured layer sensitivity rather than a uniform rate.

At a smaller size it is closer to the original release than the 2.9 bpw pack it was built against
([Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw)):

| | this pack (2.8 bpw) | Mia-AiLab 2.9 bpw |
| --- | --- | --- |
| KL to the release, 167,901 positions, as served | **0.0469** | 0.0594 |
| KL, thinking spans only, as served | **0.0078** | 0.0103 |
| KL, wikitext-2 test, as served | **0.1335** | 0.1405 |
| top-1 agreement with the release, as served | **0.9406** | 0.9321 |
| MMLU-200, 0-shot, served on 2 x DGX Spark | **88.5 %** (177 / 200) | 87.5 % (175 / 200) |
| weights in the bpw budget set | **192.65 GB** | 199.53 GB |

## Requirements

- **Engine: TensorFold, with the Zig engine.** This pack cannot be loaded by stock exllamav3 or vLLM. It needs the
  DeepSeek-V4.1 architecture and per-expert (ragged) EXL3 widths inside one fused expert launch, which TensorFold's
  DeepSeek-V4.1 engine provides. Setup, build and serving:
  **[jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)**. Until
  [PR #18](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/18) is merged, the Zig serving docs and this
  pack as the default are on that PR.
- **Hardware: 2 x NVIDIA DGX Spark (GB10, 128 GB each), tensor parallel 2**, linked by the CX7 200 Gb/s port. Both
  nodes need a byte-identical copy of this repository.
- **Engram tables from the original checkpoint.** Like every EXL3 pack of this model, this one does not carry the
  Engram tables (layers 1 and 14, about 101 GB each). They are downloaded from `deepseek-ai/DeepSeek-V4.1-Flash` and
  split per rank by the recipe's `scripts/pack_engram.py` (see its `docs/INSTALL.md`).

## Sizes

| | bytes |
| --- | --- |
| this repository (47 files) | 203,831,784,855 (203.8 GB, 189.8 GiB) |
| 40 decoder layers (`layer-00` ... `layer-39.safetensors`) | 193.4 GB |
| embedding, output head and MTP / draft layer (`head-mtp.safetensors`) | 7.6 GB |
| vision tower and other native tensors (`native.safetensors`) | 2.8 GB |

Bits per weight, on Mia-AiLab's definition (routed + shared experts + attention + CSA2 + Engram projection):
**2.8003** (Mia-AiLab: 2.9000). By part: routed experts 2.78 bpw stored, shared experts 5.9, attention 4.02, output
head 6, MTP layer 4. Routed expert matrices by width: 2 bits 17,848, 3 bits 20,998, 4 bits 6,787, 5 bits 447 (46,080).

## How it was made

Base: the original `deepseek-ai/DeepSeek-V4.1-Flash` release (FP8 dense weights, MXFP4 experts), dequantized one layer
at a time. Encoding: exllamav3 1.5.4's `quantize_exl3` (LDLQ + trellis, `mul1` codebook, output scales on),
driven per layer through a DeepSeek-V4.1 reference forward, since upstream exllamav3 does not convert V4.1.

**Per-expert widths.** Experts are not equally used, and layers are not equally sensitive. Each routed expert
matrix (w1 / w2 / w3 of each of 384 experts in each of 40 layers) gets one of four widths. An allocator minimises
predicted KL under three caps: the 2.8 bpw storage budget, a decode read-time cap (bytes actually read per token,
weighted by each expert's routing frequency, no more than the 2.9 bpw pack's), and per-domain guards. An expert that
is rarely routed costs storage but little read time, so it can be narrow without slowing decode; a hot expert in a
sensitive layer is widened.

**Layer-swap sensitivity.** Sensitivity is measured, not modelled. For each of three domains (a mixed eval of chat /
code / multilingual / reasoning / text / tool calls, held-out encyclopedic prose, held-out reasoning traces), each
layer's routed experts are swapped for a reference pack's and the KL against the clean forward is recorded: all 40
layers one at a time, all layers at once (to measure how one-layer swaps add up), and at 8 layers against a second
pack at different widths (to get the slope a bit is worth). The allocator's per-domain layer weights are fitted to
those swaps. A one-layer swap carries a part that does not depend on width; the fit separates it, so it is not read
as sensitivity. (An earlier allocation fitted to Gaussian-noise injection put the early layers at the 2-bit floor and
lost on encyclopedic prose; measured swaps moved those bits back to layers 0-18.)

Objective: lowest KL on the eval mix and on thinking spans, with prose no worse than the 2.9 bpw pack. Attention,
shared experts, CSA2 and the Engram projections are at fixed widths (attention ~4 bits, shared experts ~6 bits).

**Calibration**: 1,406,759 tokens: chat, code, multilingual text, reasoning, tool calls, encyclopedic prose (12 %,
wikitext-103 train articles, none in the evaluation sets) and DeepSeek-R1 reasoning traces in thinking mode at all
four effort levels (open-r1/Mixture-of-Thoughts). 27 % of the tokens are in thinking mode.

## Evaluation

**KL to the release.** Next-token KL divergence against the original FP8 / MXFP4 release (its top 2,048 logits),
95 % confidence intervals from a paired block bootstrap (10,000 resamples). "As served" means with the expert routing
TensorFold serves (top-p 0.85 over the router's top 6, at least 3 experts).

| set (as served) | positions | this pack | Mia-AiLab 2.9 | difference [95 % CI] |
| --- | --- | --- | --- | --- |
| eval mix | 167,901 | **0.0469** | 0.0594 | -0.0125 [-0.0143, -0.0107] (-21 %) |
| thinking spans of the eval mix | 53,018 | **0.0078** | 0.0103 | -0.0025 [-0.0028, -0.0023] (-24 %) |
| wikitext-2 test | 32,752 | **0.1335** | 0.1405 | -0.0068 [-0.0114, -0.0022] (-5 %) |

With all 6 routed experts (no pruning): eval 0.0258 vs 0.0439, thinking 0.0023 vs 0.0076, wikitext-2 0.0947 vs
0.0937 (a tie). The eval mix is better in every domain (chat, code, multilingual, reasoning, text, tools).

**Served on 2 x DGX Spark** (TP = 2, same engine and settings for both packs; TensorFold Python engine):
- MMLU-200, 0-shot: 88.5 % vs 87.5 %. 200 questions, so the difference is two answers.
- Thinking on 128 reasoning questions (GSM8K + MATH-500 levels 3-5, default effort): 97.7 % on both, mean thinking
  710 vs 724 tokens (a tie: this pack does not think longer).
- Tool-call chains: 10 / 12 pass (the gate is 10).
- Speed, **measured on TensorFold's Python engine** (the 2026-10-07 test window), against the 2.9 bpw pack on the
  same build and settings, medians of interleaved boots: cold prefill 8K / 32K tokens +2.2 / +3.5 %; single-stream
  decode code +2.0 % (greedy) / +13.3 % (T 0.7), prose +11.6 % / -0.9 %. For Zig-engine numbers, see the
  [recipe repository](https://github.com/jayleaton/deepseek-v41-tensorfold-spark).
- Memory: about 3.3 GiB a node less than the 2.9 bpw pack, which takes the served context to 1M tokens.

These are one pair of machines; measure on your own workload.

## Licence

MIT, the base model's licence. This repository is a quantized derivative of
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), Copyright (c) 2023
DeepSeek; the original copyright and permission notice are in [LICENSE](LICENSE). The tokenizer, chat template and
configuration come from the base model.
