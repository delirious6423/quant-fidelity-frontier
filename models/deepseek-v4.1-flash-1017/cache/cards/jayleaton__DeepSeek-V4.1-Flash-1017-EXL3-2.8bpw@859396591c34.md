---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
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

# DeepSeek-V4.1-Flash-1017 EXL3 2.8 bpw (flash17)

An EXL3 quantization of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
(revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, the 2026-10-17 update) at **2.80 bits per weight**, built to
serve on **two NVIDIA DGX Sparks at TP = 2**. Every routed expert matrix gets its own width (2, 3, 4 or 5 bits). The
allocation this time weights **coding, science / math and tool calls** and keeps attention at the widths of the
widely used 2.9 bpw pack. It is the successor of
[jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw](https://huggingface.co/jayleaton/DeepSeek-V4.1-Flash-EXL3-2.8bpw)
(q28-v2, built from the September release) and has the same size.

**Status: the default pack for TensorFold on 2 x DGX Spark.** Recipe, serving profile and full results:
[jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) ([PR #42](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/42), `docs/FLASH-1017.md`).

## Requirements

- **Engine: TensorFold's DeepSeek-V4.1 engine (Zig).** Tested on commit `9131e10ad17dde51e94a996da8255227727cd224`.
  This pack cannot be loaded by stock exllamav3 or vLLM: it needs the DeepSeek-V4.1 architecture and per-expert
  (ragged) EXL3 widths inside one fused expert launch. Setup, build and serving:
  **[jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark)**.
- **Hardware: 2 x NVIDIA DGX Spark (GB10, 128 GB each), tensor parallel 2**, linked by the CX7 200 Gb/s port. Both
  nodes need a byte-identical copy of this repository (`SHA256SUMS` is included).
- **Engram tables from the 1017 release, not September's.** Like every EXL3 pack of this model, this one does not
  carry the Engram tables (layers 1 and 14, about 101 GB each). They changed in 1017, so download them from
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e1` and split them per rank with the recipe's `scripts/pack_engram.py`.
  The September tables do not match these weights (every Engram weight and scale changed in 1017).
- **Image input:** the vision tower and aligner (263 bf16 tensors) are in `native.safetensors`, unchanged from the
  release. On the Zig engine set `TF_DSV41_IMAGES=native` and **`TF_DSV41_BIAS_VL=<this folder>`**, so the 1017
  image-routing biases (43 tensors, also in `native.safetensors`) are used rather than a September copy. VQA 20 / 20
  and 3 / 3 `image_url` probes passed at pools of 3.0M, 2.4M and 2.2M.
- **Chat template:** the 1017 release ships no `chat_template.jinja`; it documents its prompt format in
  `encoding/encoding.py`. This repository includes the September release's `chat_template.jinja` only because some
  TensorFold builds look for the file at boot. The Zig engine renders prompts with its own encoder and does not read
  it. The 1017 encoder's change (a user turn holding only tool results does not start a new turn) is **not** in Zig
  `9131e10a`. It only affects continuations that carry tool results without declaring `tools`.

### Serving profile

Engine Zig `9131e10a`, TP = 2: the q28-v2 serving profile with only these changes
([details](https://github.com/jayleaton/deepseek-v41-tensorfold-spark/pull/42)):

| setting | q28-v2 profile | this pack | why |
| --- | --- | --- | --- |
| model folder | q28-v2 pack | this pack | the new weights |
| Engram tables | September shards | **1017 shards** | the tables changed in 1017 |
| prepared / state folders | q28-v2's | **fresh** | prepared files and calibration belong to one pack |
| `TF_DSV41_BIAS_VL` | `/cache/dsv41-bias-vl` | **this folder** (`/model`) | the 1017 learned image biases |
| `TF_DSV41_POOL_TOKENS` | 3000000 | **2100000 (required)** | memory headroom |

Unchanged:
- context 1,048,576, 4 slots, split + compact KV;
- expert top-p 0.85, at least 3 experts, kept weights renormalised;
- native images, 4K prefill;
- memory floors 5 / 4 GiB, the 3.0 GiB serving dip.

We suggest a 2100–2200 MHz GPU clock cap to help with temperature control.

**Why 2.1M.** With this pack, larger pools (3.0M, 2.4M) let the head node's free memory fall below the 4 GiB floor
during the first long prompts: a transient 2.27 GB 4K-prefill workspace sits on top of the KV pool. At 2.1M no run
went under the floor (lowest sample 4.226 GiB). With a 2.1M pool, one or two slots can still fill 1M of context each;
four live slots share about 525K each.

## Sizes

| | bytes |
| --- | --- |
| this repository (47 pack files, plus README, LICENSE, SHA256SUMS) | 203,793,369,312 (203.8 GB, 189.8 GiB) |
| 40 decoder layers (`layer-00` ... `layer-39.safetensors`) | 193.4 GB |
| embedding, output head and MTP / draft layers (`head-mtp.safetensors`) | 7.6 GB |
| vision tower and other native tensors (`native.safetensors`) | 2.8 GB |

Bits per weight, on the 2.9 bpw pack's definition (routed + shared experts + attention + CSA2 + Engram projection):
**2.8003**. By part: routed experts 2.77 bpw, attention 5.0, shared experts 5.38, CSA2 5.0, Engram projection 5.0,
output head 6, MTP layers 4. Routed expert matrices by width: 2 bits 14,732, 3 bits 27,328, 4 bits 3,836, 5 bits 184
(46,080).

## How it was made

Base: `deepseek-ai/DeepSeek-V4.1-Flash-1017` (FP8 dense weights, MXFP4 experts), dequantized one layer at a time.
Encoding: exllamav3 1.5.4's `quantize_exl3` (LDLQ + trellis, `mul1` codebook, output scales on), driven per layer
through a DeepSeek-V4.1 reference forward, since upstream exllamav3 does not convert V4.1.

**What changed from q28-v2, and why.** q28-v2 measured closer to its release than the 2.9 bpw pack on next-token KL,
yet on coding and science tasks it did not come out ahead. Two findings shaped this pack:
- **The objective.** q28-v2's allocator minimised a KL total in which tool-call and multilingual text dominate. Code
  and math have low KL per token, but one wrong token fails a program or an answer.
- **As served, the lead shrinks most on reasoning and code.** TensorFold prunes experts at serving time (top-p
  0.85). Measured with that pruning, q28-v2's extra KL relative to the 2.9 bpw pack was largest on reasoning and code.

So this pack:
- weights the objective **24 code / 24 math + science / 30 tool calls**, with multilingual and encyclopedic prose as
  no-regression guards;
- puts attention back at 5 bits;
- moves routed bits toward the experts that code and science route to;
- keeps the same total bytes and the same decode read-time cap as q28-v2.

Per-layer sensitivities come from measured layer swaps on the September release (the router of 1017 is bit-identical
to September's; routed experts moved ~15 %).

**Calibration**: 1,215,188 tokens, rendered in the V4.1 chat format:
- 24 % code (Magicoder OSS-Instruct, OpenThoughts code reasoning);
- 24 % math + science reasoning in thinking mode (OpenR1-Math, Mixture-of-Thoughts science traces at all four
  effort levels);
- 30 % tool calls (Hermes function calling incl. agentic JSON, Glaive function calling);
- 9 % chat, 5 % encyclopedic prose (wikitext-103 train), 4 % web text, 4 % multilingual.

## Evaluation

### Fidelity to the release, as served

Next-token KL divergence and top-1 agreement against each pack's **own** release (this pack against 1017; the
September packs against September), on the same held-out per-domain set (~24.5K tokens a domain). Scored with the
expert pruning TensorFold serves (top-p 0.85, at least 3 experts). Confident mistakes: top-1 differs where the
release is at least 90 % sure, per 10,000 positions.

| domain | this pack vs 1017 | q28-v2 vs September | 2.9 bpw (Mia-AiLab) vs September |
| --- | --- | --- | --- |
| code | 0.0328 / 95.17 % / 7 | 0.0269 / 95.35 % / 4 | 0.0330 / 94.77 % / 7 |
| science | 0.0242 / 94.93 % / 2 | 0.0243 / 94.70 % / 6 | 0.0310 / 93.88 % / 0 |
| math | 0.0197 / 96.58 % / 3 | 0.0187 / 96.33 % / 3 | 0.0264 / 95.67 % / 5 |
| tool calls | 0.2013 / 87.64 % / 109 | 0.1394 / 89.66 % / 105 | 0.1889 / 87.94 % / 187 |
| multilingual | 0.0982 / 90.68 % / 55 | 0.0902 / 91.03 % / 31 | 0.1122 / 90.22 % / 57 |
| chat | 0.0688 / 91.96 % / 27 | 0.0626 / 92.28 % / 20 | 0.0731 / 91.45 % / 26 |

How to read this:
- The rows compare how much each pack keeps of **its own** base model. The bases differ (1017 vs September), so
  this is not a paired comparison.
- Teacher-forced fidelity like this did not predict task results for q28-v2, which led here and still did not beat
  the 2.9 bpw pack on coding / science tasks.
- The task results below decide usefulness.

### Tasks, served on 2 x DGX Spark

All 356 planned items. The same frozen items and scorers for all three packs:
- greedy, thinking on, effort high, max_tokens 32,768, pass@1;
- paired, so the other packs' columns count the same items;
- intervals are conservative paired 95 % intervals (Bonferroni Clopper-Pearson on the discordant counts, per suite,
  not simultaneous across suites).

| suite (completed / planned) | this pack | 2.9 bpw (Mia-AiLab) | q28-v2 | vs 2.9 bpw, points [95 % CI] | vs q28-v2, points [95 % CI] |
| --- | ---: | ---: | ---: | --- | --- |
| HumanEval+ (40 / 40) | 34 / 40 | 34 / 40 | 33 / 40 | +0.0 [-14.9, +14.9] | +2.5 [-10.3, +14.9] |
| MBPP+ (60 / 60) | 50 / 60 | 49 / 60 | 50 / 60 | +1.7 [-9.9, +12.8] | +0.0 [-12.6, +12.6] |
| MMLU-Pro STEM (100 / 100) | 93 / 100 | 91 / 100 | 90 / 100 | +2.0 [-4.1, +7.9] | +3.0 [-5.3, +10.9] |
| MATH-500 (40 / 40) | 36 / 40 | 36 / 40 | 36 / 40 | +0.0 [-10.4, +10.4] | +0.0 [-10.4, +10.4] |
| SciCode first step (10 / 10) | 6 / 10 | 6 / 10 | 5 / 10 | +0.0 [-35.5, +35.5] | +10.0 [-35.4, +49.1] |
| LiveCodeBench (20 / 20) | 16 / 20 | 13 / 20 | 13 / 20 | +15.0 [-17.2, +41.1] | +15.0 [-17.2, +41.1] |
| BFCL (40 / 40) | 39 / 40 | 39 / 40 | 39 / 40 | +0.0 [-10.4, +10.4] | +0.0 [-10.4, +10.4] |
| tool chains (6 / 6) | 6 / 6 | 6 / 6 | 4 / 6 | +0.0 [-51.8, +51.8] | +33.3 [-48.8, +81.6] |
| MGSM (40 / 40) | 40 / 40 | 40 / 40 | 40 / 40 | +0.0 [-10.4, +10.4] | +0.0 [-10.4, +10.4] |
| **all 356 items** | **320 / 356** | 314 / 356 | 310 / 356 | +1.7 [-1.4, +4.7] | +2.8 [-0.9, +6.4] |

**Every confidence interval includes zero.** At these sample sizes the three packs are statistically
indistinguishable on these tasks, and this card makes no claim that this pack beats the 2.9 bpw pack or q28-v2.

Limits:
- The comparison packs ran on TensorFold's Python engine `8474f31`; this pack ran on Zig `9131e10a`.
- This pack's rows ran across several pool settings (3.0M down to 2.1M) with the same generation settings.
- 1017 is a stronger base model than the September release both other packs were made from, so a task difference
  would not by itself be a quantization difference.
- GPQA-Diamond is not included (gated access). SciCode is first subproblems only; tool chains are simulated.

**Images, tools, agents** (at the serving profile above unless stated):
- images: VQA 20 / 20 and 3 / 3 `image_url` probes (chart, screenshot, photo), answers reviewed;
- OpenCode agent session (multi-file build): clean pass, held-out build checks passed;
- BFCL 39 / 40, tool chains 6 / 6 (the task table above).

**Speed** at the serving profile above (pool 2.1M, 2100 MHz cap), single stream, greedy, thinking off:

| measurement | value |
| --- | ---: |
| short decode (384 tokens, mean of 3 runs) | 87.9 tok/s |
| prompt throughput, ~32K-token prompt (37,979 tokens) | 2,349 tok/s |
| prompt throughput, ~131K-token prompt (154,188 tokens) | 2,198 tok/s |
| decode after a long prompt | ~64 tok/s |

Prompt throughput is prompt tokens over time to first token, including HTTP overhead. These are single runs on one
pair of machines, without confidence intervals; measure on your own workload.

## Licence

MIT, the base model's licence. This repository is a quantized derivative of
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), Copyright (c)
2023 DeepSeek; the original copyright and permission notice are in [LICENSE](LICENSE). The tokenizer and configuration
come from the base model; `chat_template.jinja` comes from
[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) (MIT).
