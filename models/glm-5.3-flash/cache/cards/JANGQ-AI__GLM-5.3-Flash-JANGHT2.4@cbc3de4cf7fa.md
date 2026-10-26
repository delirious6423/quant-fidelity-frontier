---
language:
- en
- zh
library_name: mlx
license: mit
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
tags:
- mlx
- jang
- janght
- jangt
- jangh
- quantized
- apple-silicon
- vision
- video
- reasoning
- thinking
- agent
- tool-use
- speculative-decoding
- dflash
- glm5_next
- moe
---

<p align="center">
  <a href="https://mlx.studio"><img src="https://huggingface.co/JANGQ-AI/GLM-5.3-Flash-JANGHT2.4/resolve/main/vmlx-app.png" alt="MLX Studio / vMLX — run JANG models on Apple Silicon" width="820"></a>
</p>
<h3 align="center">Run JANG models in <a href="https://mlx.studio">MLX Studio</a> / <a href="https://vmlx.net">vMLX</a></h3>
<p align="center"><a href="https://huggingface.co/JANGQ-AI"><img src="https://huggingface.co/JANGQ-AI/GLM-5.3-Flash-JANGHT2.4/resolve/main/jangq-logo.png" alt="JANGQ-AI" width="300"></a></p>

> ⚠️ **Runtime: JANGHT needs a vMLX (Python) build that is not released yet.** The bundle was built and validated
> on a vMLX development build that adds the JANGT trellis experts (decode and prefill kernels) and the bundled DFlash2
> drafter. vMLX 1.6.77 and older refuse to load it (they stop with an error; they never produce wrong output). Use it with the first vMLX release that lists JANGHT support.
> Need the previous JANGH2 weights, which released vMLX builds run? They are in this repo's history at commit
> [`4017296`](https://huggingface.co/JANGQ-AI/GLM-5.3-Flash-JANGHT2.4/tree/4017296adf8c15c49c4743e6c646b89a2eb55673)
> (`revision="4017296adf8c15c49c4743e6c646b89a2eb55673"`).

# JANGQ-AI/GLM-5.3-Flash-JANGHT2.4

**GLM-5.3-Flash for 128 GB Macs, now 4.3 GiB smaller and closer to the original model than our JANGH2 release,
with a speculative-decoding drafter in the box.**

A JANGHT bundle of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash): a 300B-class MoE (288
routed experts, top-8 + shared expert) with KDA linear attention, sparse attention, and vision + video towers.
- **Routed experts: JANGHT**, a per-layer mix of two formats, chosen by measurement:
  - **JANGT**: a trellis code at 2, 2.5 or 3 bits per weight (Hadamard-128 rotation, 12-bit state). The 2.5-bit
    variant alternates 6- and 4-bit transitions, so a layer can sit between 2 and 3 bits.
  - **JANGH**: our Hadamard-32 codebook format at 2, 3 or 4 bits, rounded with GPTQ.
- **Everything else**: 8-bit (MXFP8 where the source weights sit on an FP8-MX grid, affine 8-bit elsewhere) or full
  precision. **Vision tower**: bf16.
- **DFlash2 drafter** (`dflash2/`): z-lab's block-diffusion draft model for GLM-5.3-Flash, unmodified. vMLX uses it
  for lossless speculative decoding; code and other low-entropy text get much faster.
- **Name**: JANGHT + 2.4 bits per weight averaged over the whole model (91.57 GiB × 8 / 321.3B source parameters),
  drafter excluded.

This repository was `JANGQ-AI/GLM-5.3-Flash-JANGH2` until 2026-10-10. Old links redirect here.

## Quality vs the bf16 model
123 held-out sequences, **95,025 teacher-forced positions**, scored against the full bf16 model run layer by layer from
disk. Top-128 renormalized KL. Every row uses the same 8-bit non-expert weights, so the rows differ only in the
routed experts. The set is hard on purpose: 44k of the positions are multi-turn tool conversations, plus long
context and Chinese.

| | experts | median KL ↓ | mean KL ↓ | p90 / p95 / p99 ↓ | top-1 ↑ | top-5 ↑ | top-10 ↑ |
|---|---|---|---|---|---|---|---|
| control: bf16 experts | bf16 | 0.016 | 0.306 | — / — / 6.51 | 85.5% | 96.8% | — |
| **GLM-5.3-Flash-JANGHT2.4** | **JANGHT, 91.57 GiB bundle** | **0.251** | **1.391** | **4.31 / 7.21 / 13.32** | **65.7%** | **84.9%** | **88.6%** |
| GLM-5.3-Flash-JANGH2 (previous release, revision 2) | JANGH, 95.89 GiB bundle | 0.297 | 1.574 | 5.02 / 8.06 / 14.29 | 63.7% | 83.5% | 87.2% |

### Earlier comparison: official FP8 reference, 20 short prompts (from the JANGH2 release)
15,830 teacher-forced positions, top-128 renormalized KL against the official FP8 release. This is a different, much
easier set than the table above, and JANGHT2.4 was not run on it; the rows are kept for the comparison with other
public quants.

| Bundle | Size | median KL ↓ | mean KL ↓ | p90 / p95 / p99 ↓ | top-1 ↑ | top-5 ↑ | top-10 ↑ |
|---|---|---|---|---|---|---|---|
| GLM-5.3-Flash-JANGH2 (revision 2) | 95.89 GiB | 0.0301 | 0.376 | 0.98 / 1.97 / 5.24 | 83.9% | 96.8% | 98.3% |
| GLM-5.3-Flash-JANG (affine, earlier release) | 95.35 GiB | 0.0882 | 0.528 | 1.50 / 2.55 / 5.67 | 78.6% | 94.6% | 96.8% |
| orcarouter GLM-5.3-Flash-MLX `2bit-lite` ¹ | 95.4 GiB | 0.2122 | 0.83 | — | 71.4% | 90.8% | 94.3% |

¹ Same 20 prompts (15,850 positions), loading its shipped quantized weights natively.

On the 95k-position set above, JANGHT2.4 beats JANGH2 on every aggregate (median KL −15%, top-1 +2.1 points).

### Earlier comparison: agentic fidelity vs the bf16 model (from the JANGH2 release)
72 held-out tool-use conversations (25,867 positions; tools and phrasings disjoint from calibration), including 114
"call a tool or answer?" decision points, scored against the full bf16 model. JANGHT2.4 was not run on this set.

| | JANGH2 (revision 2) | affine JANG (earlier release) |
|---|---|---|
| median KL vs bf16 | 0.579 | 0.595 |
| top-1 agreement with bf16 | 57.3% | 54.9% |
| tool-call decisions: same choice as bf16 | 50 / 50 | 50 / 50 |
| median P(`<tool_call>`) at call points (bf16: 0.998) | 0.999 | 0.981 |
| lowest P(`<tool_call>`) at a call point | 0.914 | 0.784 |
| answer decisions: same next token as bf16 | 87.5% | 50.0% |
| decisions flipped call ↔ answer vs bf16 | 0 | 3 |

### Earlier fix kept in JANGHT2.4: long reasoning ends on its own (JANGH2 revision 2)
JANGH2 revision 1 could not end long reasoning: error-minimizing row scales shrank every low-bit matrix (12% at 2
bits), which suppressed `</think>` after a few thousand reasoning tokens. Revision 2 corrected the row scales to unit
gain, and JANGHT2.4 applies the same rule to every expert unit.

| where the reference model ends its reasoning | bf16 | JANGH2 revision 2 | revision 1 | affine JANG |
|---|---|---|---|---|
| P(`</think>`), held-out design task, after 8,025 reasoning tokens | 0.82 | 0.75 | 0.03 | 0.02 |
| P(`</think>`), coding task, after 3,288 reasoning tokens | 0.99 | 0.94 | 0.19 | - |

By domain, median KL and top-1 agreement:

| domain | positions | **JANGHT2.4** | JANGH2 |
|---|---|---|---|
| tool conversations | 44,214 | **0.498 · 59.2%** | 0.661 · 55.9% |
| coding | 9,504 | **0.074 · 73.5%** | 0.088 · 72.1% |
| agentic | 7,497 | **0.072 · 76.4%** | 0.090 · 74.0% |
| cybersecurity | 7,324 | **0.081 · 73.3%** | 0.098 · 72.3% |
| long context | 6,141 | **0.326** · 63.7% | 0.346 · **64.4%** |
| general | 5,637 | **0.179 · 68.8%** | 0.182 · 68.3% |
| science | 3,891 | 0.085 · 74.4% | **0.084** · 74.4% |
| academic multiple choice | 3,696 | **0.176 · 71.2%** | 0.197 · 69.7% |
| Chinese | 3,668 | **0.264 · 64.8%** | 0.288 · 63.9% |
| systems | 3,453 | **0.103 · 72.9%** | 0.110 · 71.9% |

## Tool decisions
Inside the 95k positions: 109 points where the bf16 model's transcript calls a tool, 96 where it answers.

| | **JANGHT2.4** | JANGH2 |
|---|---|---|
| tool-call points: same next token as bf16 | **98.2%** | 94.5% |
| tool-call points: decisions flipped call ↔ answer | **2** | 6 |
| tool-call points: model starts a tool call (bf16: 108 of 109) | **106** | 102 |
| lowest P(`<tool_call>`) at a tool-call point | **0.467** | 0.182 |
| answer points: same next token as bf16 | 78.1% | **83.3%** |
| answer points: false tool calls | 0 | 0 |

## Reasoning ends on its own
JANGH revision 1 could not end long reasoning; revision 2 fixed the cause (shrunken low-bit row scales).
JANGHT2.4 keeps that fix on every expert unit, JANGT and JANGH alike.

16 held-out transcripts written by the bf16 model in its native format (long design and coding reasoning, multi-step
tool conversations), 60,243 assistant positions, scored against bf16:

| | **JANGHT2.4** | JANGH2 |
|---|---|---|
| top-1 agreement with bf16, assistant text | **83.8%** | 82.6% |
| median KL vs bf16, assistant text | **0.058** | 0.069 |
| `</think>` points where the quant also picks `</think>` (of 34) | **18** | 14 |
| median P(`</think>`) at those points | **0.94** | 0.85 |
| lowest P(`</think>`) | **0.18** | 0.05 |

Free generation on a long design prompt (served, effort `low`, three seeds): the model closed its reasoning on its own
in **3 / 3** runs, after 8.5k, 9.7k and 12.4k reasoning tokens.

## Live behavior (served, temperature 0, both serving lanes)

| suite | **JANGHT2.4**, DFlash2 lane (default) | **JANGHT2.4**, plain lane | JANGH2 |
|---|---|---|---|
| 48-case tool-use eval (required / auto × efforts) | **48 / 48** | **48 / 48** | 48 / 48 |
| 16 behavior probes: math, follow-up, 8 tool calls, tool round trip, efforts low/high/max, image, video | **16 / 16** | **16 / 16** | 16 / 16 |
| multi-turn chat: tool call → result → answer → second call → result → answer | **4 / 4** | **4 / 4** | — |
| reasoning-loop matrix (efforts × sampling) | clean | clean | — |

These suites check behavior; every bundle that works passes them.

## Speed (M5 Max 128 GB)
Served, one request at a time; two interleaved runs per bundle (A, B, A, B), a fresh server for each run started after
a 3-minute cool-down, each run the median of 3 probes on a never-seen prompt. Same runtime build for both bundles.

| | **JANGHT2.4** | JANGH2 |
|---|---|---|
| decode, plain lane (tok/s) | **34.8 / 35.8** | 35.3 / 35.3 |
| prefill, ~5.1k-token prompt (tok/s) | 490 / 527 | **556 / 557** |
| time to first token, ~5.1k-token prompt | 10.4 s / 9.7 s | **9.2 s / 9.2 s** |

Decode is on par with JANGH2 in a 4.3 GiB smaller bundle. Long-prompt prefill is ~8% slower (the trellis tiles cost
more to decode than the JANGH codebook).

**With the bundled DFlash2 drafter (the default lane)** decode depends on how predictable the text is: **43.5 tok/s
on prose, 65-68 tok/s on code**, same conditions. Output is unchanged.

## Prompt cache (served, 8.25k-token conversation)

| | time to first token |
|---|---|
| first request (cold) | 16.3 s |
| same request again | 0.30 s |
| conversation continued (new turn appended) | 0.71 s |
| same long prefix, different final question | 0.93 s |
| after a server restart, from the SSD cache (plain lane) | 0.40 s |

The plain lane keeps the cache on SSD across restarts. The DFlash2 lane keeps it in memory (0.32 s / 0.67 s / 1.03 s
for the first three rows).

## What's in the bundle
- **Experts** (42 MoE layers):

  | projection | JANGT 2 | JANGT 2.5 | JANGT 3 | JANGH 2 | JANGH 3 | JANGH 4 |
  |---|---|---|---|---|---|---|
  | gate / up | 35 layers | 4 | 2 | 1 | — | — |
  | down | 1 | 19 | — | 10 | 8 | 4 |

- **Vision + video**: the bf16 vision tower (byte-identical to JANGH2) + the consolidated image/video processor config.
- **`dflash2/`**: the DFlash2 drafter (2.18 GiB), unmodified, under its own license (CC BY-NC-ND 4.0, see
  `dflash2/README.md`). vMLX picks it up automatically and adapts the draft length to how often drafts are accepted;
  output is identical to plain decoding at temperature 0 and has the same distribution when sampling. Start vMLX with
  `--no-bundled-dflash2` to serve without it.
- **No MTP**: layer 45 is omitted; its bytes went into expert precision.
- **Thinking + agentic**:
  - Thinking is ON by default (the template opens `<think>`).
  - Reasoning efforts are **`low` / `high` / `max`** (default `max`). There is no `medium`, and no thinking-off
    mode: the template renders any other value, and a missing value, as Max.
  - Budget for thinking: 8-12k tokens at `low` on a large design task, 17-23k at `max`. Use `low` or `high` for
    interactive work and a generous `max_tokens`.
  - `clear_thinking=false` preserves thinking in history.
- **Tool calls**: GLM's XML dialect (`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`),
  declared as `tool_parser: glm_xml_args`; tool results render as `<|observation|>`. Hermes-style JSON parsers will
  not work.
- **Self-describing**:
  - `config.json` carries the per-module `quantization` map: 102 JANGT expert projections (`"mode": "jangt"`,
    `"kbits"`), 24 JANGH (`"mode": "jangtq2"`, `"bits"`), 147 MXFP8, 214 affine 8-bit. The `jangt` block holds the
    trellis parameters and the per-layer outlier columns; the `jangtq` block (version 2) holds the JANGH codebooks.
  - `jang_config.json` records calibration, the allocation plan, the drafter and capabilities.
- **Memory**: 91.57 GiB of weights + 2.18 GiB drafter. Fixed-size linear-attention state + compressed-latent KV
  (~6 KB/token), so long contexts do not balloon memory.
- Every shard is alignment-safe (zero-copy memory mapping). 18 shards.

## Serving contract
- Sampling: `temperature=1.0, top_p=0.95` (vendor defaults), no repetition penalty
- EOS: `[154820, 154827, 154829]` · context: 1M native
- Reasoning: `reasoning_effort` chat-template kwarg (`low` / `high` / `max`), default `max`
- JANGH units keep their original on-disk identifiers (`jangtq` block, `"mode": "jangtq2"`, `*.tq2_packed` /
  `*.tq2_scales`). They are not loadable by, and must not be routed to, JANGTQ v1 loaders.

## Build details
- Source: `zai-org/GLM-5.3-Flash-BF16` @ `a5b45eb`
- Calibration: 552k tokens rendered with the GLM chat template (coding, tool conversations, cybersecurity, agentic,
  general, Chinese, science, academic, long context, systems), referenced to the bf16 model run layer by layer;
  every tenth calibration sequence and all evaluation sets held out
- Allocation: per projection and layer, the format and width with the lowest measured cost on held-out rows, each
  layer's error weighted by its measured effect on the output distribution
- Row scales: unit gain along the source row on every expert unit (the JANGH2 revision-2 rule)
- Drafter: [z-lab GLM-5.3-Flash DFlash2](https://inco.ai/blog/dflash2/), included unmodified

Quantized and validated by **Jinho Jang** — eric@jangq.ai
