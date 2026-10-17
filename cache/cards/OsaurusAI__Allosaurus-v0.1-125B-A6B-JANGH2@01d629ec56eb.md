---
language:
- en
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
- jang
- jangh
- quantized
- apple-silicon
- vision
- video
- reasoning
- thinking
- agent
- tool-use
- qwen4_exp
- moe
- mtp
- gptq
- imatrix
---

<p align="center">
  <a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a>
</p>

> **Runtime:** Osaurus **0.25.18 or newer** (`osaurus.json`: `required_osaurus_version` 0.25.18, `model_version` 1).

# OsaurusAI/Allosaurus-v0.1-125B-A6B-JANGH2

**Qwen3.8-Flash-Next for 48 GB Macs.** About 35 GiB of the model lives in memory; the 54 GiB n-gram embedding
table is read from the SSD on demand. Against our previous smallest release (JANG_1L, 41 GiB in memory) it is
**6 GiB smaller, with 3x lower median KL to the bf16 model and +9 points top-1**, and it keeps tool calls and
reasoning closes on track where the previous release slipped.

A JANGH bundle of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next): a 125B-parameter MoE with
512 routed experts (top-10 + a shared expert), gated-delta-net and sparse attention layers, one MTP layer, and image +
video input, plus a 51B-entry n-gram embedding table. About 6.7B parameters are active per token (not counting
embedding lookups).
- **Routed experts**: JANGH 2-bit. Codebook-quantized with per-row scales and a blockwise Hadamard rotation,
  calibrated with GPTQ on each expert's own input covariance.
- **Everything else in memory**: affine 4-bit (linear-attention projections, token embeddings) or 8-bit (sparse
  attention, shared experts, `lm_head`); routers in bf16; all calibrated (importance matrix + AWQ).
- **n-gram embedding table**: affine 8-bit, kept on the SSD and read per token (16 rows × 180 bytes).
- **Vision tower**: 6-bit, calibrated on 985 images and video clips (earlier Flash-Next releases quantized it
  uncalibrated).
- **MTP layer**: 4-bit, calibrated from its own capture (earlier releases quantized it uncalibrated).

## JANGH
JANGH replaces our earlier JANGTQ (TurboQuant) expert format: a fixed blockwise Hadamard-32 transform (no random signs)
and a two-constant codebook per bit width, stored bit-for-bit in MLX's own packing layout. On disk the identifiers keep
their original names: `config.json` → `jangtq` block (`version: 2`), per-module `"mode": "jangtq2"`, tensors
`*.tq2_packed` / `*.tq2_scales`. These are the JANGH format and are not loadable by JANGTQ v1 loaders.

## Quality vs the bf16 model
514,498 teacher-forced positions on four held-out sets never used for calibration, scored against the full bf16 model
run layer by layer from disk. Top-128 renormalized KL, scored on the bundle exactly as stored.

**Read this first:** some next tokens are nearly tied, so any change to the weights flips a few of them. We measured that
on the **unquantized** model (relative noise of 0.001 on a single tensor). That run is the **ceiling** below; the
distance to the ceiling, not to 100%, is what a quantization costs.

| held-out set | positions | | median KL ↓ | mean KL ↓ | p90 / p95 / p99 ↓ | top-1 ↑ | top-5 ↑ | top-10 ↑ |
|---|---|---|---|---|---|---|---|---|
| general: code, security, science, systems, chat (34 seqs) | 78,417 | ceiling | 0.0023 | 0.054 | 0.092 / 0.228 / 0.944 | 94.54% | 99.81% | 99.90% |
| | | **Allosaurus JANGH2** | **0.0471** | **0.425** | **1.228 / 2.294 / 5.043** | **81.32%** | **97.90%** | **98.88%** |
| | | JANG_1L (previous) | 0.1430 | 0.838 | 2.467 / 4.210 / 9.117 | 72.09% | 95.64% | 97.35% |
| reasoning traces at low / medium / xhigh (120 seqs) | 276,237 | ceiling | 0.0003 | 0.008 | 0.007 / 0.017 / 0.163 | 98.19% | 99.98% | 99.99% |
| | | **Allosaurus JANGH2** | **0.0086** | **0.108** | **0.175 / 0.376 / 2.119** | **91.58%** | **99.58%** | **99.83%** |
| | | JANG_1L (previous) | 0.0243 | 0.209 | 0.323 / 0.786 / 4.100 | 88.16% | 99.09% | 99.49% |
| tool conversations (96 seqs) | 66,561 | ceiling | 0.0003 | 0.125 | 0.285 / 0.673 / 2.183 | 92.45% | 99.04% | 99.52% |
| | | **Allosaurus JANGH2** | **0.0024** | **0.677** | **2.290 / 4.088 / 8.496** | **83.31%** | **94.26%** | **96.17%** |
| | | JANG_1L (previous) | 0.0050 | 1.090 | 3.822 / 6.453 / 13.245 | 80.25% | 91.60% | 93.84% |
| office, finance, math reasoning (400 seqs) | 93,283 | ceiling | 0.0002 | 0.007 | 0.011 / 0.023 / 0.094 | 97.95% | 99.99% | 100.00% |
| | | **Allosaurus JANGH2** | **0.0053** | **0.107** | **0.266 / 0.527 / 1.659** | **91.63%** | **99.23%** | **99.95%** |
| | | JANG_1L (previous) | 0.0114 | 0.314 | 0.608 / 1.663 / 5.683 | 86.63% | 97.68% | 98.94% |

General set by domain, median KL · top-1 agreement:

| domain | positions | ceiling | **Allosaurus JANGH2** | JANG_1L (previous) |
|---|---|---|---|---|
| cybersecurity | 10,199 | 0.0019 · 96.3% | **0.0246 · 88.1%** | 0.0619 · 82.1% |
| coding | 10,133 | 0.0021 · 95.7% | **0.0274 · 86.7%** | 0.1338 · 73.4% |
| systems | 3,148 | 0.0035 · 93.7% | **0.0331 · 84.3%** | 0.1033 · 76.8% |
| general chat | 5,597 | 0.0034 · 94.4% | **0.0341 · 85.7%** | 0.0795 · 81.7% |
| science | 3,919 | 0.0016 · 95.6% | **0.0441 · 82.1%** | 0.1585 · 73.1% |
| agentic | 7,586 | 0.0056 · 91.1% | **0.0656 · 78.9%** | 0.1169 · 75.9% |
| academic multiple choice | 3,158 | 0.0013 · 96.7% | **0.0154 · 87.0%** | 0.0621 · 77.2% |
| Chinese | 4,857 | 0.0006 · 97.2% | **0.0134 · 90.5%** | 0.0440 · 85.9% |
| cybersecurity, long prompts | 4,445 | 0.0041 · 94.6% | **0.0371 · 86.6%** | 0.1835 · 71.6% |
| agentic, long prompts | 5,145 | 0.0037 · 91.8% | **0.0488 · 79.8%** | 0.3557 · 62.1% |
| coding, long prompts | 9,590 | 0.0014 · 96.4% | **0.1044 · 73.8%** | 0.3583 · 61.0% |
| systems, long prompts | 5,845 | 0.0005 · 97.0% | **0.2181 · 69.5%** | 1.1963 · 48.1% |
| long documents | 4,795 | 0.0209 · 85.9% | 0.3574 · 64.8% | **0.2191 · 72.1%** |

Long documents are the one domain where the previous release stays closer to bf16; on the live long-context recall
tests below this bundle still scores higher.

### Decisions
| decisions (same choice as bf16) | **Allosaurus JANGH2** | JANG_1L (previous) |
|---|---|---|
| `</think>` closes, reasoning traces | **120 / 120** | 120 / 120 |
| `</think>` closes, office / finance traces | **297 / 300** | 286 / 300 |
| "call a tool or answer?" points: flipped vs bf16 | **11 / 108** | 34 / 108 |

## Live behavior (served, greedy)

| check | **Allosaurus JANGH2** | JANG_1L (previous) |
|---|---|---|
| long-context recall at 16k / 32k / 64k (needles, multi-key, multi-value, variable tracking, counting) | **19/20 · 18/20 · 17/20** | 17/20 · 18/20 · 15/20 |
| networking tasks with exact answers (subnets, routes, firewall rules, DNS, packet traces, IPv6, ports) | **35 / 40** | 34 / 40 |
| office / finance / math tasks with exact answers | **48 / 60** | 45 / 60 |
| invoice image: read every line item and compute the total | ✅ | ✅ |
| video: counter, background color change, moving dot | ✅ | ✅ |
| insurance premium question at low / medium / xhigh and with thinking off: reasoning closes, answer correct | ✅ all four | ✅ |

Recall, multi-key and variable tracking are perfect to 64k; counting markers degrades with length (the previous release
too). Both bundles run out of an 8k token budget on the same seven long hand-arithmetic office items at low effort.

### Reasoning efforts, same prompts (84 per effort; reference = our 6-bit release of the same model)
| xhigh | closes on its own | median tokens | p90 tokens |
|---|---|---|---|
| 6-bit reference | 97.6 % | 825 | 15,548 |
| **Allosaurus JANGH2** | **90.5 %** | **1,352** | 22,501 |
| JANG_1L (previous) | 82.1 % | 2,096 | 24,576 |

Low, medium and thinking-off match the 6-bit reference. At xhigh this 2-bit model reasons longer than the reference
and more often reaches the 24k-token budget on long coding and verification tasks; it does not loop (no repetition in
any unfinished trace), and on long reasoning that the reference closes, the probability it closes at the same point is
0.99983 (bf16: 0.99994; previous release lowest case 0.55). Give xhigh a generous `max_tokens`, or use `medium`.

## MTP (speculative decoding)
| | **Allosaurus JANGH2** (calibrated 4-bit) | JANG_2L (previous, uncalibrated) | bf16 |
|---|---|---|---|
| MTP top-1 agreement with bf16 MTP | **93.6 %** | 92.8 % | — |
| greedy acceptance (MTP guess = next token), 514k positions | **70.5 %** | 69.9 % | 70.7 % |

The MTP proposal head ships as a calibrated 4-bit refit of this bundle's `lm_head` (in `mtp_draft/`, with its
measured head description at the bundle root).

## Speed and memory (Osaurus engine, M5 Max, pre-release build, median of 3)

| | **Allosaurus JANGH2** | JANG_1L (previous) |
|---|---|---|
| model weights in memory | **34.8 GiB** | 40.8 GiB |
| load time | **1.3 s** | 19.2 s |
| prefill, ~4k-token prompt | 902 tok/s | 914 tok/s |
| decode, short prompt | 46.5 tok/s | 53.1 tok/s |
| decode right after a ~4k-token prompt | 31.7 tok/s | 52.7 tok/s |

Expert weights are memory-mapped from disk, so the process footprint stays at 7–9 GB while the OS pages the experts in.
Decode after long prompts is the known gap in the pre-release engine (it recovers to 43.9 tok/s when the experts are
held resident) and is being worked on; quality is unaffected.

## What's in the bundle
- **Vision + video**: Qwen3-VL tower, image and video processor configs; video uses timestamped temporal groups.
- **MTP**: 1 layer, calibrated, plus the calibrated proposal head.
- **Thinking + agentic**:
  - Thinking is ON by default (the template opens `<think>`); `preserve_thinking` keeps reasoning in history.
  - Reasoning efforts `low` / `medium` / `xhigh` (default `xhigh`), via the `reasoning_effort` chat-template kwarg.
  - Thinking off (`enable_thinking: false`) renders `<think>\n\n</think>\n\n`.
- **Tool calls**: XML function calls (`<tool_call>\n<function=name>\n<parameter=arg>…</parameter>\n</function>\n</tool_call>`),
  declared as `tool_parser: qwen`; reasoning parser `qwen3`.
- **Self-describing**:
  - `config.json` carries the JANGH format block, a per-module `quantization` map, and `jang_config` with the stamps
    (chat, reasoning efforts, capabilities, MTP, context) and per-layer expert bits.
  - Raw evaluation results are in `evaluation/`.
- Every shard is alignment-safe (zero-copy memory mapping).

## Serving contract
- Sampling: `temperature=1.0, top_p=0.95, top_k=20` (vendor defaults), no repetition penalty
- EOS: `[248046, 248044]` · context: 262,144
- Reasoning: `reasoning_effort` = `low` / `medium` / `xhigh` (default `xhigh`)

## Build details
- Source: `Qwen/Qwen3.8-Flash-Next` (bf16)
- Calibration: 607k tokens (code, security, agentic, general, science, systems, reasoning traces, tool conversations)
  with held-out sets kept disjoint; plus a dedicated MTP capture and a vision/video capture (985 images and clips:
  documents, screenshots, charts, code and terminal screenshots, invoices, real and screen-recorded video)
- Experts: JANGH 2-bit (odd-cubic codebook, fp16 per-row scale, Hadamard-32 rotation), GPTQ on all 48 MoE layers with
  each expert's centered input covariance; plain GPTQ scales (a unit-gain variant was A/B'd and lost on every held-out set)
- Resident weights: linear attention 4-bit, embeddings 4-bit, sparse attention 8-bit, shared experts 8-bit, `lm_head`
  8-bit, routers bf16 — chosen by measured KL cost per GiB saved; importance-matrix refit + AWQ throughout
- n-gram embedding table: affine 8-bit (group 32), read from SSD

Quantized and validated by **Jinho Jang** — eric@osaurus.ai
