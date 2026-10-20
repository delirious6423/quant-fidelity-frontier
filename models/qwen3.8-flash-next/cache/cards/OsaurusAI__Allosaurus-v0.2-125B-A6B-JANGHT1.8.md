---
language:
- en
library_name: mlx
license: other
license_name: qwen-community-1.0
license_link: LICENSE
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- mlx
- jang
- janght
- quantized
- apple-silicon
- reasoning
- thinking
- agent
- tool-use
- qwen4_exp
- moe
- mtp
- 48gb
---

<p align="center">
  <a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a>
</p>

> **Runtime:** Osaurus **0.25.24 or newer** (`osaurus.json`: `required_osaurus_version` 0.25.24, `model_version` 1).

# OsaurusAI/Allosaurus-v0.2-125B-A6B-JANGHT1.8

**Qwen3.8-Flash-Next for 48 GB Macs.** About 27 GiB of the model stays in memory; the 54 GiB n-gram embedding table is
read from the SSD on demand. [Allosaurus v0.1](https://huggingface.co/OsaurusAI/Allosaurus-v0.1-125B-A6B-JANGH2) needs
35 GiB in memory, more than a 48 GB Mac can lock for the GPU, so it slows down sharply there. v0.2 fits, up to 32k-token
contexts with speculative decoding on.

**v0.2 is the smaller model, not the better one.** It keeps 75 % top-1 agreement with the bf16 model where v0.1 keeps 81 %,
and it is text-only (the vision tower is dropped). **On a 64 GB or larger Mac, use v0.1.**

A JANGHT bundle of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). The source is a
125B-parameter MoE with 512 routed experts (top-10 plus a shared expert), gated-delta-net and sparse attention layers, and one MTP layer,
plus a 51B-entry n-gram embedding table. About 6.7B parameters are active per token, not counting embedding lookups.
- **Routed experts**: JANGT trellis codes at **1.62 bits/weight** on average. Gate and up use 1 bit in 20 layers and 1.5 bits
  in 28 layers; down uses 1.5 bits in 6 layers and 2 bits in 42 layers. The codes use a Hadamard-128 rotation with column signs
  and unit-gain per-row scales, and up to 8 outlier columns per projection are kept exactly. Widths were allocated per layer and
  projection from the measured error on the real MoE output.
- **Everything else in memory**: calibrated affine (importance matrix and AWQ).
  - 8-bit: mixers, sparse-attention indexer, n-gram projections, `lm_head`.
  - 4-bit: linear attention, sparse attention, shared experts, token embeddings.
  - Routers and norms stay bf16.
- **n-gram embedding table**: affine 8-bit (group 32), kept on the SSD and read per token.
- **MTP layer**: 4-bit (2-bit experts), calibrated. `lm_head` stays 8-bit so the engine can derive a 4-bit proposal head
  for speculative decoding (measured head description at the bundle root).
- **No vision**: no vision tower or processor configs; image and video input are not supported.

Whole-model bits/weight 1.797 = shard bytes × 8 / BF16 source parameters, with the SSD-resident n-gram table excluded from
both (the name rounds it to 1.8).

## Quality vs the bf16 model

### Whole model, Osaurus engine
Teacher-forced on 8 held-out general sequences (13,385 positions), scored against the full bf16 model.

| | in memory | median KL ↓ | mean KL ↓ | top-1 ↑ |
|---|---|---|---|---|
| **Allosaurus v0.2 JANGHT1.8** | **27.2 GiB** | 0.0585 | 0.951 | 75.00 % |
| Allosaurus v0.1 JANGH2 | 34.8 GiB | 0.0196 | 0.700 | 80.82 % |

Some next tokens are nearly tied, so even a 0.1 % random change to one tensor of the unquantized model flips a few of them:
on v0.1's general set that alone limits top-1 to about 94.5 %. Compare against that ceiling, not against 100 %.

### Where the error comes from: routed experts alone
Here only the routed experts are quantized; everything else stays bf16 and routing is frozen to the bf16 choice. Held-out sets
were never used for calibration. KL is top-128 renormalized.

| held-out set | positions | median KL ↓ | mean KL ↓ | p99 KL ↓ | top-1 ↑ | top-5 ↑ | tool-call tokens top-1 (bf16) |
|---|---|---|---|---|---|---|---|
| general: code, security, systems, office, chat (42 seqs) | 67,716 | 0.0093 | 0.309 | 5.12 | 86.60 % | 97.79 % | 89.0 % (91.4 %) |
| reasoning traces (16 seqs) | 35,252 | 0.0102 | 0.159 | 3.10 | 89.71 % | 99.44 % | — |
| tool conversations (32 seqs) | 44,057 | 0.0128 | 0.653 | 7.71 | 80.95 % | 93.95 % | 89.5 % (91.6 %) |
| office, finance, math (44 seqs) | 10,729 | 0.0034 | 0.132 | 2.69 | 91.02 % | 99.03 % | — |
| model-written traces (28 seqs) | 45,369 | 0.0058 | 0.197 | 3.29 | 87.97 % | 98.64 % | 78.6 % (81.3 %) |
| Osaurus-style agent sessions: tools, terminal, office (21 seqs) | 105,296 | 0.1792 | 0.813 | 6.38 | 71.52 % | 90.59 % | 90.8 % (93.3 %) |

General set by domain (median KL · top-1): long context 0.0010 · 92.6 %, office 0.0052 · 91.5 %, long systems prompts
0.0035 · 90.8 %, Chinese 0.0123 · 90.6 %, coding 0.0099 · 88.5 %, math 0.0236 · 87.9 %, systems 0.0295 · 86.3 %,
networking 0.0211 · 84.1 %, cybersecurity 0.054 · 81.1 %, agentic 0.0289 · 80.9 %, terminal 0.0348 · 78.9 %,
creative writing 0.216 · 69.1 %.

### What to expect
- On our everyday-task set (finance, reasoning, coding, terminal, tool calls) 93 % of answers are correct at the default
  temperature, with no loops (table below).
- **Long hand arithmetic is where the extra compression shows.** The model slips on digit-by-digit calculations (Collatz
  lengths, large Fibonacci numbers, digit sums of 2^100) more often than v0.1.
- **Reasoning does not always close.** Some hand-computation prompts make it re-check its arithmetic until the token budget
  runs out. v0.1 does the same on the same mortgage-style prompts, and lowering the temperature does not fix it. Give
  thinking a sensible `max_tokens`, or use `medium` effort.

## Sampling: temperature 0.45 by default
The vendor's temperature 1.0 makes this model reason longer than needed, so this bundle ships **`temperature 0.45, top_k 20,
top_p 0.95`**. The same values are in `generation_config.json` and `config.json` → `jang_config.chat.sampling_defaults`.

The temperature was measured, not picked. Two prompt sets were each run with several seeds at every temperature: 30 everyday
tasks (finance, reasoning, coding, terminal, tool calls) and 42 hard tail prompts. The sweeps ran on an earlier build of v0.2
with the same quality; raw results are in `evaluation/sampling_sweep_*.json`.

| temperature | everyday correct | everyday median / p90 thinking tokens | tail correct | tail median thinking | token loops |
|---|---|---|---|---|---|
| 1.0 (vendor) | 93.3 % | 158 / 464 | 81.0 % | 279 | 0 / 72 |
| 0.6 | 93.3 % | 159 / 379 | 81.0 % | 230 | 0 / 72 |
| **0.45** | **93.3 %** | **149 / 324** | **85.7 %** | **215** | **0 / 72** |
| 0.3 | 100 % | 129 / 453 | 81.0 % | 218 | **2 / 72** |

Going lower is not safer. At 0.3, and with greedy decoding, the model can fall into real token loops, such as repeating a
"subsequence … length 4" enumeration. A temperature you set in Osaurus overrides the bundle default.

## MTP (speculative decoding)
| | **Allosaurus v0.2** | bf16 |
|---|---|---|
| MTP proposal = next token (teacher-forced, 113,711 held-out positions) | **69.8 %** | 72.1 % |

In Osaurus on a 48 GB memory budget, Auto MTP vs Off made tool-calling turns 27–36 % faster. Code and essay replies came out
about even. Verification is exact: with MTP on, greedy output is identical to plain decoding.

## Speed and memory (Osaurus engine, M5 Max)
| | |
|---|---|
| model weights in memory | **27.2 GiB** (n-gram table on SSD) |
| peak active GPU memory incl. weights, 8k context, plain / MTP | 28.1 / 29.3 GiB |
| peak active GPU memory incl. weights, 32k context, plain / MTP | 29.3 / 30.6 GiB |
| decode, greedy, short prompt | 56.1 tok/s |
| load time, cold / warm | 10.6 s / 2–3 s |

The 48 GB numbers come from an M5 Max running with a 48 GB memory budget (the engine's memory policy plus locked-away RAM),
not from a physical 48 GB Mac. Osaurus' default memory mode loads the bundle. **Strict memory mode refuses it**: its estimate is
above Strict's budget, so switch back to the default mode if loading is refused.

## What's in the bundle
- **MTP**: 1 layer, calibrated, plus the measured proposal-head description at the bundle root.
- **Thinking + agentic**:
  - Thinking is ON by default (the template opens `<think>`); `preserve_thinking` keeps reasoning in history.
  - Reasoning efforts `low` / `medium` / `xhigh` (default `xhigh`), via the `reasoning_effort` chat-template kwarg.
  - Thinking off (`enable_thinking: false`) renders `<think>\n\n</think>\n\n`.
- **Tool calls**: XML function calls (`<tool_call>\n<function=name>\n<parameter=arg>…</parameter>\n</function>\n</tool_call>`),
  declared as `tool_parser: qwen`; reasoning parser `qwen3`.
- **Self-describing**:
  - `config.json` carries the `jangt` expert-format block (code multipliers, step bits) and a per-module `quantization` map.
  - `jang_config` carries the stamps (chat, sampling, reasoning efforts, capabilities, MTP, context) and per-layer expert widths.
  - `jang_config.format` reads `jangtq2` for loader compatibility; every routed expert is JANGT (`mode: jangt` per module).
- Raw evaluation results are in `evaluation/`.
- Every shard is alignment-safe (zero-copy memory mapping).

## Serving contract
- Sampling: `temperature=0.45, top_p=0.95, top_k=20`, no repetition penalty
- EOS: `[248046, 248044]` · context: 262,144
- Reasoning: `reasoning_effort` = `low` / `medium` / `xhigh` (default `xhigh`)

## Build details
- Source: `Qwen/Qwen3.8-Flash-Next` (bf16)
- Calibration: per-expert input statistics captured over a general corpus (code, security, agentic, office, science,
  systems, reasoning traces, tool conversations) and Osaurus-style agent sessions, with held-out sets kept disjoint; MTP
  calibrated from its own capture
- Experts: JANGT trellis (12-bit state, 1 / 1.5 / 2 bits per weight per layer and projection), Hadamard-128 rotation, unit-gain
  row scales, exact outlier columns
- Resident weights: affine 4/8-bit as above, importance-matrix refit and AWQ throughout
- n-gram embedding table: affine 8-bit (group 32), read from SSD

Quantized and validated by **Jinho Jang** — eric@osaurus.ai
