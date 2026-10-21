---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
tags:
- qwen3
- 2-bit
- quantization
- escha
- sglang
- code
- reasoning
- conversational
metrics:
- accuracy
- code_eval
model-index:
- name: Qwen3.8-27B-Escha-W2
  results:
  - task: {type: text-generation, name: Commonsense Reasoning}
    dataset: {type: commonsense, name: Commonsense-6 (avg of 6 tasks, full sets)}
    metrics:
    - {type: accuracy, value: 79.25, name: avg acc (thinking-off)}
  - task: {type: text-generation, name: Graduate-level Science}
    dataset: {type: gpqa, name: GPQA-Diamond (n=198)}
    metrics:
    - {type: accuracy, value: 88.38, name: acc (thinking-on, 28k budget)}
  - task: {type: text-generation, name: Code Generation}
    dataset: {type: livecodebench, name: LiveCodeBench v6 (since 2025-01-01, n=182)}
    metrics:
    - {type: pass@1, value: 86.81, name: pass@1 (thinking-on, 28k budget)}
---

# Qwen3.8-27B-Escha-W2 — 2-bit quantized (`escha`)

By **[Escha Labs Inc.](https://eschalabs.com/)**

**Escha-W2** is a 2-bit quantized build of **Qwen3.8-27B**. It holds the full 27B
parameter count in **10.15 GB** of weights — small enough that the whole model, its KV
cache and a 64k context fit on a single 24 GB consumer card, with room left over
(or **128k context** on the same card with a tuned config — see [Long context](#long-context)).

On the three axes we measured it against a same-backend FP8 reference, this build is
**not measurably worse**: it is ahead on commonsense reasoning, behind by exactly one
question on GPQA-Diamond, and ahead on LiveCodeBench within that benchmark's own noise.

| | |
|---|---|
| **Base model** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| **Quantization** | 2-bit (`escha`; mixed 2/3-bit per projection, **2.469 bits/weight**), int8 embedding + head |
| **Download size** | **10.18 GB** total — **10.15 GB** of weights (10,153,088,224 bytes) plus tokenizer and configs |
| **Verified GPUs** | RTX 5090 (32 GB, sm_120), RTX 4090 (24 GB, sm_89), RTX 3090 (24 GB, sm_86). 16 GB should fit at reduced context; untested. |
| **Platform** | Linux x86-64, NVIDIA sm_80+ |
| **CUDA / Python** | 12.8 runtime / 3.12 |
| **Interface** | OpenAI-compatible HTTP server |

## Contents

| Path | What it is |
|---|---|
| `model-*.safetensors`, `*.json`, `tokenizer.json` | the quantized weights, tokenizer and config |
| `opencode.json` | a ready-made [opencode](https://opencode.ai) provider block pointing at a local server |
| `LICENSE`, `THIRD_PARTY_LICENSES/` | licensing and attribution |

This repo holds **only the model**. The runtime that serves it lives in
**[EschaLabs/escha-runtime-qwen3dense](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense)**
— an SGLang build with the decode kernels this format needs. That is the engine this
model is served and measured on throughout this card.

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -U pip wheel

# PyTorch FIRST, and PINNED. The kernels are ABI-linked to it, and a bare
# "torch>=2.9" resolves to 2.11 with nothing to correct it — you would only find
# out later, via `undefined symbol: _ZN3c10...`.
pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu128

# The runtime wheel. It brings the SGLang build and the full dependency closure
# with it — do NOT also install sglang from PyPI, it will fight this one.
pip install -U "huggingface_hub[cli]"
hf download EschaLabs/escha-runtime-qwen3dense --include "sglang/*" --local-dir runtime
pip install ./runtime/sglang/escha-*.whl

# Weights (a separate repo; flat folder, no nested subdir)
hf download EschaLabs/Qwen3.8-27B-Escha-W2 --local-dir Qwen3.8-27B-Escha-W2

MODEL=./Qwen3.8-27B-Escha-W2 bash runtime/sglang/serve.sh
```

Sanity check before serving — all three must print `True`:

```bash
python -c "import torch, escha, sglang; print(torch.cuda.is_available(), \
hasattr(torch.ops.escha, 'escham_decode_gemv'), bool(sglang.__version__))"
```

(`import sglang` is part of the check on purpose — an earlier `torch, escha`-only
version passed on a box that could not actually serve.)

Then, from another shell:

```bash
curl http://127.0.0.1:30000/v1/models
```

> If generation is fluent but wrong — confident, well-formed nonsense — you are almost
> certainly on `transformers < 5.8`, which loads this architecture with a silently
> different attention path. Upgrade before reporting a quality bug.

## Connecting a client

| | |
|---|---|
| Base URL | `http://127.0.0.1:30000/v1` |
| Model id | `escha-qwen38-27b-w2` |
| API key | any non-empty string |

`opencode.json` in this repo is a working provider block — drop it in and it points at
the local server.

> The launch script binds to localhost. If you set `HOST=0.0.0.0` to reach it from
> another machine, set `API_KEY` at the same time; the server has no auth of its own.

## Thinking mode

This is a thinking model. Toggle it through `chat_template_kwargs` — a **top-level
`enable_thinking` is ignored**:

```json
{
  "model": "escha-qwen38-27b-w2",
  "messages": [{"role": "user", "content": "..."}],
  "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}
}
```

When thinking is on, the answer may arrive split across **both** `reasoning_content`
and `content` — read both, or you will score empty strings. `THINK=0` on the launch
script defaults it off.

### `reasoning_effort` — the knob most people should touch first

**The default is `xhigh`**, and it is the single biggest lever on how long an answer
takes. It goes in `chat_template_kwargs` alongside `enable_thinking`, and it only
applies while thinking is on.

| Value | What the template does | When to use it |
|---|---|---|
| `xhigh` | **default.** Prepends: *"think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity."* | hard reasoning, math, code. Every benchmark on this card ran here. |
| `medium` | **prepends nothing** — the neutral, unsteered model | general chat, agent turns, anything where `xhigh` is over-thinking a simple request |
| `low` | Prepends: *"Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration."* | latency-sensitive or high-volume traffic |

Two things to know before you rely on it:

- **It is a prompt, not a limit.** Each level injects (or omits) one sentence of system
  instruction. It *asks* for shorter reasoning; nothing enforces it, and a hard problem
  can still produce a long chain at `low`. When you need a guarantee — a benchmark, an
  agent loop, anything with a timeout — use a **thinking budget**, which forces
  `</think>` after N tokens so an answer always gets produced:
  [runtime cookbook → Bounded thinking](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense/blob/main/sglang/INSTALL.md#bounded-thinking-thinking_budget).
- **Any other value is a hard error.** The template raises on anything outside those
  three, which surfaces as an HTTP 400 — not as a silent fallback to the default.

## Tuning

`serve.sh` documents every knob at the top of the file. The ones that matter on a 24 GB card:

| Variable | Default | Notes |
|---|---|---|
| `MEM` | `0.72` | Fraction of VRAM the weight + KV pool is carved from. **Too low fails too**, with "Not enough memory … increase --mem-fraction-static". If CUDA-graph capture OOMs, step *down* (0.70, 0.68), not up. Raise it on a card larger than 24 GB. |
| `CTXLEN` | `65536` | Per-request context cap — a **default, not a ceiling**, and it allocates nothing by itself. What binds is the shared pool the server prints at startup as `max_total_num_tokens`, which must be ≥ *concurrent streams × context*. Raising `CTXLEN` without raising `MEM` gets you a pool too small for it and, with `TRUNCATE=1`, silent truncation. See **[Long context](#long-context)**. |
| `MAMBA_RATIO` | `0.3` | `--mamba-full-memory-ratio`. Not sglang's `0.9` — this is a hybrid-SSM model and every concurrent stream holds recurrent state whose size **does not shrink with context**. `0.3` is what leaves room for KV *and* graph capture at `CTXLEN=65536`. |
| `GRAPHS` | `1` | CUDA graphs. Mandatory for performance — this architecture runs many small kernels per token, so eager decode is launch-bound. `0` only to debug a capture failure. |
| `CUDA_GRAPH_BS` | `1 2 4 8 12 16` | Batch sizes captured. **Must include your maximum batch size** or that batch silently runs eager (~15% low). Note the `12`/`16` entries do **not** take effect at the shipped `MEM`/`MAMBA_RATIO` — the recurrent pool clamps a 24 GB card to 8–9 streams and those entries are dropped. Raise `MAXREQ`/`MAXMAMBA`/`MEM` first, then extend to `"1 2 4 8 12 16 24 32"`. |
| `RADIX` | `0` | Prefix caching, **off by default here** — the radix cache without speculative decoding disables the overlap scheduler, and on this hybrid it only serves **exact, complete** repeats (measured: [Long context](#long-context)). It does **not** speed up a conversation that grows, so it is not an agentic-latency lever. If you do set `1`, also set `MAXREQ` ≥ 2 — prefix caching consumes a request slot, and `RADIX=1` with `MAXREQ=1` empties the capture list and **fails at startup** with `AssertionError: capture_bs=[0]`. |
| `THINK` | `1` | `0` serves thinking-off by default. Either way, clients can flip it per request via `chat_template_kwargs`. |
| `ATTN_BACKEND` | *(unset)* | **Set `triton` on consumer Blackwell (RTX 50-series, sm_120)** — the default resolves to flashinfer, which the fork rejects for hybrid models there. Leave unset on Ampere/Ada/Hopper. |
| `ESCHA_ROUTE` | *(auto)* | Kernel launch geometry, auto-selected per GPU. **On Ampere (sm_80/86) set `blackwell` for single-user work — 1.72× at batch 1 on an RTX 3090**, identical output. Parity at batch 2–16, so leave it auto for batched serving. |
| `SERVED_NAME` | `escha-qwen38-27b-w2` | The model id clients must send. |

Per-architecture and per-VRAM launch recipes live in the
**[runtime's "Running on your GPU" cookbook](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense/blob/main/sglang/INSTALL.md#running-on-your-gpu)**.

## Verified configurations

Three tiers, each measured on the physical card named, 2026-08-20. These are the commands
behind the numbers in [Performance across GPUs](#performance-across-gpus).

```bash
# 24 GB, single user — the shipped defaults, full 64k context
MODEL=./Qwen3.8-27B-Escha-W2 bash sglang/serve.sh              # 4090: 67 tok/s bs1 · 18.1 GB
# on an RTX 3090 (or any sm_80/86) add ESCHA_ROUTE=blackwell   # 3090: 23.6 -> 40.7 tok/s bs1

# 24 GB, throughput — more streams, shorter context, graphs captured at every batch swept.
# MAXREQ/MAXMAMBA are what actually lift the stream ceiling: the shipped MEM/MAMBA_RATIO
# clamp a 24 GB card to 8-9 streams no matter what CUDA_GRAPH_BS says.
MODEL=./Qwen3.8-27B-Escha-W2 MEM=0.86 CTXLEN=32768 MAXREQ=32 MAXMAMBA=32 \
  CUDA_GRAPH_BS="1 2 4 8 12 16 24 32" bash sglang/serve.sh     # 4090: 649 tok/s @ 16 streams

# 32 GB (RTX 5090, sm_120) — triton attention is REQUIRED on consumer Blackwell
MODEL=./Qwen3.8-27B-Escha-W2 ATTN_BACKEND=triton MEM=0.85 CTXLEN=65536 MAXREQ=32 MAXMAMBA=32 \
  CUDA_GRAPH_BS="1 2 4 8 12 16 24 32" bash sglang/serve.sh     # 5090: 87.1 tok/s bs1 · 955 @ 16
```

Context and concurrency draw on **one shared pool**, and which half binds depends on your prompt
length: with short prompts it is the recurrent state (~0.15 GB per stream, independent of context),
with long prompts it is the KV pool. Either way, shortening `CTXLEN` alone does not buy you
streams — raise `MAXREQ`/`MAXMAMBA` together with `MEM`, then check `#running-req` in the server
log against the batch you actually asked for.

## Long context

`CTXLEN=65536` is a default, not a limit. All of the following was measured on an RTX 4090
(24 GB) on 2026-08-21, through this `serve.sh` and the shipped wheel, `GRAPHS=1 INT8=on`.

**What context costs.** Only **16 of the 64 layers are full attention** (every 4th); the other 48
are gated-delta-net and hold a fixed ~0.15 GB of recurrent state per *stream* that does not grow
with context. So the KV cache is **64 KiB per token** — 16 layers × 4 KV heads × 256 head-dim ×
2 (K and V) × 2 bytes. A conventional dense 27B with full attention on every layer would spend
4× that. This is why 128k fits on a consumer card at all.

**Sizing.** The pool grows linearly with `MEM`, at **~366,600 tokens per 1.0** on a 24 GB card:

| `MEM` | `CTXLEN` | pool (`max_total_num_tokens`) | headroom | peak VRAM |
|---|---|---|---|---|
| `0.72` *(shipped default)* | 65,536 | 68,686 | +3,150 | 18.3 GB |
| `0.80` | 98,304 | 112,105 | +13,801 | 19.9 GB |
| `0.84` | 131,072 | 126,767 | **−4,305 — too small** | 20.8 GB |
| **`0.88`** | **131,072** | **141,431** | **+10,359** | 21.8 GB |
| `0.92` | 147,456 | 156,092 | +8,636 | 22.7 GB |
| `0.94` | 163,840 | 163,424 | **−416 — too small** | 23.1 GB |

```bash
# 24 GB, one long stream at 128k — measured: a 120,000-token prompt answers in 68 s,
# peak 22.1 GB of 24. This is the largest context with comfortable margin on this card.
MODEL=./Qwen3.8-27B-Escha-W2 MEM=0.88 CTXLEN=131072 MAXREQ=1 MAXMAMBA=2 \
  CHUNK=2048 INT8=on bash sglang/serve.sh
```

**147,456 at `MEM=0.92` is the practical maximum**; 160k does not fit. Note that the `0.84` and
`0.94` rows came up *healthy* with a pool smaller than their own `CTXLEN` — the server logs no error,
and with the default `TRUNCATE=1` an over-long prompt is then silently trimmed rather than
rejected. **The startup line is the only check that matters: `max_total_num_tokens` must be ≥
your concurrent streams × your context.** For agentic work consider `TRUNCATE=0` so an over-long
prompt fails loudly instead of quietly losing its head.

**Context and concurrency are the same budget.** The shipped default's pool is 68,686 tokens —
just 3,150 above its own 65,536 ceiling. It holds exactly *one* full-length 64k stream. The
"8–9 concurrent streams" figure elsewhere in these docs assumes short prompts (~8k each); you
cannot have both. The single-stream recipe above is single-stream on purpose.

**Prefix caching does not help a growing conversation.** Measured at 120k with `RADIX=1`:

| request | re-prefilled | cached | wall |
|---|---|---|---|
| fresh 120k prompt | 120,000 | 0 | 66.5 s |
| **the identical prompt again** | 1,216 | **118,784** | **1.4 s** |
| 116k shared prefix + 4k new | 120,000 | 0 | 65.7 s |
| full 60k prefix + 4k appended *(measured on a 60k working set, pool to spare)* | 64,000 | 0 | 29.4 s |

Only an **exact, complete** match is reused. A pure append — the cached sequence being a complete
prefix of the new request, with pool to spare — reuses nothing. We did not trace this to a code
path; it is what you would expect if the recurrent state is only valid at the boundary it was
captured at, so there is nothing to resume from part-way through. So `RADIX=1` pays for retries, cache
warming, and multi-sampling one fixed prompt; it does nothing for an agent loop that appends a
tool result and re-sends. Budget the full prefill on every turn: **~68 s at 120k, ~27 s at 60k.**
Keeping the working context small is the latency lever here, not caching it. (If you enable it
anyway, `RADIX=1` needs `MAXREQ` ≥ 2 — with `MAXREQ=1` the server fails at startup.)

> **Quality above 64k is unvalidated.** 262,144 is the architecture's own limit and the memory
> above is measured, but our long-context retrieval results come from a different
> (mixture-of-experts) model and do not transfer to this one. A 120k-token prompt produces
> coherent, on-topic continuation — that is evidence it functions, not that retrieval accuracy
> holds. Measure on your own workload before relying on it.

> This checkpoint is **text-only**. The `qwen3_5` config declares a vision tower, but the
> quantized weights contain none — it is in the quant `ignore` list, and the checkpoint has zero
> `visual.*` tensors. `serve.sh` sets `SGLANG_VLM_TEXT_ONLY=1` so the tower is never
> instantiated. Do not send image inputs.

## Requirements in detail

- **GPU:** NVIDIA sm_80 or newer (Ampere, Ada, Hopper, Blackwell). The wheel ships a
  fatbin covering sm_80/86/89/90/100/120.
- **Driver only** — no CUDA toolkit needed. If you have one, `TRITON_PTXAS_PATH` must
  point at the `ptxas` **binary**, not the directory containing it.
- **glibc ≥ 2.28** (manylinux_2_28). RHEL 8+, Ubuntu 20.04+, Debian 10+.
- `vm.overcommit_memory=1` avoids a large-allocation failure at load on memory-tight hosts.
  In a container `/proc/sys` is usually read-only; if you cannot set it, ample host RAM (≥32 GB
  free) is sufficient in practice — full campaigns have run at `overcommit_memory=0`.
- **`transformers >= 5.8`.** Ignore the `transformers_version` recorded in
  `config.json` — it reflects the machine that produced the export.

### Format notes

- Each quantized projection stores `escha_code` plus the vectors `escha_rin`,
  `escha_rout`, `escha_s_in`, `escha_s_out`, and an `escha_config` header. The runtime
  derives each projection's code rate from the shape of its code tensor — `bits: 2.0`
  in `quantize_config.json` is a family label, not the realized rate. The realized
  average over quantized projections is **2.469 bits/weight**.
- 400 projections are coded; 449 tensors (norms, SSM `A_log`/`dt_bias`) stay in fp16, and the
  embedding and output head are int8. Every coded projection also carries an **fp32 `bias`
  vector** — a per-output-channel correction learned during the end-to-end fine-tune.
  **The runtime does not apply it**, and every number on this card was produced without it.
  Applying it is a measured wash on our commonsense suite (79.16 → 79.15 average; boolq
  +1.19, hellaswag −0.51), so a third-party loader may apply or ignore these vectors —
  ignoring them is what reproduces the results published here.
- `layer_meta` in `config.json` is informational. The runtime does not read it.

## Benchmarks

Two questions matter for a quantized build: **how much quality did it cost**, and
**what does it take to run**. Quality first, then performance.

> All quality numbers below were produced by serving this exact repo through the
> SGLang runtime linked above — not through a PyTorch reference path. Quantized
> models can score differently across backends, so a number measured on a backend
> nobody deploys is not a number worth publishing.

### Quality vs FP8

The reference arm is the **same base model served as FP8 on the same SGLang fork** —
which removes the backend itself as a variable. As a check that FP8 is a fair stand-in
for the unquantized model, it scores **88.89** on GPQA-Diamond under our protocol
against the **89.2** Qwen publishes for this model. Every arm below ran an identical
protocol.

**Commonsense-6** — full sets, no subsampling, thinking off:

| | GB | arc_c | arc_e | boolq | hellaswag | piqa | winogrande | **avg** |
|---|---|---|---|---|---|---|---|---|
| FP8 reference | ~29 | 58.62 | 82.49 | 86.30 | **82.91** | **81.45** | **76.01** | 77.96 |
| **Escha-W2** | **10.15** | **65.27** | **85.69** | **88.20** | 80.19 | 81.18 | 74.98 | **79.25** |

**Reasoning and code** — thinking on, 28k-token budget:

| Capability | Benchmark | FP8 | Escha-W2 | Δ |
|---|---|---|---|---|
| Graduate science | GPQA-Diamond (n=198) | 88.89 | **88.38** | −0.51 |
| Code generation | LiveCodeBench v6, 2025 cut (n=182) | 85.16 | **86.81** | +1.65 |
| Commonsense | Commonsense-6 avg | 77.96 | **79.25** | +1.29 |

**How to read this.** At 2.9× smaller than FP8, this build shows no measurable quality
loss on the axes we tested. That is a stronger claim than it usually is at 2 bits, so
here is the honest accounting of what each number can and cannot support:

- **GPQA-Diamond −0.51** is **one question** out of 198 (175 vs 176 correct). Cap-hit
  rates are matched at 1/198, so no truncation confound — but the comparison is
  **unpaired** (we hold aggregates for the FP8 arm, not per-item answers) and the two
  arms ran on different cards. Read it as a tie, not as a deficit.
- **LiveCodeBench +1.65** sits inside this build's own bootstrap σ of 2.51pp. Read it
  as a tie the quant happens to win. The FP8 arm's cap-hit rate was not recorded, so
  this pair has no truncation control.
- **Commonsense-6 +1.29** is the one genuinely directional result, driven by the two
  ARC tasks (+6.65 and +3.20). One scoring choice to disclose: **arc_easy is scored on
  raw `acc` for both arms**, the other five on `acc_norm`. We made that call because
  `acc_norm` degenerates on arc_easy for this model family — length normalization, not
  the model, decides the answer — but it is a choice, it favors this build, and
  `acc_norm` on that one task would move the FP8 arm up. We hold aggregates rather than
  per-item answers for the FP8 arm, so we cannot publish the exact alternative number.

#### Evaluation protocol (reproduce these numbers)

| Benchmark | Thinking | Budget | Notes |
|---|---|---|---|
| Commonsense-6 | off | — | full sets, no `--limit`; arc_easy scored on raw `acc`, the other five on `acc_norm` |
| GPQA-Diamond | on | 28,672 tok inside a 32,768 cap | n=198, seeded choice shuffle; cap-hits 1/198 |
| LiveCodeBench v6 | on | 28,672 tok inside a 32,768 cap | since-2025-01-01 cut, n=182; cap-hits 4/182 (2.2%), mean generation 17,099 tok |

Sampling for the thinking-on benchmarks: temperature 1.0, top_p 0.95, top_k 20,
min_p 0, seed 42, `reasoning_effort: "xhigh"` (the default — see
[above](#reasoning_effort--the-knob-most-people-should-touch-first)). The thinking budget
is enforced server-side, closing the reasoning block so at least 4,096 tokens always
remain for the answer.

Scoring reads only the text after the final reasoning block, uses multi-pattern answer
extraction, and takes **all prompts as the denominator** — a truncated or unparseable
generation counts as wrong, never as a skip.

> **Cross-box caveat.** Commonsense-6 ran on an L40; GPQA-Diamond and LiveCodeBench ran
> on an RTX 4090. Within each benchmark both arms used identical prompts, budget,
> sampling and scorer, but the FP8 arm could not be run on the 4090 — it does not fit.
> Engine dtype therefore differs (fp16 vs bf16) and cannot be matched.

### Performance across GPUs

Measured end-to-end on three consumer cards with the SGLang runtime, 2026-08-20 — the 4090 by
us, the **5090 and 3090 by independent evaluators on their own hardware, from these public docs
alone**. *Decode = what one user sees streaming (`1000/TPOT`); peak throughput = total server
output at the best concurrency.*

| GPU | VRAM | arch | 1-user decode | TTFT (2k prompt) | prefill | peak server throughput |
|---|---|---|---|---|---|---|
| RTX 5090 | 32 GB | sm_120 | **87.1 tok/s** | 0.68 s | ~2,850–3,120 tok/s | **955 tok/s** @ 16 streams |
| RTX 4090 | 24 GB | sm_89 | **67.0 tok/s** | 0.73 s | ~2,600–2,820 tok/s | **649 tok/s** @ 16 streams |
| RTX 3090 | 24 GB | sm_86 | **40.7 tok/s**[^amperoute] | 1.58 s | ~1,210–1,300 tok/s | **383 tok/s** @ 16 streams |

[^amperoute]: **Ampere owners: set `ESCHA_ROUTE=blackwell`.** The kernel route auto-selects
    `lovelace` on sm_80/sm_86, and on an RTX 3090 that is the *slower* choice at batch 1 —
    **23.6 → 40.7 tok/s, a 1.72× speedup** (TPOT 42.4 → 24.6 ms), with identical output on all
    sanity checks. The gain is batch-1-specific: parity at batch 2–16, and +25% again at batch 32
    on short shapes. The 23.6 figure is what you get out of the box, so every 3090 row in the
    grid below is the un-forced route unless noted.

`INT8` resolves ON automatically at ≤ 24 GB. On a 32 GB card it is off by default and worth
turning on for single-user work: **+7.2–7.4% decode at bs=1** on the 5090, measured as a paired
A/B (83.7 → 89.9 tok/s at a 128-token prompt, 78.4 → 84.0 at 2k). Read the gain, not those
absolutes — that session ran a few percent below the main sweep throughout. `INT8=on` does *not*
free VRAM under a fixed `MEM` fraction: the ~1.2 GB saved on `lm_head` is recycled into the pool,
so the KV cache grows instead (157,017 → 176,217 tokens, +12.2%). A user watching `nvidia-smi`
for that 1.2 GB will conclude the flag did nothing.

**This is not a VRAM ranking and not a generation ranking.** The 24 GB 3090 and the 24 GB 4090
hold the same weights in the same pool, and the 3090 decodes at 61% of the 4090's rate even with
the route fix (35% without it) while reaching only ~46% of its prefill rate. A 3090 has ~93% of a
4090's memory bandwidth, so a purely bandwidth-bound model would predict far better than either
figure. **Decode on this architecture is not purely bandwidth-bound at batch 1** — the
small-M GEMV path matters, which is exactly why the route flag is worth 1.72× there. Corroborating
that it is an occupancy effect: on the 3090, per-stream decode gets *faster* when a second stream
is added (23.6 → 36.0 tok/s at 128/128, and a similar jump on every decode-heavy shape).

### Serving — NVIDIA ISL/OSL grid, three GPUs

**Single-stream decode tok/s** by input/output shape, on the standard NVIDIA grid[^grid]. A dash
means that evaluator did not run that shape — the three chose overlapping but not identical grids,
so this is a union rather than a full matrix[^gridconv].

| ISL / OSL | 5090 32 GB | 4090 24 GB | 3090 24 GB[^amperoute] |
|---|---|---|---|
| 128 / 128 | **87.1** | **67.0** | **23.6** |
| 128 / 1024 | — | 66.2 | — |
| 128 / 2048 | 85.0 | 65.7 | 23.3 |
| 128 / 4096 | — | 65.3 | — |
| 500 / 2000 | 84.0 | 65.5 | 23.3 |
| 1000 / 1000 | — | 65.5 | — |
| 1000 / 2000 | — | 65.3 | — |
| 1024 / 1024 | 83.8 | — | 23.4 |
| 1024 / 2048 | — | 65.3 | — |
| 2048 / 128 | 82.6 | 65.7 | 23.4 |
| **2048 / 2048** | **82.0** | **65.0** | **23.2** |
| 4096 / 1024 | — | 64.5 | — |
| 5000 / 500 | 77.1 | 64.5 | 23.1 |
| 20000 / 2000 | — | 60.0 | — |

Mean TTFT at the same shapes, where a long prompt actually costs you something:

| ISL / OSL | 5090 | 4090 | 3090 |
|---|---|---|---|
| 128 / 128 | 98 ms | 96 ms | 171 ms |
| 1024 / 1024 | 328 ms | — | 849 ms |
| 2048 / 2048 | 676 ms | 727 ms | 1.58 s |
| 5000 / 500 | 1.67 s | 1.88 s | 3.91 s |
| 20000 / 2000 | — | 7.75 s | — |

**Single-stream decode barely moves with prompt length, on all three.** From a 128-token prompt
to a 5,000-token one it falls 11.5% on the 5090, 3.7% on the 4090 and 2.1% on the 3090; the 4090
still holds 90% of its short-prompt rate at a 20,000-token prompt. Decode is dominated by the
weight read and the KV read is small beside it, so what a long prompt costs is TTFT, not tokens
per second. The 5090's steeper decay is the one open question here — sm_120 is forced onto
`ATTN_BACKEND=triton` while the others ran flashinfer, and a costlier attention path would
produce exactly that signature (a penalty growing with KV size, short-prompt decode untouched).
Settling it needs a `triton` vs `trtllm_mha` A/B on one 50-series card, which nobody has run.

For scale: fast reading is about 5 words/s (≈7 tok/s), so even the slowest card here decodes
roughly 3× faster than you can read, and a 5090 about 12×.

#### Under concurrency

Aggregate output as streams are added:

| streams | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| **RTX 5090** (128/2048) | 85 | 149 | 283 | 568 | 771 | **955** | 695 | 841 |
| **RTX 4090** (128/1024) | 67 | — | — | 402 | — | **649** | 518 | 579 |
| **RTX 3090** (128/2048) | 23 | 73 | 137 | 241 | — | **382** | — | 309 |

The 4090 row is a 1,024-token output and the other two are 2,048 — both decode-heavy, but they
are not the same cell, so read down a column, not across a row.

**Throughput peaks at 16 streams on all three cards.** An earlier build had a much sharper cliff
past 16, because the fused decode kernel handled at most 16 rows per pass and anything above fell
through to a path meant for prefill; that boundary now sits at 32 rows
(`torch.ops.escha.escham_decode_gemv_max_m()`), worth about +20% at 24 streams on the 4090.

What is left is the shape in the table, and **we still have not established why 16 beats 24 and
32**. What we can now say is what it is *not*: the peak-at-16 shape reproduces on **sm_86, sm_89
and sm_120** — three architectures, two attention backends (flashinfer and triton), 24 and 32 GB,
different pool sizes — and every run used captured CUDA graphs, so it is not the "uncaptured batch
reads 15% low" artifact either. That rules out a card-specific or backend-specific cause and
points at the scheduler or launch geometry. A memory-traffic model does not predict it. We would
rather say that than publish a mechanism we have not verified.

Practically: **16 streams is the throughput sweet spot, and 32 is the ceiling** of the fused path.

Three settings genuinely matter, all covered in the
[runtime cookbook](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense/blob/main/sglang/INSTALL.md):

- **`CUDA_GRAPH_BS` must list your maximum batch size — and the shipped defaults do not reach 16.**
  At `MEM=0.72 MAMBA_RATIO=0.3` the recurrent-state pool admits only **8–9 concurrent streams** on
  a 24 GB card, so `max_running_requests` clamps there and the `12`/`16` entries are silently
  dropped from the capture list. To actually serve more, raise `MAXREQ`/`MAXMAMBA` **and** `MEM`
  together — see [Verified configuration](#verified-configurations). Capture is supported up to 32;
  above that the runtime refuses rather than bake in the wrong kernel.
- **Recurrent state bounds concurrency at short context; the KV pool bounds it at long context.**
  Every stream holds ~0.15 GB of `ssm_state` regardless of context length, which is what limits you
  with short prompts. With long ones the KV pool binds first, at **64 KiB per token** (only 16 of
  the 64 layers hold KV — see [Long context](#long-context)): `max_total_num_tokens` (printed at
  startup) divided by your typical ISL+OSL is your real ceiling. **If the server logs a
  `#running-req` below the batch you asked for, the surplus is queueing and your TTFT includes
  queue time** — a 3090 at 2048/2048 asked for 32 streams, ran 18, and reported a 140 s mean TTFT
  that looks like a prefill catastrophe and is not.
- **On Ampere, set `ESCHA_ROUTE=blackwell`** if you serve one user at a time — see the footnote
  above; it is worth 1.72×.

[^grid]: Grid conditions: CUDA graphs on (`GRAPHS=1`, the `serve.sh` default), prefix caching
    **off** (`RADIX=0`, also the default here), output length pinned so every cell decodes its full
    OSL, and `CUDA_GRAPH_BS` extended to cover every batch swept. Reproducing these numbers without
    `RADIX=0` and without the extended capture list will read low.

[^gridconv]: Three evaluators on three boxes with their own harnesses, so a word on what is
    comparable. **Single-stream decode is the one metric all three report identically** — per-stream
    rate, `1000 / TPOT` — and it is quoted as measured, never derived. Their *aggregate* conventions
    do not match: some report a decode-window rate and some an end-to-end rate that amortizes
    prefill, which differ by ~4% at batch 1 on the same run, so the concurrency table is each card's
    own curve rather than a cross-card ranking. Two caveats those peaks inherit: **short-output
    cells at batch understate steady decode** (with a 128-token output the ramp in and out of the
    batch is most of the wall time — the 3090's 2048/128 peak of 68 tok/s is nearly all ramp), and
    **at long inputs the KV pool cannot hold every requested stream**, so some queue and the figure
    is pessimistic rather than a tuning miss. Prefill is quoted as ISL ÷ TTFT at batch 1 for all
    three cards; a batched prefill number is a different and much larger quantity, so do not mix them.

## Licenses and attribution

This repository contains **model weights only**, released under Apache-2.0.

- **Weights** are a quantized derivative of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
  Apache-2.0. See `THIRD_PARTY_LICENSES/Qwen-LICENSE.txt`.
- **Tokenizer and chat template** are carried over from the base model unmodified.

The runtime is distributed separately, in
[EschaLabs/escha-runtime-qwen3dense](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense),
with its own `LICENSE` and `THIRD_PARTY_LICENSES/` covering SGLang and the other
components it builds on — all permissive, no copyleft.
