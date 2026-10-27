---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - Qwen/Qwen3.8-27B-FP8
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - fp8
  - e4m3
  - multimodal
  - conversational
  - cuda
  - rtx-4090
  - rtx-pro-6000
---

# Qwen3.8-27B FP8 for NInfer

This repository contains `Qwen3.8-27B-FP8-v3-full.ninfer`, the full NInfer v3 artifact for
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) built from the official
[Qwen/Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) block-wise FP8 checkpoint. The
artifact is intended only for NInfer; it is not a Transformers checkpoint, Safetensors
distribution, or GGUF file.

The Text tower keeps the official E4M3 quantization. The checkpoint stores the linear
projections as E4M3 codes with a BF16 `weight_scale_inv` grid per 128×128 weight block; the
converter decodes each block exactly (an E4M3 mantissa times a BF16 scale fits FP32 without
rounding) and re-encodes the values with one BF16 scale per output row (`fp8_row_maxabs`). The
token embedding, output head, and every text projection except the GDN `a`/`b` gating
projections (kept direct BF16) use that row-scaled FP8, which the engine runs on its
FP8-activation routes. MTP uses Q8 group-32, Vision is full BF16, and DFlash2 uses
Q4/Q5/Q6 group-64. The official card reports the block-wise FP8 checkpoint's performance as
nearly identical to the BF16 original.

## Artifact

| Field | Value |
|---|---|
| Filename | `Qwen3.8-27B-FP8-v3-full.ninfer` |
| Size | 30,275,067,136 bytes (28.20 GiB) |
| SHA-256 | `437d1d1aaeb16e20c252fa194471e6d869edab68d04d78198c9b2cf146d31c43` |
| Container version | 3 |
| Artifact id | `66ef2afa4acd42cfb55dfea28daadb96` |
| Architecture | `Qwen3_5ForCausalLM` |
| Public model name | `qwen3.8-27b-fp8` |
| Chat template | [qwen3_8.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_8.jinja); override with `--chat-template FILE` |
| Template defaults | thinking on; effort `xhigh`; closed-turn reasoning retained |
| Stored objects | 1,126 (1,120 tensors and 6 resources) |
| Row-scaled FP8 E4M3 tensors | 258 (embedding, output head, text projections) |
| Q8 group-32 (FP16) tensors | 28 (MTP) |
| Q4/Q5/Q6 group-64 (FP16) tensors | 55 / 54 / 1 (DFlash2) |
| BF16 tensors | 627 (Vision tower, GDN `a`/`b` gating, control) |
| FP32 / INT32 tensors | 96 / 1 |

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '437d1d1aaeb16e20c252fa194471e6d869edab68d04d78198c9b2cf146d31c43' \
  'Qwen3.8-27B-FP8-v3-full.ninfer' | sha256sum --check
```

The file contains Text, Vision, MTP, DFlash2 and the frontend resources. Vision weights come
from the checkpoint's own full-BF16 visual tower. Vision and speculative weights are loaded
only when selected at startup.

## Conversion and verification

The official checkpoint quantizes the text linear projections with E4M3 codes plus a BF16
`weight_scale_inv` grid per 128×128 weight block. NInfer's native storage is per-row: one BF16
scale per output row. How the conversion closes that gap:

**Block-wise to per-row, exactly.** The built-in compressed-tensors reader previously resolved
only per-row FP8 (`weight_scale`) and NVFP4, so block-wise checkpoints failed signature
resolution. A matrix stored as `F8_E4M3` with a sibling `weight_scale_inv` now resolves to a
block-wise value source: the grid is taken from `quantization_config.weight_block_size` (or
derived by divisibility), validated against the code shape, and each code is decoded in FP32,
where an E4M3 mantissa times a BF16 scale fits exactly. The decoded values are then re-encoded
with one BF16 scale per output row (`fp8_row_maxabs`). Row ranges decode only the block rows
they touch, and the decode was checked bit-exact against an independent E4M3-table times
BF16-grid oracle on real checkpoint tensors (MLP gate/up, full-attention `q_proj`, MTP `o_proj`,
including rows spanning block boundaries).

**Deliberate per-operator filtering.** The `qwfp8ovr.py` override intercepts only the token
embedding, the output head, and the standard text attention/MLP projections. The GDN `a`/`b`
gating projections are not suitable for row FP8 and are kept as direct BF16; Vision, MTP, and
DFlash2 fall through to the base recipe's Q4–Q8 groupwise selection.

**Error at the format's noise floor.** Dequantized artifact objects (MLP gate/up, attention
q/k/gate/v under per-head interleaved row selection, embedding rows) match the independent
source decode within the theoretical half-ulp bound of the stored row scale on every row. The
overall relative L2 is ~2.6% (2.62–2.65% across the sampled projection families): the inherent
E4M3 quantization noise, since both sides are E4M3 with different scale granularity. The
re-scaling rounding therefore reaches the mathematical limit of the format; the GDN
`a_projection` object is bit-exact BF16, and the artifact contains no NaNs.

**Hardware fit.** Every large text projection sits in the row-scaled FP8 bank, so prefill and
decode run the FP8-activation tensor-core routes directly instead of dequantizing to BF16
first, and `--kv-dtype fp8` keeps the attention matrices in the same dtype as the weights.

`python -m tools.artifact.inspect` validates configuration, resources, bindings, and uses of
all four components (text, vision, mtp, dflash2).

## Model overview

From the official Qwen3.8-27B card; capability benchmarks are published there.

- Type: causal language model with vision encoder
- Parameters: 27B; hidden size 5,120; 64 layers; padded vocabulary 248,320
- Layer layout: 16 × (3 × (Gated DeltaNet → FFN) + 1 × (Gated Attention → FFN))
- Gated DeltaNet: 48 value heads, 16 QK heads, head dimension 128
- Gated Attention: 24 query heads, 4 KV heads, head dimension 256, rotary dimension 64
- Feed-forward intermediate size: 17,408; MTP head trained
- Vision encoder: 27 layers, hidden 1,152, patch 16, temporal patch 2
- Context: 262,144 tokens natively, extensible to 1,000,000 with RoPE scaling (YaRN)

## Requirements

- [NInfer-all](https://github.com/iamwavecut/ninfer-all) revision
  [`7a89828a`](https://github.com/iamwavecut/ninfer-all/commit/7a89828a3446cd1fa7bb1465ad10386469718e05) or later,
  built from source for the target architecture (`-DCMAKE_CUDA_ARCHITECTURES=89` for the RTX 40
  series, `120a` for the RTX 50 series and the RTX PRO 6000 Blackwell), or downloaded from the
  [binary releases](https://github.com/iamwavecut/ninfer-all/releases);
- Applicable GPUs: NVIDIA GeForce **RTX 4090 (48 GB)** and NVIDIA **RTX PRO 6000 (96 GB)**, and
  other cards the engine supports. The 262,144-token, four-lane configuration below is sized
  for 48 GB of VRAM; the 96 GB card runs the same line with headroom to raise
  `--max-context` and `--kv-capacity`;
- 64-bit Linux;
- CUDA Toolkit 13.1 or newer.

The upstream [NInfer](https://github.com/Neroued/ninfer) build targets `sm_120a` (RTX 50 series)
only, and the startup line below uses NInfer-all's `--gdn-state-fp16`, the checkpoint-catalog
context cache (`--auto-long-anchors`, `--auto-prefix-grid`, `--derive-session-keys`), the fast
int8-prefill kernel, and the Vision tower with overlay residency.
NInfer does not provide an install target or packaged binary. See the
[repository build guide](https://github.com/iamwavecut/ninfer-all#building).

## Start the server

The verified startup line for the full FP8 + DFlash2 configuration (NInfer-all `ninfer-serve`):

```bash
export CUDA_MODULE_LOADING=LAZY

ninfer-serve models/Qwen3.8-27B-FP8-v3-full.ninfer \
    --model-id qwen3.8-27b-fp8 \
    --max-context 262144 \
    --kv-capacity 262144 \
    --prefill-chunk 32768 \
    --prefill-cublas \
    --max-concurrency 4 \
    --kv-dtype int8 \
    --fast-prefill-kernel \
    --concurrent-prefill \
    --wddm-evictable-budget \
    --gdn-state-fp16 \
    --host "127.0.0.1" \
    --port 10111 \
    --vision --vision-residency overlay \
    --default-reasoning-effort xhigh \
    --auto-long-anchors \
    --auto-prefix-grid \
    --derive-session-keys \
    --spec dflash2 --draft-tokens 5
```

| Option | Value | Effect |
|---|---|---|
| `--model-id` | `qwen3.8-27b-fp8` | public model name reported by the server (independent of the artifact's file name) |
| `--max-context` | 262,144 | per-request logical context ceiling — the model's native window |
| `--kv-capacity` | 262,144 | shared KV pool in tokens; a request running alone may use the whole pool |
| `--prefill-chunk` | 32,768 | prefill chunk size in tokens (a multiple of 128); `--fast-prefill-kernel` rounds it down to whole attention waves |
| `--prefill-cublas` | - | routes wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost; benefits from a larger --prefill-chunk |
| `--max-concurrency` | 4 | four active request lanes |
| `--kv-dtype` | `int8` | int8 KV cache, the better choice when used with `--fast-prefill-kernel`, downgrade to `rk8v4` if VRAM is tight. Note: with `fp8`, --fast-prefill-kernel is silently ignored and prefill falls back to the slower fp8-specific branch. |
| `--fast-prefill-kernel` | - | prefill an int8 or rk* KV cache with the fast prompt-attention kernel (FP16 PV per 64-key tile) |
| `--concurrent-prefill` | - | admit waiting requests to free lanes while other requests prefill |
| `--wddm-evictable-budget` | - | Windows builds: budget against dedicated memory, committing arenas at startup |
| `--gdn-state-fp16` | — | keeps the GDN recurrent state in FP16 (halves the size of each host-side state image) |
| `--vision` / `--vision-residency` | `overlay` | accept images and video; overlay keeps the Vision tower in host memory and borrows device memory per image |
| `--default-reasoning-effort` | `xhigh` | the effort for requests that set no effort and keep thinking on |
| `--auto-long-anchors` | - | anchor the message boundaries of every request, so a request that rewrites earlier history resumes from the nearest anchor instead of root |
| `--auto-prefix-grid` | - | offer shared prefix candidates on a token grid, so callers whose prompts start alike share a prefix without a client hint |
| `--derive-session-keys` | - | give a request without a session key one derived from its instructions and first user message, so the conversation keeps a session lineage |
| `--spec` / `--draft-tokens` | `dflash2` / 5 | DFlash2 speculative decoding with five drafts per round (1-15 supported) |

The cache options (`--auto-long-anchors`, `--auto-prefix-grid`, `--derive-session-keys`) select
NInfer-all's checkpoint-catalog context cache; the hybrid prefix cache is the upstream default.
`--wddm-evictable-budget` is a no-op outside Windows builds.

The 262,144-token ceiling and four lanes are sized for 48 GB of VRAM, leaving a small system
headroom. The same line runs MTP instead of DFlash2 by replacing
`--spec dflash2 --draft-tokens 5` with `--spec mtp --draft-tokens 3`.

See the [HTTP serving guide](https://github.com/iamwavecut/ninfer-all/blob/master/README.md) for
the API surface and the [resource scheduling reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/resource-scheduling-and-context-cache.md)
for cache and admission semantics.

## Typed decisions (Jev / System One)

This build also serves the **Jev-class "System One" decision API** — TypeSafe's wire format for
decision models, which read a structured `state` plus typed `questions` and return
**a probability for every option in one forward pass, with no text to generate or parse** — the
quicker, cheaper alternative to text generation for triage, routing, moderation, scoring, and
agent/game decision loops. It runs in both wire shapes of the ecosystem:

- `POST /v1/systemone` — TypeSafe's System One request shape (the one llama.cpp, vLLM and Ollama
  serve, so existing Jev/OptiQ clients point at it unchanged), and
- `POST /v1/decisions` — the same reads in OpenAI's Decisions API shape.

No separate decision head is needed: **the served chat model is the decision model**. Each
question becomes one prompt — a fixed system turn asking for one label only (the letter of the
best option, yes or no, or the number of the level), then a user turn with `State:`, `Question:`
and the options or levels — and the answer is read from the first reply token's exact log
probabilities over one-token labels, without sampling past it. Labels: `A`..`Z` for a choice of up
to 26 options, then digits `00`..`99`, `000`.. past 100 options (read a digit at a time), up to
255, `yes`/`no` for a noul (`Yes`/`No` mass added in), `0`.. for a score. `max_tokens` is 1,
thinking is off — the client cannot and need not tune the read.

```bash
curl http://127.0.0.1:10111/v1/systemone -d '{
  "state": {"ticket": "My card was charged twice for one order."},
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this?",
             "criteria": {"billing": "payments and refunds", "shipping": "deliveries"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "tone": {"type": "score", "instructions": "How upset is the customer?",
             "criteria": ["calm", "annoyed", "furious"]}}}'
```

The request takes `state` (a string, JSON, or chat messages; compact JSON is shown as its text in
the prompt) and `questions` (1 to 200 ids, each exactly `{type, instructions, criteria}`, answered
in request order); unknown fields, `joint: true` and a thinking request are refused with 400.
`choice` takes 2 to 255 named options (`{name: description}`), `score` 2 to 10 level descriptions
lowest first, and a `noul` takes no criteria or `{true, false}` descriptions. The response holds
`answers` keyed by question id (a `choice` with its most likely option, `probabilities` by option
summing to 1, and `confidence`; a `noul` with `noul` = P(yes); a `score` with the expected level
index, its `legend` and probabilities), every answer carrying `margin`, top probability minus
second, so a client can send a near tie on to a reasoning generation. `confidence` is the chosen
label's probability over the whole vocabulary (`probabilities[winner] × label_mass`), while
`probabilities` are the option-relative softmax; `diagnostics.<id>` reports `label_mass`,
`argmax_is_label` and `readout` (`exact` on this model's family) for each question.
`POST /v1/decisions` takes OpenAI's shape instead — `input` (a string or user messages with
`input_text`/`input_image` parts), `questions` as an array with `choices` / `levels`, a
`predicate` for a noul; it answers with `answers` as an array in question order (each echoing its
`name`) and `safety_identifier` is accepted and ignored. The same question returns the same
probabilities on both endpoints — it is the same underlying read.

Jev-style extras the engine implements:

- **Images and video** as data URLs (http(s) URLs are refused): `files`/`images` on
  `/v1/systemone`, the state's `image_url` parts, and `input_image` parts of `input` on
  `/v1/decisions` (video on that dialect: use `/v1/systemone`) — needing the `--vision` flag
  already on the line above (501 without it). A request about its media alone may omit `state`; a
  choice asked about media is capped at 26 options (wide labels read digits from token prompts
  that carry no media); the media bytes times the reads may not exceed 512 MiB.
- `conversation` — a full OpenAI chat-completion request whose messages come first, rendered as
  that request's prompt, so a session the context cache holds serves every question at the cost of
  the question's own tokens;
- `runoff: true` — a choice's answer independent of the options' order (pointwise yes/no
  judgements, then ranked listings);
- `recheck_margin` — answers with a smaller margin are re-read alone, so near ties do not depend
  on what else shared their pass.

Vision decisions, measured on this artifact: 12 solid geometric PNGs covering all five shape
types (circles, ellipses, squares, rectangles, triangles in up- and down-pointing) across eight
test colors, one request per image asking its shape (five options) and its color (eight options)
together. Run against the engine on the line above on 2026-10-27, both dialects answered 24/24
correct with zero HTTP errors; every read was `exact` with `label_mass` around 0.99999 (text-only
reads run about 0.98–0.99), and the chosen option's `confidence` was 0.97–1.000 for shape and
0.957–1.000 for color. Each request cost about 760–780 input tokens — image tokens plus the
questions' own text — and a second pass over the same media resumed from the context cache
(2.4–5.8 s first run → 0.7–4.3 s).

Execution and cost: within a request the `state` is prefilled once (or resumed from the context
cache) and each question runs as a branch that sees only the state and itself, so a multi-question
request pays the state's prefill once and each question only its own tokens. A request with media
runs no branched read: its first read prefills the media, the rest resume past it from the context
cache, and repeated requests for the same media hit that cache. `usage.input_tokens` sums every
question's prompt; `output_tokens` counts one token per question.

The probabilities are those of the general chat model under this prompt: a workable Jev-class
decision mode without a trained decision head, but not calibrated to any data — compare options
within a question, not absolute values across questions, and use `diagnostics` to check a read.
The head's logits are BF16, so two options the model scores within one BF16 step (about 0.1 nat at
these magnitudes) share one probability. Full request and response semantics, limits and the
branched-read execution path are in the engine's [serving guide, "Typed
decisions"](https://github.com/iamwavecut/ninfer-all/blob/master/docs/serving.md#typed-decisions).

## Measured on an RTX 4090 (48 GB)

Single card, NInfer-all build `v0.6.0-rtx3090-1211-gf118551f` (built-in `rtx-4090-sm89` device
profile), started with the previous version of the line above (`--spec dflash2 --draft-tokens 5`,
a 278,000-token pool and `--prefill-chunk 2048`), 2026-10-01.

Startup:

| Stage | Value |
|---|---|
| Weights materialized | 27.2 GiB in 5.7 s (4.80 GiB/s) |
| Pinned host state / KV | 918.6 MiB / 8.00 GiB |
| CUDA Graph capture | 3.1 s |
| Engine ready, total | 13.4 s |
| KV pool | 278,016 tokens, FP8 |
| VRAM after startup | 27.2 GiB weights + 13.8 GiB runtime, 6.70 GiB free of 48 GB |

One request (OpenAI Responses stream, thinking mode, two tools, 1,047-token prompt, 25,897
committed output tokens, stopped on the stop token under a 150,000-token output limit):

| Metric | Value |
|---|---|
| Time to first token | 870 ms |
| Wall time | 3 m 17.3 s |
| Prefill | 1.21k tok/s |
| Committed decode | 132.0 tok/s |
| Steady-state 5-second decode windows | 86.2–201.6 tok/s, median 126.6, mean 131.7 |
| Speculation accepted | 20,437 / 32,560 (62.8%); ngram 3,142 / 7,890, 526 rounds |

The rates count committed decode tokens, net of rejected drafts, for one active request; ngram
drafting runs alongside `--spec` by default in this build, so the acceptance figure mixes both
drafters. This is one long thinking-mode answer, not a saturation or benchmark run.

## One-shot CLI

```bash
ninfer models/Qwen3.8-27B-FP8-v3-full.ninfer \
    --prompt "Explain prefill and decode in three sentences." \
    --max-context 32768 \
    --max-new 8192 \
    --kv-dtype fp8 \
    --spec dflash2 --draft-tokens 5
```

For images, videos, and structured chat history, start with `--vision` and see the
[CLI guide](https://github.com/Neroued/ninfer/blob/master/docs/cli.md).

## Supported use

The artifact supports:

- text generation in thinking and non-thinking modes;
- image, multi-image, video, and mixed multimodal messages (start with `--vision`);
- MTP speculative decoding with draft windows from one to fifteen;
- DFlash2 with draft windows from one to fifteen using the included DFlash2 companion weights;
- BF16, INT8, FP8, NVFP4, K8V4, RK8V4, RK4V4, RK4V4-E8, and RK2V4-E8 KV cache;
- CUDA Graph decode and compatible-prefix reuse;
- startup-bounded small-scale concurrent serving with true batched decode;
- the NInfer CLI;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages serving.

## Limits

- NInfer executes on one GPU per Engine (NInfer-all can additionally split the layers over up to
  eight GPUs with `--devices`), with a startup-fixed capacity of 1–8 active requests.
- It does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  CPU/GPU offload, or distributed serving.
- Context allocation is subject to GPU memory and the selected KV-cache type; the documented
  48 GB configuration caps the shared pool at 262,144 tokens.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | `Qwen/Qwen3.8-27B-FP8` (official block-wise FP8 checkpoint) |
| Base model repository | `Qwen/Qwen3.8-27B` |
| DFlash2 source repository | `z-lab/Qwen3.8-27B-DFlash2` |
| Conversion recipe | `qwen3_8_27b` with the `qwfp8ovr.py` override |
| Converter | NInfer v3 converter, `https://github.com/Neroued/ninfer` |
| Converted | 2026-10-01 on a 20-core CPU in ~34 minutes (2,029 s), ~1.3 GB peak RSS, `--rows-per-chunk 1024` |
| Container / artifact id | v3 / `66ef2afa4acd42cfb55dfea28daadb96` |

The full object inventory and conversion provenance are recorded in the `.conversion.json`
report produced alongside the artifact. The exact storage contract is maintained in the
[v3 container reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/artifact-container.md).

## License

This NInfer artifact is distributed under the Apache License 2.0. The source
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) and
[Qwen3.8-27B-FP8](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) repositories are also licensed
under Apache-2.0. Users remain responsible for complying with the license and applicable laws.
