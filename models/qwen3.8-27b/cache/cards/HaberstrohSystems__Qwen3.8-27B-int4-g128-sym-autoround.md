---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - int4
  - w4a16
  - autoround
  - gptq
  - vllm
  - intel-arc
  - xpu
  - battlemage
  - arc-pro-b70
language:
  - en
---

# Qwen3.8-27B — INT4 (AutoRound), tuned for Intel Arc Pro B70

A 27-billion-parameter, natively multimodal model with tool calling, served
from **a single Intel Arc Pro B70**.

INT4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
via [AutoRound](https://github.com/intel/auto-round) (W4A16, group size 128,
symmetric), packed as `auto_round:auto_gptq`. Weights land at **17.69 GiB**
(19.00 GB) across 7 shards plus an extra-tensor file, leaving room on a 32 GB
card for a substantial KV cache.

The configuration is the interesting part: Battlemage rewards a different setup
than CUDA does, and the settings below are the ones that came out ahead. Every
figure here was taken on a single B70, with its measurement conditions stated
alongside it.

## Read this before you start it

**Reasoning is on by default, at the most expensive setting.** The chat
template resolves `reasoning_effort` to `xhigh` unless told otherwise:

```jinja
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
```

Supported levels are `xhigh` (default), `medium` and `low`. At `xhigh`, an
8,000-token prompt took 77–100 s before the first word of the *answer* appeared
in our runs — the model writes a long, invisible reasoning pass first. If you
start the server with the command below and send a plain request, that is what
you get, and it is easy to mistake for a broken setup.

Per request:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
{"chat_template_kwargs": {"reasoning_effort": "low"}}
```

**`preserve_thinking` also defaults to true**, so reasoning blocks from earlier
turns stay in the prompt and context grows faster than a plain turn count
suggests. Set it to `false` if you want conversations to stay compact.

**Sampling:** the shipped `generation_config.json` carries the upstream
recommendation for reasoning mode — `temperature 1.0`, `top_p 0.95`,
`top_k 20`. For instruct-style use without reasoning, upstream suggests
`temperature 0.7`, `top_p 0.80`, `presence_penalty 1.5`.

## What you get

| | |
|---|---|
| Base | Qwen/Qwen3.8-27B (Apache 2.0) |
| Quantization | AutoRound 0.14.2 — 4-bit int, group size 128, symmetric |
| Packing | `auto_round:auto_gptq` |
| Weights on disk | **17.69 GiB** (19.00 GB), 7 shards + `model_extra_tensors.safetensors` |
| Language model | 64 layers, hidden 5120, 24 query / 4 KV heads, head_dim 256 |
| Native context | 262,144 (served at 131,072 — see below) |
| Vision tower | 27 layers, hidden 1152, 16 heads — **kept unquantized** |
| Tool calling | yes — served here with vLLM's `qwen3_coder` parser |
| Reasoning | **on by default at `xhigh`** — switchable per request, see above |
| Tuned on | Intel Arc Pro B70 (Battlemage BMG, 32 GB), single GPU |

### What exactly was quantized

`block_name_to_quantize` is `model.language_model.layers` — the transformer
stack of the language model, and nothing else. Three groups were deliberately
held back at 16-bit:

| Kept at 16-bit | Why |
|---|---|
| `*.linear_attn.in_proj_a` (48 layers) | Gated-DeltaNet input projections are sensitive; int4 here costs more accuracy than it saves memory |
| `*.linear_attn.in_proj_b` (48 layers) | same |
| `*.mtp.fc*` | multi-token-prediction head, left in float |

The **vision tower is untouched** — no entry in the quantization config refers
to it. That is why image understanding survives the compression intact.

AutoRound ran with `batch_size 4`, `gradient_accumulate_steps 2` and
`low_gpu_mem_usage true`. The checkpoint does not record the calibration set,
`nsamples`, `seqlen` or `iters`, so those are not stated here rather than
guessed.

## Measured on one Arc Pro B70

Numbers only mean something with their conditions attached, so each block below
states them. They are not interchangeable — a single-user figure without
reasoning and a three-user figure at `xhigh` describe different workloads.

**Single user, reasoning off, XPU graphs on** (the serving config below):

| | |
|---|---|
| KV cache | **314,572 tokens** |
| Decode | **32.7 tok/s** |
| Time to first token, 500-token prompt | 0.28 s |
| Time to first token, 8,000-token prompt | ~4.5 s |
| Prefill | ~1,800 tok/s |

Decode speed does not depend on prompt size; waiting time does:
`wait ≈ prompt ÷ 1800`, then ~32 tok/s.

**Three concurrent users, reasoning on at `xhigh`, eager mode**, twelve tasks
per run — a deliberately heavy workload, and the reason these aggregates sit
below the single-user figure above:

| 3 users, each holding | aggregate |
|---|---|
| 29k | 20.5 tok/s |
| 60k | 11.3 tok/s |
| 100k | ~3.0 tok/s |

The shape is the point: throughput falls sharply as context grows, far more
than it falls with added users. Keeping conversations compact is the most
effective performance lever on this card. These three rows predate enabling XPU
graphs; expect them to improve, but take the ratios rather than the absolutes.

## Serving

```bash
docker run -d --name vllm --stop-timeout 120 \
  --device /dev/dri:/dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
  --group-add "$(getent group render | cut -d: -f3)" \
  --group-add "$(getent group video  | cut -d: -f3)" \
  --shm-size 2g \
  -v /srv/models:/models:ro -p 127.0.0.1:8000:8000 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DISABLE_ESIMD_GDN_OUTPROJ=1 -e DISABLE_ESIMD_PAGE_ATTN=1 \
  intel/llm-scaler-vllm:0.21.0-b3.1 \
    --model /models/Qwen3.8-27B-int4-g128-sym-autoround \
    --served-model-name qwen \
    --dtype float16 --kv-cache-dtype fp8_e4m3 --max-model-len 131072 \
    --max-num-seqs 3 --max-num-batched-tokens 8192 --block-size 64 \
    --gpu-memory-utilization 0.95 --enable-prefix-caching \
    --trust-remote-code --limit-mm-per-prompt '{"image":3}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3
```

### Host requirements

| | |
|---|---|
| Driver | `xe` (not `i915`) — Ubuntu 24.04 with the HWE kernel |
| GuC firmware | driver asks for **≥ 70.54.0**; Ubuntu ships 70.44.1 in `linux-firmware`. Take the newer blob from the upstream kernel.org tree — the figures here were produced on **70.72.1** |
| Runtime | Intel GPU compute runtime from `repositories.intel.com/gpu/ubuntu` |
| Container image | `intel/llm-scaler-vllm:0.21.0-b3.1` |
| Group access | container needs the host's `render` and `video` GIDs |

## Why this configuration

Each of these was chosen against a measurement, and several run counter to
what works on CUDA.

**`--dtype float16`** — set it explicitly rather than inheriting from
`config.json`. It is the dtype this stack is built around on Battlemage, and
the one to use for coherent output.

**`--kv-cache-dtype fp8_e4m3`** — both fast and roomy here. It yields 318,317
KV slots against 166,285 for fp16, nearly double, and stayed ahead of fp16 on
throughput at every context size we compared. Use the explicit `e4m3` name.

**`DISABLE_ESIMD_GDN_OUTPROJ=1`** — required for int4 weights. Intel's ESIMD
fast path for the Gated-DeltaNet output projection is built for unquantized
layers; switching it off puts the model on the general path, which is exactly
where an int4 checkpoint belongs.

**`DISABLE_ESIMD_PAGE_ATTN=1`** — Intel's ESIMD paged attention is optimised
for GQA ratios of 2, or ≥4 and divisible by 4. Qwen3.8-27B runs 24 query heads
against 4 KV heads — a ratio of 6, which the generic xe_2 decode path handles
directly and faster.

The gain grows with context. Switching ESIMD paged attention off improved
aggregate throughput for three concurrent users by:

| each holding | gain |
|---|---|
| 11k | +3.8 % |
| 32k | +11.9 % |
| 55k | +21.9 % |

Which of those applies to you depends entirely on how long your conversations
get — the longer they run, the more this flag is worth.

**XPU graphs enabled** (i.e. *no* `--enforce-eager`) — worth **+29 %**. This is
the biggest single win and the most surprising one, since graphs are commonly
avoided on Battlemage:

| same workload, same session | eager | XPU graphs |
|---|---|---|
| Decode, single user, 500-token prompt, reasoning off | 25.3 tok/s | **32.7** |
| Requests completed in 30 min, 3 users, reasoning on | baseline | **+32 %** |

Cost: about 66 s of extra startup, once (62.8 s compile + 3 s capture), and a
small amount of KV capacity — 318,317 slots in eager mode against 314,572 with
graphs, i.e. roughly 1.2 %. vLLM separately reports 0.89 GiB taken during graph
capture; that figure and the 3,745-slot difference are not the same accounting,
so do not read one as the cause of the other. This was established against image
`intel/llm-scaler-vllm:0.21.0-b3.1`, `fp8_e4m3` and GuC firmware 70.72.1 — if
you change any of those three, re-check before relying on it.

**`--max-model-len 131072`** rather than the native 262,144. The value also
caps a *single* request: at 262,144 one user could occupy the entire cache and
starve the other two. 131,072 allows 100k of context plus 31k of output and
keeps multi-user service intact.

One observation we can report but not explain: measured KV capacity *rose* with
larger `--max-model-len` on this stack — 293,187 slots at 65,536, 318,317 at
131,072, 332,570 at 262,144. That runs against the usual intuition that a
longer maximum costs cache. Take it as a reason not to shrink the value hoping
to buy capacity, and verify on your own build before relying on it.

**`--max-num-seqs 3`** — the design point for one B70. Idle sessions cost
nothing; vLLM allocates KV only for requests actually in flight, so a single
active writer gets the whole card. Beyond three concurrent *generations*,
requests queue.

**`--limit-mm-per-prompt '{"image":3}'`** — set a limit explicitly. Without the
flag there is no cap, and one request can carry an arbitrary number of images.
Three covers multi-page document work; raise it if your workload needs more,
keeping in mind that the vision tower runs unquantized and image tokens are
correspondingly expensive.

Note this caps images only. The base model also accepts **video**, which is by
far the most expensive input path here — add a `video` entry to the same flag
(or set it to 0) rather than leaving it uncapped.

**`--block-size 64`** — the page size all measurements here were taken with.
We did not compare block sizes, so treat this as the known-good value rather
than a tuned one.

**`--max-num-batched-tokens 8192`** — matches the compile range vLLM captures
graphs for (`compile range (1, 8192)`), so batched prefill stays inside the
captured shapes.

**`--gpu-memory-utilization 0.95`** — the card has no display attached and
nothing else competes for it; 0.95 turns the remaining headroom into KV cache.
Leave lower if the GPU also drives a desktop.

**`--enable-prefix-caching`** — pays for itself in any chat workload, where each
turn resends the whole conversation as prefix. Note it also means a repeated
identical prompt measures the cache, not the model, so vary the leading bytes
when benchmarking.

**`/dev/dri/by-path` mount** — required alongside `--device`, which passes only
the device nodes. oneCCL scans that directory during initialisation.

**`--stop-timeout 120`** (or `stop_grace_period: 120s` in compose) — give the
server room to finish in-flight work before it exits. Cutting a busy GPU
container short can leave the card's scheduler in a state that only a reboot
clears, so let it drain.

**Speculative decoding (MTP): leave it off on this card.** It is the largest
win on CUDA and a loss here — 17–28 % slower above 32k context, because each
user occupies triple query rows, KV slots drop by 24 %, and
`supports_spec_as_decode=False` routes every step through prefill attention.

## Which kernels this actually runs on

Worth knowing, because it explains the flags above and it is not what a CUDA
user would expect.

**A note on the name first.** The `sym` in this repo's name describes the
*numeric scheme*: symmetric int4, zero-point fixed at zero. vLLM separately has
a loader class called `sym_int4`, and the two are not the same thing. Dispatch
keys off the `quant_method: auto-round` string in the config, which vLLM maps
to **`quantization=inc`** (Intel Neural Compressor) — so these weights are
symmetric int4 loaded through the `inc` path, not through the class that shares
the word.

That distinction has a consequence: several int4 ESIMD fast paths
(`_use_esimd_int4_gemv`, `_moe_int4_esimd_ok`, `_gdn_proj_int4_esimd_ok`) test
for the loader name `sym_int4` and therefore stay inactive here, even though
the underlying tensors are byte-identical to what that loader would produce.
Both loaders end up calling the same `int4_gemm_w4a16` kernel.

The two `DISABLE_ESIMD_*` environment switches are unrelated to that gate — they
address two different places:

`DISABLE_ESIMD_GDN_OUTPROJ=1` covers a spot that does *not* check
quantization at all. The eligibility test for the Gated-DeltaNet output
projection reaches straight for `self.out_proj.weight`, which on an int4 layer
is `qweight` — so it has to be switched off explicitly rather than declining on
its own.

`DISABLE_ESIMD_PAGE_ATTN=1` is about attention shape, not quantization at all
(see the GQA note below).

That leaves this arrangement on a B70:

| Component | Path taken |
|---|---|
| Quantized linear layers | oneDNN via the `inc` linear method |
| Gated-DeltaNet output projection | general path (ESIMD variant disabled — it assumes unquantized weights) |
| Attention decode | generic **xe_2** decode from `vllm-xpu-kernels` |
| Paged attention | xe_2, *not* Intel's ESIMD kernel — see the GQA note below |
| `rms_norm`, `fused_add_rms_norm` | `xpu_kernels` with a native fallback |
| KV cache | fp8_e4m3 |
| Graph execution | XPU graphs, piecewise capture |

The attention choice is the one with a measurable price tag. Intel's ESIMD
paged attention is written for GQA ratios of 2, or ≥4 and divisible by 4
(`csrc/eagle/eagle.sycl`). This model runs 24 query heads against 4 KV heads —
a ratio of **6**, matching neither. The xe_2 path handles ratio 6 directly, and
does so faster; hence `DISABLE_ESIMD_PAGE_ATTN=1`.

## Getting the most out of it

**Choose reasoning per request, and use the effort levels.** Reasoning does not
slow generation down — the token rate is essentially the same either way. It
adds a second, invisible answer written before the visible one, and you pay for
it in waiting time.

All figures below were taken at the default effort, `xhigh`. Measured on an
8,000-token prompt: the first word of the answer arrived after ~4.5 s with
reasoning off, and after 77–100 s with it on. On a 500-token prompt: 0.3 s
against 30–48 s. As a rule of thumb, roughly **40 seconds per 1,000 reasoning
tokens** at this decode rate.

Reasoning length is not predictable, even at temperature 0: the same short
prompt produced traces of roughly 750 and 1,180 tokens on consecutive attempts.
On long report-style tasks we have seen traces reach the tens of thousands of
tokens.

Two consequences worth planning for:

- Interactive chat is a different product with `enable_thinking: false` or
  `reasoning_effort: low`. Unattended work — overnight agents, document
  pipelines — is where `xhigh` earns its keep.
- If you cap `max_tokens` with reasoning on, size the cap for the trace *plus*
  the answer. A budget smaller than the trace leaves nothing for the reply.

**Keep working context compact.** Look again at the throughput table: 29k of
context per user gives nearly seven times the aggregate throughput of 100k.
Summarising or rolling older turns out of the prompt buys more speed than any
kernel flag.

**Tool calling holds up as conversations grow.** It behaves consistently at
20k and 60k of context, which is the case that actually matters in production —
tools are rarely called on an empty prompt.

**Plan concurrency around generations, not seats.** Ten people with the tab
open cost nothing. Three writing at once is the design point; the fourth waits.

## Provenance and license

Quantized from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
released under **Apache 2.0**; this quantization carries the same license.

Throughput figures above come from a single Intel Arc Pro B70 running the
configuration shown. Conditions are stated so you can reproduce them on your
own card — and quality is best judged on your own workload.
