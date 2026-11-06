---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- multilingual
tags:
- gemma
- gemma-4
- google
- gguf
- vector-quantization
- vq
- mixture-of-experts
- moe
- 262k-context
- ds4
- ds4-dfm-rs
---

# Gemma-4 26B A4B VQ (R13+8)

A text-only dense-VQ conversion of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)
at revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`, built with recipe
`gemma-4-26b-a4b-family-r8` and served by the `gemma4_vq` family of the ds4
runtime (CUDA only; there is no Metal path for this family). The family is
merged upstream in [Baekpica/ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)
as PR #141 (merge `502019da`). Standard llama.cpp, Ollama and Transformers
loaders do not read this format: the release is a directory whose `release.json`
names three GGUF parts.

> **Status, 2026-11-06.** The release bytes are unchanged. The 2K-64K sweep
> table is the pre-merge build `1a8c5199`, kept as the historical record. The
> whole cold prompts table is the merged main `502019da`, measured 2026-11-06:
> the GB10 runs full-resident BF16 at ctx 262,144, the 4070 SUPER runs the 128K
> pool arm. The merge review changed defaults and removed the launcher and
> advisor. Qualified context is still unset.

The routed experts carry a 13-bit dense-VQ stage (an E4M3FN codebook of 8,192
words, coded as a 12-bit stream plus a 1-bit plane) and a GVR8 residual plane
(a 256-word E4M3 book with one index byte per 8-weight vector), plus one f16
gain per row. Reconstruction is `bf16(gain * (C1 + C2))`. Attention and the
dense FFN projections stay Q4_K, the dense down matrices Q8_0, and the norms,
the per-layer `layer_scalar` tensor, router and token table BF16.

The release is 9.95 GiB, 23,805 tensors, release format 2, at 2.65 bits per
weight (1.6498 stage + 1.0 residual). Weight-space reconstruction error over
the 60 routed-expert stacks is 23.38 percent mean relL2 (23.3931 on an
independent decode, 23.3772 from the build). This is a weight-space figure,
not a logits or quality verdict.

| Part | Role | Tensors | Bytes | SHA-256 |
|---|---|---:|---:|---|
| `gemma4-vq8-trunk.gguf` | 30 layers, tokenizer, chat template | 23,756 | 8,588,461,824 (8.00 GiB) | `0b25b7cb284be1fb59c1bed759e775b589bcae178a9faf18903657a7244d936e` |
| `gemma4-vq8-embedding-BF16.gguf` | BF16 token table, 262,144 x 2,816, read in rows from the host | 1 | 1,476,395,648 (1.38 GiB) | `2043b3362749e781ca47b567d24af70f5282db1c2b93904867f54980b1789636` |
| `gemma4-vq8-mtp-Q4_K.gguf` | optional assistant drafter, Q4_K/BF16 | 48 | 621,995,136 (0.58 GiB) | `f85b05b4b16204aff464c35ffce7b18d054748f7eecaffa5463e9cd3d1d3bb57` |
| **All three parts** | | **23,805** | **10,686,852,608 (9.95 GiB)** | |

`release.json` carries the part roles, sizes, hashes and source revisions;
`files.json` and `SHA256SUMS` cover the rest. Pass the directory to the
runtime, never a single part and never a concatenation.

## At a glance

| | |
|---|---|
| Source | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) at `4d7ae4984b7db7de8f8457170b3f1a419ee76d52` (BF16, 2 shards); drafter from [`google/gemma-4-26B-A4B-it-assistant`](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant) at `6e5aaaf4c42b98394530b8fda2e95cadd65c151c` |
| Recipe | `gemma-4-26b-a4b-family-r8` ([RECIPE.md](RECIPE.md), [`recipe.json`](recipe.json)) |
| Routed experts | 30 MoE layers x 128 experts x gate/up/down, plus one shared/dense FFN of 2,112 in every layer. 8-dim VQ, one E4M3 main book and one 256-entry E4M3 residual book per expert stack, one F16 gain per row (1.0 in this uncalibrated build). **Main: 13-bit (K=8192, 12-bit stream + 1-bit plane); residual: 8-bit GVR8** - 1.64 + 0.59 index bits per weight. |
| Attention, dense FFN gate/up | Q4_K; the dense down matrices are Q8_0 because 2,112 columns are not a multiple of the Q4_K block of 256; the tied output head is the BF16 embedding |
| Kept at source precision | Router weights, norms, the per-layer `layer_scalar` tensor, token embedding and tied output head (BF16) |
| Token embedding | BF16, in its own part; `release.json` marks it `ssd-rows`. On a GB10 the runtime maps it for device access and the gather and the head read one table; on a discrete card the engine copies the 9.37 GiB trunk and embedding to the device. |
| Context | **262,144 tokens**, the model's own: 25 sliding-window-1024 layers and 5 full-attention layers; no RoPE scaling |
| Chat template | The official Gemma 4 canonical Jinja template, carried in the trunk part under `tokenizer.chat_template`; thinking is the `<|channel>thought ... <channel|>` channel; tools are `<|tool_call>...<tool_call|>` |
| Fidelity (reference forward, release bytes vs fp32) | Probe A, 28 tokens: cosine **0.9906**, KL **0.000548**, argmax 818; probe B, 1,243 tokens: cosine **0.9901**, KL **0.419508**, argmax 1509. No held-out KL, perplexity or benchmark. |
| Drafter | `google/gemma-4-26B-A4B-it-assistant`, 48 tensors, Q4_K/BF16, 0.58 GiB; `--mtp` and `--mtp-draft` are declared, not qualified |
| Runtime | [`Baekpica/ds4-dfm-rs`](https://github.com/Baekpica/ds4-dfm-rs) `main` at merge `502019da`; CUDA on sm_89 (RTX 4070 SUPER) and sm_121 (GB10) |
| RTX 4070 SUPER speed | 128K pool arm, merged main `502019da` (2026-11-06): 1,275.9 tok/s prefill and 41.2 tok/s decode at 11,437 tokens; 904.6/34.6 at 124,648; peak 11,018 MiB. |
| GB10 full context | 248,182 tokens: 730.7 tok/s prefill, 16.9 tok/s decode; BF16 full-layer KV, 8,192-row chunks, ctx 262,144, merged main `502019da`. |

## Model structure

Gemma 4 26B A4B is a hybrid-attention MoE: 30 layers, 25 sliding-window
(1,024) and 5 full-attention, with a dense FFN and 128 routed experts in every
layer. Total parameters are 25.2B, active 3.8B; the routed-expert stacks are
90.5% of the encoded elements, so that is where the VQ budget goes. The
source's vision tower (~550M parameters) is not included in this release.

The split is visible in the weights: the 60 routed-expert stacks hold
22,837,985,280 weights (22.84B), so the dense skeleton is about 2.36B. With 8
of 128 experts active per layer (1.43B), the active set is the source's 3.8B.

```
embed x sqrt(2816) -> [ rms -> GQA (sliding 1,024 or full) -> +h
                        rms -> dense FFN 2,112 -> +h
                        rms -> top-8 of 128 experts (FF 704) -> +h
                        x layer_scalar ] x 30 -> rms -> tied head -> 30*tanh(x/30)
```

| Field | Value |
|---|---|
| Layers | 30: 25 sliding at window 1,024, 5 full at layers 5/11/17/23/29 |
| Sliding layers | 16 query heads, 8 KV heads, head_dim 256, RoPE theta 1e4 on all 128 pairs |
| Full layers | head_dim 512, 2 KV heads, proportional RoPE (partial_rotary_factor 0.25, 64 of 256 pairs), theta 1e6, `attention_k_eq_v` |
| Hidden / dense FFN | 2,816 / 2,112 |
| Experts | 128 routed per layer, top-8, FF 704; one shared/dense FFN of 2,112 |
| Vocabulary | 262,144, tied to the head |
| Logit softcap | 30 |
| RMS eps | 1e-6 |
| Native context | 262,144 (`max_position_embeddings`) |

### What is quantized

| Part | Storage |
|---|---|
| Routed experts (60 stacks: 30 layers x gate_up/down) | GVQ1 R13+8: 13-bit main index (12-bit stream + 1-bit plane) into an 8,192-word E4M3 book, 8-bit GVR8 residual into a 256-word E4M3 book, one F16 gain per row; 2.65 bits per weight |
| Attention q/k/v/o and output-gate | Q4_K |
| Dense FFN gate/up | Q4_K |
| Dense FFN down | Q8_0 (2,112 columns are not a multiple of the Q4_K block of 256) |
| Router, norms, the per-layer `layer_scalar` tensor, token embedding and tied output head | BF16 |
| Drafter (optional) | Q4_K/BF16 |

The VQ budget goes to the 60 expert stacks because they are 90.5% of the
encoded elements. The dense skeleton and the token table stay at the cheaper
formats because they are read on every token.

## Why this size: the 12 GiB card decides it

Gemma 4 26B A4B is 25.2B parameters total and 3.8B active. The routed-expert
stacks are 90.5% of the encoded elements, so that is where the VQ budget goes.
The release is 9.95 GiB: an 8.00 GiB trunk, a 1.38 GiB embedding part and a
0.58 GiB optional drafter. On a GB10 the release is mapped for device access.
The 128 GB unified pool holds the full 9.37 GiB model, 6.76 GiB BF16 KV and a
3.26 GiB session workspace at 262,144 with 96.81 GiB spare; no expert pool or
CPU offload is used.
On a discrete 12 GiB card there is no pageable memory access, so the engine
copies the 9.37 GiB trunk and embedding to the device; the three placement
lanes then trade VRAM for speed. The 128K pool arm peaks at 10,916 MiB of
12,282.

The full 262,144-token context is the model's own, but only ctx 2,048 has a
serving-check record; qualified context is unset. Servers booted at 262,144
and answered a 248,182-token prompt on a GB10 and 126K on the 12 GiB card, but
no gate covers the range past the 64K sweep. `ds4-server` keeps a 384 MiB
free-memory floor and refuses to listen, naming the deficit, when the serial
session cannot hold it; on a 12 GiB card set `DS4_SESSION_GRAPH_HEADROOM_MB=0`
for the 128K pool arm. `--ssd-streaming` streams expert records through the
slot pool; on the 4070 the slot pool is the streaming lane that works, and the
resident path is the one measured for the speed rows.

## Recipe

- **Routed experts.** Every 8 consecutive weights of an expert row become one
  index into an E4M3 codebook, times one F16 row gain. One main book per
  expert stack serves all 128 experts and the gate/up/down matrices; one
  256-entry E4M3 residual book per stack serves the GVR8 plane. The main index
  is 13 bits (K=8192): a 12-bit stream plus a 1-bit plane. The residual is one
  8-bit index per 8-weight vector. Reconstruction is
  `W[r, 8j+d] = gain[r] * (C1[i1][d] + C2[i2][d])`, rounded to the runtime's
  BF16 storage.
- **Codebook training.** The encoder strided-subsamples at most 200,000
  segments, starts from evenly spaced samples, runs eight Lloyd iterations,
  and round-trips every centroid through f16 and E4M3 before the final
  assignment. The builders pass `X = NULL`: the codebooks are fitted to the
  weights alone, with no calibration corpus and no activation importance. The
  F16 row-gain field is therefore `1.0` throughout this artifact; the codeword
  carries the whole approximation. Calibration is the remaining quality
  lever, not a property of the format.
- **What is VQ.** 60 expert stacks (30 layers x gate_up/down). The emitter
  splits a fused gate_up stack into gate and up, and each stack into one matrix
  per expert: 11,520 GVQ1 matrices, 11,520 GVR8 residuals and 120 books in the
  trunk.
- **What is not VQ.** Attention q/k/v/o and output-gate, and the dense FFN's
  gate/up are Q4_K; the dense down matrices are Q8_0; norms, the per-layer
  `layer_scalar` tensor, router weights and the token embedding are BF16, and
  the tied output head reads the same BF16 table. The embedding is a separate part. The source's
  FP32 router logits, tanh-GELU expert activation and tanh-30 vocabulary cap
  stay in the runtime.
- **Weight-space reconstruction.** 23.38% mean relL2 over the 60 expert stacks
  at about 2.22 bits per weight, against 39.29% for the same stacks at 1.6377
  bits per weight without the residual; the residual recovers 15.8 points for
  0.59 more bits per weight. The payload containers are 7.01 GiB (4.35 stage
  plus 2.66 residual). On layer 0's `down_proj`, the measured error is 23.4212%
  with the residual against 39.2647% without; the residual is 31,719,552 bytes
  in a 52,330,552-byte container.

## Measured on the converted bytes

### Weight-space check

An independent decode of the release's 60 expert stacks measures 39.2897% relL2
at stage 0 (without the GVR8 residual), cosine 0.919582, 8,192/8,192 codewords
used, index entropy 12.87, 1.6377 bits per encoded weight, and no copied
tensors. The build's combined figure, including the GVR8 residual, is
22.8066-24.1997% per stack, mean 23.3763%. On one expert, the residual-inclusive
decode measures 24.0021% against 39.5105% without the residual; the stage-1
decode equals the source container bit for bit (1,982,464 values), and the
residual-inclusive decode equals the encoder's plane and book bit for bit.

The reference gate ran on a GB10 and compared post-softcap full-vocabulary
distributions at the last position against transformers at commit `cc1451df`
(fp32 reference, same ids, thinking off). The release side reads the weights
from this repository's files; the reference side reads the BF16 checkpoint.

| Probe | ids | Artifact | cosine | KL ref/port | top-10 | argmax |
|---|---:|---|---:|---:|---:|---:|
| A short | 28 | **R13+8** | **0.9906** | **0.000548** | 6/10 | 818 |
| A short | 28 | 13-bit | 0.9879 | 0.002371 | 6/10 | 818 |
| B long | 1,243 | **R13+8** | **0.9901** | **0.419508** | 8/10 | 1509 |
| B long | 1,243 | 13-bit | 0.9873 | 0.003730 | 8/10 | 1509 |
| C window edge | 1,024 | 13-bit | 0.9801 | 0.00038 | 8/10 | 1509 |
| D first eviction | 1,025 | 13-bit | 0.9800 | 0.00038 | 8/10 | 1509 |

Read the B KL carefully: it is one token. 1509 contributes 0.425 of the
0.4195, because this artifact puts the top logit 0.6 nats below the reference
(24.5 against 25.125), where the 13-bit artifact put it 0.75 above (25.875).
Every shape metric improved: cosine, rms delta (2.44 to 2.38 on A, 1.72 to
1.36 on B) and the runner-up order. Weight error 39 to 23 percent is a better
approximation of the same model, not a different model; no claim beyond these
numbers, and no perplexity or benchmark was run.

**Serving checklist.** `ds4 --serving-check` on this artifact at ctx 2,048
(sm_89, 2026-11-03): prefix, rewind, continue, disk session, snapshot, banks
warm/fork/partial/disk/width - 10 works, 0 FAIL, 0 WRONG, 0 undeclared. The
check is a structural and numerical comparison against the live state it
saved; it is not a quality claim.

**Speed.** The current records are in Performance: the GB10 2K-64K sweep and
the 4070 SUPER 128K pool arm. The early `d4f6d4f9` 13.42 tok/s record is
superseded.

## Runtime surface

**OpenAI-compatible serving.** `/v1/chat/completions`, `/v1/completions`,
`/v1/messages` and `/v1/responses`, streaming included, with live tool calls
(the tool result is carried back into the transcript in the canonical shape)
and the thought channel delivered separately from the answer. The trunk
carries the official Jinja template. Context: 262,144 on a GB10, 65,536
resident on a 12 GiB card, and 131,072 on the same card with the expert slot
pool.

**Two attention geometries and one KV format per lane choice.** The sliding
and full layers run their own tensor-core prefill kernels; decode uses split-K
attention with `DS4_GEMMA4_ATTN_SPLIT_SPAN` (default 64, 16..4096), and the
five full layers keep either BF16 rows (default on GB10, tensor-core) or E4M3
rows (one 2,056-byte combined row per token instead of 4,096; the same
tensor-core kernels since the port, decoding each element on the way into the
shared-memory tile).

**Expert placement, the llama.cpp-shaped lanes.** `DS4_GEMMA4_CPU_MOE=1`
keeps every expert host-cold in the mmap and runs the top-8 on the CPU with an
AVX2 R13+8 GEMV. `DS4_GEMMA4_MOE_CPU_LAYERS=N` sends layers below N to the
host and keeps the rest device-resident. `DS4_GEMMA4_MOE_CACHE_MIB=N` keeps
every expert host-cold and fills N MiB of VRAM slots from the mapping as the
router asks, with `DS4_GEMMA4_MOE_MISS=evict|cpu` (evict by default) and the
staging ring (`DS4_GEMMA4_MOE_STAGING_MIB`, `DS4_GEMMA4_MOE_STAGING_PREAD`,
`DS4_GEMMA4_MOE_WARM`). `DS4_GEMMA4_MOE_TRACE=1` prints a per-layer routing
histogram at exit, the sizing input for the pool. The fill, the parking lane
and the tokenizer numbers are in Performance.

**The serving quote and the boot admission.** The quote prices the KV rows,
the session workspace (activation set + linear workspace + routed-expert
scratch), the expert slot pool and the device-resident spans. The server
refuses to boot when the serial session cannot keep 384 MiB free at peak,
naming the deficit:

```
ds4: Gemma-4 session does not fit: ctx=8192 chunk=4096 need=2.61 GiB
     (KV 1.05 GiB + workspace 1596.5 MiB: activation 594.3 + linear 32.1
     + moe 970.1) + margin 384 MiB > free 2823.75 MiB (deficit 237.1 MiB)
```

**Environment knobs, not flags.** All arithmetic switches are
`DS4_GEMMA4_*` environment variables; the merged tree has no `--gemma4-*`
flags. Set them before starting a fresh process. The full list is in the
[family page](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/gemma4-vq.md).

## Performance

### GB10 (DGX Spark), pre-merge 2K-64K sweep

The sweep (server binary `1a8c5199`, R13+8 parts of the release's byte sizes,
2,048-row chunks, 128 generated tokens per frontier) ran outside the clock
policy: busy SM clocks were 2,437-2,450 MHz, above the 300-2,200 MHz range
every other family's GB10 row uses, and the record carries no power or thermal
samples. Only ratios measured in one session are comparable. This table is
kept as the historical 2K-64K record; the current merged-main rows are below.

| Frontier | Prefill tok/s | Decode tok/s |
|---:|---:|---:|
| 2K | 1,061.5 | 35.7 |
| 8K | 1,077.9 | 34.9 |
| 16K | 1,044.3 | 33.7 |
| 32K | 929.2 | 31.8 |
| 63K | 833.9 | 28.1 |
| mean of 31 frontiers | 957.2 | 31.75 |

Each row prefills the next ~2,048 tokens over the cached prefix. The 65,536
frontier is missing: the byte-estimated prompt landed a few tokens over the
booted context and was refused.

### Whole cold prompts

GB10 rows: merged main `502019da`, full-resident, BF16 full-layer KV,
8,192-row chunks, ctx 262,144, 32 greedy tokens, the response's `timings`,
busy SM clock 2,398 MHz, outside the 300-2,200 MHz policy. 4070 SUPER rows:
merged main `502019da`, measured 2026-11-06, 128K pool arm (E4M3 full-layer
KV, 512-row chunks, 5,856 MiB expert pool, evict), cold prefix, 32 greedy
tokens, cached 0; peak VRAM 11,012-11,018 MiB.

| Host | Prompt tokens | TTFT | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| GB10 | 11,450 | 10.43 s | 1,099.2 | 34.8 |
| GB10 | 46,257 | 44.80 s | 1,033.9 | 29.9 |
| GB10 | 64,031 | 64.36 s | 996.2 | 28.2 |
| GB10 | 124,664 | 140.86 s | 886.4 | 23.0 |
| GB10 | 248,182 | 340.14 s | 730.7 | 16.9 |
| 4070 SUPER | 11,437 | 9.00 s | 1,275.9 | 41.2 |
| 4070 SUPER | 46,244 | 40.16 s | 1,153.8 | 39.5 |
| 4070 SUPER | 64,016 | 59.09 s | 1,085.2 | 37.5 |
| 4070 SUPER | 124,648 | 138.01 s | 904.6 | 34.6 |

The GB10 KV/chunk sweep picked BF16 full-layer KV with 8,192-row chunks for
prefill: 1,099.2 tok/s at 11K and 1,033.9 at 46K. FP8 full-layer KV was the
better decode lane at 46K (31.9 tok/s against BF16's 30.1) but prefilled at
890.7 tok/s against 1,028.4, and 512-row chunks were slower on both. The
full-resident BF16-8192 configuration is the one left serving.

### 12 GiB card placement

Cold server per arm, 32 greedy tokens, empty prefix cache; VRAM is boot/peak.
These placement rows are build `8137cb52`; the served 128K arm's current
numbers are in the whole cold prompts table.

| Mode | ctx | Prompt | TTFT | Decode | VRAM |
|---|---:|---:|---:|---:|---|
| resident | 8,192 | 7,555 | 5.87 s | 54.8 tok/s | 10,121/10,861 MiB |
| resident | 32,768 | 31,909 | 26.45 s | 52.5 tok/s | 10,121/11,113 MiB |
| resident | 65,536 | 31,909 | 26.43 s | 51.8 tok/s | 10,121/11,433 MiB |
| CPU MoE (`DS4_GEMMA4_CPU_MOE=1`) | 8,192 | 7,555 | 362.8 s | 17.3 tok/s | 3,617/3,673 MiB |
| CPU MoE (`DS4_GEMMA4_CPU_MOE=1`) | 65,536 | 31,909 | 1,410.0 s | 19.2 tok/s | 4,177/4,233 MiB |
| cache 4 GiB, `miss=cpu` | 8,192 | 7,555 | 22.1 s | 28.0 tok/s | 7,713/7,773 MiB |
| cache 4 GiB, `miss=cpu` | 65,536 | 31,909 | 82.0 s | 27.6 tok/s | 8,273/8,345 MiB |

The host lane costs about 1.6 ms per token per CPU layer: six layers turn a
7,555-token prefill from 5.9 s into 42 s for 734 MiB less VRAM. The CPU arms
shared the host with other jobs, so their decode is a floor.

### The expert fill

One prompt (7,239 tokens, ctx 8192, chunk 2048) with the reply md5
`57a0da50adbfbc90ca06b5e73cb6d842` in every row, so the changes are
token-identical:

| Step | TTFT | Prefill tok/s |
|---|---:|---:|
| cache 4 GiB, chunk 512, linear scan | 22.0 s | 329 |
| + chunk 2048 | 13.8 s | 524 |
| + pool 7 GiB | 12.2 s | 594 |
| + per-layer slot map | 10.1 s | 718 |
| + warm the mapping at open | 5.71 s | 1,279 |

### Disk KV parking

On the 4070 SUPER, every request cold (an evicting prompt between them), 64
greedy tokens; A is a conversation's first turn, B the same plus a short
follow-up after the live session was dropped:

| Prompt tokens | A off | A on | B off | B on | B gain |
|---:|---:|---:|---:|---:|---:|
| 8,294 | 6.26 s | 7.74 s | 6.65 s | 0.66 s | 10.0x |
| 16,243 | 12.54 s | 14.30 s | 12.94 s | 0.70 s | 18.5x |
| 31,909 | 26.29 s | 28.64 s | 26.79 s | 0.74 s | 36.4x |
| 45,737 | 39.78 s | 42.54 s | 40.23 s | 0.80 s | 50.5x |

The cost is the cold turn's write: 1.10 s per 989 MB checkpoint at ctx 65,536.
The device sees about 1.98 GB per checkpoint, because the native payload is
staged to a file and copied into the KVC file, both fsynced.

### Tokenizer

The BPE finds the lowest merge rank with a min-heap instead of rescanning the
symbol list; the merge order ranks created pairs at once, and the token stream
is byte-identical to the reference's `merge_all`.

| Request on a GB10 | Before | After |
|---|---:|---:|
| 61,872 tokens, cold | 98.20 s TTFT | 62.92 s TTFT (62.84 s of prefill) |
| 61,871 tokens, fully cached | 34.46 s TTFT | 0.12 s TTFT |
| 60 KB tokenize | 7.43 s | 0.27 s |

## Serving with ds4

Build the runtime (`make cuda CUDA_ARCH=sm_89` on a 12 GiB card, `sm_121` on a
GB10), then serve the release directory:

```sh
# GB10: bf16 full-layer KV, the model's own 262,144-token context
./ds4-server --cuda -m <model-dir> --native-chunk 2048

# 12 GiB card, 65,536 tokens resident: E4M3 full-layer rows
DS4_GEMMA4_FULL_KV=fp8 DS4_SESSION_GRAPH_HEADROOM_MB=0 \
  ./ds4-server --cuda -m <model-dir> -c 65536

# 12 GiB card, 131,072 tokens: experts in a 5,856 MiB VRAM slot pool
DS4_GEMMA4_FULL_KV=fp8 DS4_GEMMA4_MOE_CACHE_MIB=5856 \
  DS4_SESSION_GRAPH_HEADROOM_MB=0 \
  ./ds4-server --cuda -m <model-dir> -c 131072

# the optional drafter (declared, not qualified)
./ds4-server --cuda -m <model-dir> --native-chunk 2048 \
  --mtp <model-dir>/gemma4-vq8-mtp-Q4_K.gguf --mtp-mode on --mtp-draft 2
```

`<model-dir>` is the extracted release directory and the advertised
`/v1/models` id is that directory's name. The three parts are independent GGUF
files: `gemma4-vq8-trunk.gguf`, `gemma4-vq8-embedding-BF16.gguf` and
`gemma4-vq8-mtp-Q4_K.gguf`. Do not concatenate them.

- Standard llama.cpp, Ollama and Transformers loaders do not read this format.
- The release is text-only. The vision tower is not included. Audio is declared
  in the checkpoint but has zero tensors and `audio_config: null`; do not wire
  it from the processor configuration.
- The trunk part carries the tokenizer and the official chat template under
  `tokenizer.chat_template`; the runtime renders the prompt with the shared
  Jinja engine.
- The drafter part is present and validated by the inventory; `--mtp` and
  `--mtp-draft` are declared, but the drafter is not qualified.

Environment knobs used above: `DS4_GEMMA4_FULL_KV=bf16|fp8`,
`DS4_GEMMA4_MOE_CACHE_MIB`, `DS4_GEMMA4_CPU_MOE`,
`DS4_GEMMA4_MOE_CPU_LAYERS`, `DS4_GEMMA4_PREFILL_CHUNK`,
`DS4_GEMMA4_ATTN_SPLIT_SPAN`, `DS4_SESSION_GRAPH_HEADROOM_MB`. Every one is an
environment variable; the merged tree has no `--gemma4-*` flags.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
reply = client.chat.completions.create(
    model="<model-dir-name>",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```

## What stays native

- **Context:** 262,144 tokens. Sliding layers: 25 layers, window 1,024,
  head_dim 256, 8 KV heads, RoPE theta 1e4 on all 128 pairs. Full layers: 5
  layers (5, 11, 17, 23, 29), head_dim 512, 2 KV heads, proportional RoPE
  (partial_rotary_factor 0.25, 64 of 256 pairs), theta 1e6, and
  `attention_k_eq_v` so the value projection is the key projection. Nothing is
  rescaled or extended.
- **Sampling:** the source `generation_config.json` (do_sample true,
  temperature 1.0, top_p 0.95, top_k 64). Requests that omit sampling take
  these values.
- **Numerics kept in FP32 by the source:** router logits, tanh-GELU experts,
  logit soft-capping at 30.
- **Chat:** the official Jinja template, rendered by the shared Jinja engine.
  Thinking is the `<|channel>thought ... <channel|>` channel; tools are
  `<|tool_call>call:NAME{args}<tool_call|>`. Stop ids are 1, 106 and 50.
  Special tokens: turn 105/106, channel 100/101, think 98, tool 46-51, image
  258880, video 258884, audio 258881 (declared, unbacked), eos 1, bos 2.

## Files

| Path | Contents | Size |
|---|---|---:|
| `gemma4-vq8-trunk.gguf` | The text model: 11,520 GVQ1 + 11,520 GVR8 + 120 GVBC, 175 Q4_K, 30 Q8_0, 391 BF16; 23,756 tensors | 8,588,461,824 B (8.00 GiB) |
| `gemma4-vq8-embedding-BF16.gguf` | BF16 token table, 262,144 x 2,816, one tensor | 1,476,395,648 B (1.38 GiB) |
| `gemma4-vq8-mtp-Q4_K.gguf` | The assistant drafter, 48 tensors, Q4_K/BF16 | 621,995,136 B (0.58 GiB) |
| **All 3 parts** | | **10,686,852,608 B (9.95 GiB)** |
| `release.json` | Entry point: every part with role, size and SHA-256 | |
| `files.json` | Index of every file with its size and SHA-256 | |
| `config.json` | The checkpoint config, byte-identical to the copy the runtime validates | |
| `recipe.json` | The recipe's machine-readable form | |
| `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `processor_config.json` | Original files, unchanged | |
| `RECIPE.md`, `FORMAT.md`, `SOURCE.md` | Recipe, byte format, provenance | |
| `SHA256SUMS` | Hashes of every file in this repository | |
| `LICENSE` | Apache-2.0 | |

The Hub's GGUF panel parses one `.gguf` file at a time. Its "Model size" comes
from the BF16 embedding part (0.7B), not the 25.2B text model, and its variant
selector reads the filenames. The `BF16` and `Q4_K` suffixes are for that panel;
the release is the directory and `release.json`, and the panel's selector is
not the runtime. The hardware line is the viewer's suggestion for the selected
file, not a requirement; the release runs on a 12 GiB RTX 4070 SUPER.

## Upstream scores

Google's scores for the original checkpoint, from the source model card. They
describe the source model, not this quantization. The 26B A4B has no audio, so
the CoVoST and FLEURS rows do not apply; the source card lists them as "-".

| Benchmark | Gemma 4 26B A4B |
|---|---:|
| MMLU Pro | 82.6% |
| AIME 2026 no tools | 88.3% |
| LiveCodeBench v6 | 77.1% |
| Codeforces ELO | 1718 |
| GPQA Diamond | 82.3% |
| Tau2 (average over 3) | 68.2% |
| HLE no tools | 8.7% |
| HLE with search | 17.2% |
| BigBench Extra Hard | 64.8% |
| MMMLU | 86.3% |
| MMMU Pro | 73.8% |
| OmniDocBench 1.5 (average edit distance, lower is better) | 0.149 |
| MATH-Vision | 82.4% |
| MedXPertQA MM | 58.1% |
| CoVoST | - (no audio) |
| FLEURS (lower is better) | - (no audio) |
| MRCR v2 8 needle 128k (average) | 44.1% |

## Limits

- The drafter is present, not qualified: draft 6 lost to plain decode at a
  13K prompt (rounds keep 1-2 tokens), its 2-8-row verify blocks stay on the
  tile attention kernel, and its exact-path byte gate fails because the exact
  switch does not cover every batched piece of the verify forward. The final
  build's plain decode has no matched MTP run.
- The qualified context is unset: servers booted at 262,144 and answered a
  248,182-token prompt (GB10) and 126K (12 GiB card), but no gate covers the
  range past the 64K sweep.
- With `DS4_GEMMA4_MOE_CPU_LAYERS` between 1 and 29, or a slot pool, at ctx
  8,192, a 7,555-token prompt was refused as exceeding the context while N = 0
  and N = 30 accepted it: the session context is right-sized below the
  advertised one on those lanes.
- Disk KV parks the whole cache, sliding rings included (989 MB at 65,536),
  writes it twice (a staged native payload copied into the KVC file, both
  fsynced; 1.10 s per checkpoint), and a request that restored a checkpoint
  stores nothing deeper, so a growing chat restores one turn behind.
- The tokenizer splits every GGUF control and user-defined token, where
  tokenizers splits only the release's 23 added tokens: raw text that spells
  another control token can encode differently.
- No teacher-forced NLL corpus or long-context quality gate exists for the
  family; the reference gate above covers single positions.
- Text-only: the vision tower is a sidecar part, inventoried and refused,
  never encoded; audio is declared but unbacked.
- This is lossy: 23.38% mean relL2 over the routed experts at about 2.22 bits
  per weight. The release has no held-out KL, perplexity or benchmark; the
  reference gate is two probes, not a corpus.
- The B probe's KL is 0.4195, one token's 0.6-nat deficit at the sampled
  position; the distributions are close, not equal, and the residual is the
  encoding's error.
- The 65,536 frontier is missing from the sweep; the resident 65,536 arm is
  refused at boot with a desktop compositor resident (94.2 MiB short).
- Token identity is defined against `DS4_GEMMA4_MOE_PERSIST=0`; the persistent
  MoE decode is bit-identical on 26 of 30 layers.
- The E4M3 full-layer lane runs on tensor cores since `6dfe55cd`: at 32,768
  tokens it prefills at 746.6 tok/s against bf16's 804.6 and decodes at 19.27
  tok/s against 18.71. The 9.6x prefill and 61x decode figures at the same
  context belong to the scalar fallback (`DS4_GEMMA4_ATTN_MMA=0` with
  `DS4_GEMMA4_ATTN_SPLIT_MMA=0`), not to the lane.
- The 2K-64K sweep is the pre-merge build `1a8c5199`; the whole cold prompts
  table is the merged main `502019da`, measured 2026-11-06 (GB10 full-resident
  BF16 at 262,144; 4070 SUPER 128K pool arm). The GB10 rows are outside the
  300-2,200 MHz clock policy.
- The GB10 full-context lane decodes 16.9 tok/s at 248,182 tokens (BF16 KV,
  8,192-row chunks); the 4070 SUPER's 128K pool arm decodes 34.6 at 124,648.
  The GB10's lower memory bandwidth is the limit; FP8 KV raises the 46K decode
  to 31.9 but costs prefill (890.7 against 1,028.4 tok/s).

## Links

- Runtime: [Baekpica/ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)
- Merge: [PR #141](https://github.com/Baekpica/ds4-dfm-rs/pull/141)
- Family page: [docs/gemma4-vq.md](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/gemma4-vq.md)

## Credits and sources

- **[Google DeepMind](https://deepmind.google/models/gemma/)** - Gemma 4
  26B A4B, Apache-2.0.
- **R13+8 recipe (this release)** - the conversion line is JLFN's own remake:
  a 13-bit main index (12-bit stream plus 1-bit plane) into an 8,192-word
  E4M3 book, an 8-bit GVR8 residual into a 256-word E4M3 book, one F16 gain
  per row, and E4M3 codebooks. It is not YoungAi's recipe.
- **[GLM-5.3 Flash VQ](https://github.com/Baekpica/ds4-dfm-rs/blob/main/docs/glm53-vq.md)** -
  the GVQ1 container and the expert kernels the R13+8 residual stage runs on.
- **Beam-1.0 VQ and MiMo-V2.6 VQ** - the E4M3 full-layer row format the
  runtime follows.
- **[ds4-dfm-rs](https://github.com/Baekpica/ds4-dfm-rs)** - the runtime: the
  GVQ1/GVR8 reader, the Gemma 4 family and its CUDA kernels. The family is
  merged upstream as PR #141 (merge `502019da`); the runtime is the upstream
  `main`, not a separate branch or worktree.
- **[llama.cpp / GGML](https://github.com/ggml-org/llama.cpp)** - the GGUF
  container and the Q4_K and Q8_0 layouts.

## License

Apache-2.0, as the source model. The release includes a copy of the
Apache-2.0 license. Gemma is a trademark of Google; this is an unofficial
quantization, not a Google release.
