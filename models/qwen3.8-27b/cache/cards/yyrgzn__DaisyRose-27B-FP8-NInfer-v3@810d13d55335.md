---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - RadixArk/DaisyRose-27B-FP8
  - RadixArk/DaisyRose-27B
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - daisyrose
  - qwen3.8
  - fp8
  - e4m3
  - reinforcement-learning
  - agent
  - speculative-decoding
  - dflash2
  - mtp
  - multimodal
  - conversational
  - cuda
  - rtx-4090
  - rtx-pro-6000
---

# DaisyRose-27B FP8 for NInfer

This repository contains `DaisyRose-27B-FP8-v3-full.ninfer`, the full NInfer v3 artifact for
[RadixArk's DaisyRose-27B](https://huggingface.co/RadixArk/DaisyRose-27B), built from the official
[RadixArk/DaisyRose-27B-FP8](https://huggingface.co/RadixArk/DaisyRose-27B-FP8) block-wise FP8
checkpoint. The artifact is intended only for NInfer; it is not a Transformers checkpoint,
Safetensors distribution, or GGUF file. It is the same artifact family as the NInfer conversion of
the official Qwen3.8-27B-FP8 checkpoint, built with the identical recipe, override, and
verification scope.

DaisyRose-27B is RadixArk's first model: a multimodal reasoning model built on
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with reinforcement learning only, without
any supervised fine-tuning or distillation. It was trained on about 1,200 curated long-horizon
agent tasks, with sessions of up to 1M tokens, for 18 days on 144 H200 GPUs. Terminal Bench 4.1
pass@1 rises from 5.6 to 12.0; the full table is under [Model overview](#model-overview). It keeps
the base model's architecture, tokenizer, and chat template, along with its image understanding
and 256K-token context, so it is a drop-in replacement for Qwen3.8-27B.

The Text tower keeps the official E4M3 quantization. The checkpoint stores the linear
projections as E4M3 codes with a BF16 `weight_scale_inv` grid per 128×128 weight block; the
converter decodes each block exactly (an E4M3 mantissa times a BF16 scale fits FP32 without
rounding) and re-encodes the values with one BF16 scale per output row (`fp8_row_maxabs`). The
token embedding, output head, and every text projection except the GDN `a`/`b` gating
projections (kept direct BF16) use that row-scaled FP8, which the engine runs on its
FP8-activation routes. MTP uses Q8 group-32, Vision is full BF16, and DFlash2 uses
Q4/Q5/Q6 group-64.

## Artifact

| Field | Value |
|---|---|
| Filename | `DaisyRose-27B-FP8-v3-full.ninfer` |
| Size | 30,275,059,200 bytes (28.20 GiB) |
| SHA-256 | `27E603948280CBE601E8327D82F8667B8F0A6CE7120EC4BE2FD4EF1E1B844E38` |
| Container version | 3 |
| Artifact id | `95fed92b9ed9407ca2503924e231de7a` |
| Architecture | `Qwen3_5ForCausalLM` |
| Public model name | `daisyrose-27b-fp8` |
| Chat template | [qwen3_8.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_8.jinja) (the base model's template with NInfer's positional-system, final-assistant-continuation, and removed-tool-turn extensions); override with `--chat-template FILE` |
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
  '27E603948280CBE601E8327D82F8667B8F0A6CE7120EC4BE2FD4EF1E1B844E38' \
  'DaisyRose-27B-FP8-v3-full.ninfer' | sha256sum --check
```

The file contains Text, Vision, MTP, DFlash2 and the frontend resources. Vision weights come
from the checkpoint's own full-BF16 visual tower (the source checkpoint's
`modules_to_not_convert` keeps the visual tower, the GDN gating projections, and the MTP
norms/`fc` out of FP8). Vision and speculative weights are loaded only when selected at startup.

## Conversion and verification

The checkpoint's `quantization_config` is `quant_method: fp8`, `fmt: e4m3`,
`activation_scheme: dynamic`, `weight_block_size: [128, 128]`. NInfer's native storage is
per-row: one BF16 scale per output row. How the conversion closes that gap:

**Block-wise to per-row, exactly.** A matrix stored as `F8_E4M3` with a sibling
`weight_scale_inv` resolves to a block-wise value source: the grid is taken from
`quantization_config.weight_block_size` (or derived by divisibility), validated against the code
shape, and each code is decoded in FP32, where an E4M3 mantissa times a BF16 scale fits exactly.
The decoded values are then re-encoded with one BF16 scale per output row (`fp8_row_maxabs`).
Row ranges decode only the block rows they touch, and the decode was checked bit-exact against
an independent E4M3-table times BF16-grid oracle on real checkpoint tensors, including rows
spanning block boundaries.

**Deliberate per-operator filtering.** The `qwfp8ovr.py` override intercepts only the token
embedding, the output head, and the standard text attention/MLP projections. The GDN `a`/`b`
gating projections are not suitable for row FP8 and are kept as direct BF16; Vision, MTP, and
DFlash2 fall through to the base recipe's Q4–Q8 groupwise selection.

**Error at the format's noise floor.** Dequantized artifact objects (MLP `down_proj`, GDN
`out_proj`, full-attention `o_proj`, embedding rows, output head) match the independent source
decode within the theoretical bound `|v|·(2⁻⁴ + 2⁻⁹ + 2⁻¹⁰) + scale·2⁻¹⁰` (E4M3 half-ulp plus
BF16 source/scale rounding) on every row. The overall relative L2 is 2.6–2.9% across the
sampled families: the inherent E4M3 quantization noise, since both sides are E4M3 with
different scale granularity. The re-scaling rounding therefore reaches the mathematical limit
of the format; the GDN `a_projection` object is bit-exact BF16, and the artifact contains no
NaNs.

**Full-report verification.** `python -m tools.artifact.inspect` validates configuration,
resources, bindings, and uses of all four components (text, vision, mtp, dflash2). An
independent numerical report re-decoded the source with the safetensors library and compared
against the artifact through the v3 reader and codec dequantizers — 19 checks, all passing:
row-FP8 objects within the bound above; the `q8_g32_fp16` objects (MTP `o_proj`, DFlash2
`o_proj`) within the half-quantum bound; the direct BF16 objects (GDN `a`/`b` projections,
DFlash2 candidate codebooks) bit-exact; `proposal/token_ids` exactly equal to a recomputation
of the reference shortlist selection (131,072 rows over the tokenizer's 248,077-token public
domain, 21 forced special tokens); `proposal/head` fully dequantized and matched against
`lm_head[ids]` within the Q4 half-quantum bound; and all six resource objects byte-identical
to their declared origins.

**Hardware fit.** Every large text projection sits in the row-scaled FP8 bank, so prefill and
decode run the FP8-activation tensor-core routes directly instead of dequantizing to BF16
first, and `--kv-dtype fp8` keeps the attention matrices in the same dtype as the weights.

## Model overview

From the official [DaisyRose-27B card](https://huggingface.co/RadixArk/DaisyRose-27B); the
architecture is the Qwen3.8-27B design, unchanged by the RL post-training.

- Type: causal language model with vision encoder
- Parameters: 27.8B (incl. a 0.46B vision encoder and a 0.42B multi-token-prediction head); hidden size 5,120; 64 layers; padded vocabulary 248,320
- Layer layout: 16 × (3 × (Gated DeltaNet → FFN) + 1 × (Gated Attention → FFN))
- Gated DeltaNet: 48 value heads, 16 QK heads, head dimension 128
- Gated Attention: 24 query heads, 4 KV heads, head dimension 256, rotary dimension 64
- Feed-forward intermediate size: 17,408; MTP head trained
- Vision encoder: 27 layers, hidden 1,152, patch 16, temporal patch 2
- Context: 262,144 tokens natively

Evaluations against the base model, from the official card (vendor-run; per-benchmark
protocol notes are on the source card):

| Benchmark | DaisyRose-27B | Qwen3.8-27B |
|---|---:|---:|
| Terminal Bench 4.1 | 12.0 | 5.6 |
| Terminal Bench 2.1 | 75.5 | 73.0 |
| DeepSWE v1.1 | 46.2 | 42.2 |
| SWE Bench Pro v2 | 94.7 | 91.3 |
| HLE (text only) | 34.3 | 33.9 |
| IFBench | 82.4 | 79.5 |
| GPQA Diamond | 90.7 | 89.2 |

## Requirements

- [NInfer-all](https://github.com/iamwavecut/ninfer-all) revision
  [`f118551f`](https://github.com/iamwavecut/ninfer-all/commit/f118551f) or later, built from
  source for the target architecture: `-DCMAKE_CUDA_ARCHITECTURES=89` for the RTX 40 series,
  `120a` for the RTX 50 series and the RTX PRO 6000 Blackwell;
- Applicable GPUs: NVIDIA GeForce **RTX 4090 (48 GB)** and NVIDIA **RTX PRO 6000 (96 GB)**, and
  other cards the engine supports. The 268,288-token, four-lane configuration below is sized
  for 48 GB of VRAM; the 96 GB card runs the same line with headroom to raise
  `--max-context` and `--kv-capacity`;
- 64-bit Linux or Windows 11;
- CUDA Toolkit 13.1 or newer.

The upstream [NInfer](https://github.com/Neroued/ninfer) build targets `sm_120a` (RTX 50 series)
only, and the startup line below uses NInfer-all's `--gdn-state-fp16` and the FP8-activation
routes. NInfer does not provide an install target or packaged binary. See the
[repository build guide](https://github.com/iamwavecut/ninfer-all#building).

## Start the server

The verified startup line for the full FP8 + DFlash2 configuration (NInfer-all `ninfer-serve`):

```bash
export CUDA_MODULE_LOADING=LAZY

ninfer-serve models/DaisyRose-27B-FP8-v3-full.ninfer \
    --model-id daisyrose-27b-fp8 \
    --max-context 262144 \
    --kv-capacity 268288 \
    --prefill-chunk 32768 \
    --prefill-cublas \
    --max-concurrency 4 \
    --kv-dtype fp8 \
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
    --kv-lease-growth \
    --prefill-stall-budget-ms 500 \
    --spec dflash2 --draft-tokens 5
```

| Option | Value | Effect |
|---|---|---|
| `--model-id` | `daisyrose-27b-fp8` | public model name reported by the server (the artifact's own name) |
| `--max-context` | 262,144 | logical context ceiling of each request (the model's native context) |
| `--kv-capacity` | 268,288 | shared KV pool in tokens, the sum of the four lanes' worst-case demand (256K + 2K + 2 × 2K) resolved on the 16-token page grid; a request running alone may use the whole pool |
| `--prefill-chunk` | 32,768 | prefill chunk in tokens (a multiple of 128) |
| `--prefill-cublas` | — | hand wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost |
| `--max-concurrency` | 4 | four active request lanes |
| `--kv-dtype` | `fp8` | FP8 KV cache, the same dtype as the Text weights, so attention runs homogeneous FP8 matrix products |
| `--fast-prefill-kernel` | — | prefill the FP8 KV cache with the in-register dequant prompt-attention kernel (a small perplexity cost) |
| `--concurrent-prefill` | — | admit waiting requests to free lanes while other requests prefill |
| `--wddm-evictable-budget` | — | Windows builds: budget the KV cache against dedicated memory less the weights and a 512 MiB desktop floor; drop the flag on Linux |
| `--gdn-state-fp16` | — | keeps the GDN recurrent state in FP16 (halves each host state image) |
| `--vision` / `--vision-residency` | on / `overlay` | multimodal input; the Vision tower stays in pinned host memory and encodes inside a window borrowed from the evict-ranked weight tail, so it reserves no device memory |
| `--default-reasoning-effort` | `xhigh` | effort for requests that name none |
| `--auto-long-anchors` / `--auto-prefix-grid` | — | the engine anchors message boundaries itself, so a later request that rewrites earlier history (a compacted or edited transcript) resumes from the nearest retained anchor |
| `--derive-session-keys` | — | chat/Messages requests that name no session get a key derived from their instruction and first user message, keeping a session lineage for checkpoint retention |
| `--kv-lease-growth` | — | admission reserves the prompt plus a bounded output window and extends it at decode-round boundaries instead of the whole `max_tokens` budget |
| `--prefill-stall-budget-ms` | 500 | while other requests generate, cut each prefill step to at most ~500 ms so running streams wait about that instead of a whole chunk |
| `--spec` / `--draft-tokens` | `dflash2` / 5 | DFlash2 speculative decoding with five drafts per round (1..15 supported) |

The same line runs MTP instead of DFlash2 by replacing
`--spec dflash2 --draft-tokens 5` with `--spec mtp --draft-tokens 3`.

See the [HTTP serving guide](https://github.com/iamwavecut/ninfer-all/blob/master/README.md) for
the API surface and the [resource scheduling reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/resource-scheduling-and-context-cache.md)
for cache and admission semantics.

## Measured on an RTX 4090 (48 GB)

Single card, NInfer-all build `2026-11-23.7+2026-11-23.7-48-g84c67b8c` (nvcc 13.4.59, built-in
`rtx-4090-sm89` device profile), Windows 11 host, started with the line above
(`--spec dflash2 --draft-tokens 5`), 2026-11-24.

Startup:

| Stage | Value |
|---|---|
| Weights ready | 27.8 GiB in 5.4 s (5.17 GiB/s) |
| Activation policy | FP8-E4M3 A8 kept 513 uses, demoted to A16: 0 |
| Pinned host state / KV | 918.6 MiB / 8.00 GiB |
| CUDA Graph capture | 2.2 s |
| Engine ready, total | 12.5 s |
| KV pool | 268,288 tokens, FP8 |
| VRAM after startup | 27.5 GiB weights + 19.8 GiB runtime, 2.13 GiB free of 48 GB |

Production serving session, one user (OpenAI Chat Completions and Anthropic Messages, thinking
`xhigh`, 32-tool agent loop):

| Workload | Result |
|---|---|
| Long thinking answer (154-token prompt, stop token, 37,078 committed tokens) | TTFT 414 ms; wall time 5 m 29.4 s; committed decode 112.8 tok/s; speculation accepted 27,732/50,367 (55.1%), ngram 1,947/5,467 over 365 rounds |
| Same conversation, next turn | prefix cache 96.8% (turn closure); 30,103 committed tokens at 116.6 tok/s; speculation accepted 56.0% |
| Fresh 102,279-token agent prompt (179 messages, tools) | TTFT 35.1 s (chunked prefill at 32,768 tokens); 224 output tokens, 1 tool call |
| Subsequent agent turns (history grown to 125K tokens) | TTFT 172 ms–3.1 s with 89–100% prefix-cache hits; committed decode 86–110 tok/s; one turn finished in 896 ms end to end |
| Steady-state decode, 5-second windows over the session (n = 185) | 65.4–192.0 tok/s, median 100.4, mean 106.2 |

The rates count committed decode tokens, net of rejected drafts, for one active request; ngram
drafting runs alongside `--spec` by default in this build, so the acceptance figures mix both
drafters. This is a production serving session, not a saturation or benchmark run.

## One-shot CLI

```bash
ninfer models/DaisyRose-27B-FP8-v3-full.ninfer \
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
  48 GB configuration caps the shared pool at 268,288 tokens.
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | `RadixArk/DaisyRose-27B-FP8` (official block-wise FP8 checkpoint) |
| Base model repositories | `RadixArk/DaisyRose-27B` (RL-only post-training); `Qwen/Qwen3.8-27B` |
| DFlash2 source repository | `z-lab/Qwen3.8-27B-DFlash2` |
| Conversion recipe | `qwen3_8_27b` with the `qwfp8ovr.py` override |
| Chat template resource | `qwen3_8.jinja` from the NInfer repository (the base model's template with NInfer's extensions) |
| Converter | NInfer v3 converter, `https://github.com/Neroued/ninfer` |
| Converted | 2026-11-24 on a 20-core CPU in ~32 minutes (1,899.5 s), Python 3.11, `--rows-per-chunk 1024` |
| Container / artifact id | v3 / `95fed92b9ed9407ca2503924e231de7a` |

The full object inventory and conversion provenance are recorded in the `.conversion.json`
report produced alongside the artifact. The exact storage contract is maintained in the
[v3 container reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/artifact-container.md).

## License

This NInfer artifact is distributed under the Apache License 2.0, matching the source
[RadixArk/DaisyRose-27B-FP8](https://huggingface.co/RadixArk/DaisyRose-27B-FP8) package. The base
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.
