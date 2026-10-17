---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - nerkyor/Qwen3.8-27B-FP8-EfficientThink-Uncensored
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ninfer
  - qwen3.8
  - fp8
  - e4m3
  - efficient-thinking
  - reasoning
  - speculative-decoding
  - dflash2
  - mtp
  - multimodal
  - conversational
  - cuda
  - rtx-4090
  - rtx-pro-6000
---

# Qwen3.8-27B EfficientThink-Uncensored FP8 for NInfer

This repository contains `Qwen3.8-27B-FP8-EfficientThink-Uncensored-v3-full.ninfer`, the full NInfer
v3 artifact for the
[nerkyor/Qwen3.8-27B-FP8-EfficientThink-Uncensored](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2)
post-trained model, built from its static FP8 Block128 checkpoint. The artifact is intended only for
NInfer; it is not a Transformers checkpoint, Safetensors distribution, or GGUF file. It is the same
artifact family as the NInfer conversion of the official Qwen3.8-27B-FP8 checkpoint, built with the
identical recipe, override, and verification scope.

The model behind it is post-trained with several rounds of SFT and RLOO on top of
[Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2),
which is itself the official [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) post-trained
with SFT and SimPO. It is built to fix the original model's habit of failing to stop: it already has
an answer, yet keeps re-deriving until the generation cap with no final answer. Per the source card
(full suites, 100K context, static FP8 tier): GPQA **178/198**, MMLU **450/500**, LCB **90/100**,
vs. 177 / 444 / 83 for the original Qwen3.8-27B (FP8), with 94K-cap truncations dropping 4 → 1 on
GPQA and 13 → 3 on LCB under the same protocol.

The Text tower keeps the source checkpoint's E4M3 quantization. The checkpoint stores the linear
projections as E4M3 codes with a `weight_scale_inv` grid per 128×128 weight block; the converter
decodes each block exactly and re-encodes the values with one BF16 scale per output row
(`fp8_row_maxabs`). The token embedding, output head, and every text projection except the GDN
`a`/`b` gating projections (kept direct BF16) use that row-scaled FP8, which the engine runs on
its FP8-activation routes. MTP projections use Q8 group-32, the Vision linear layers use
Q4/Q5/Q6 group-64 with the merger in Q8 group-32, and DFlash2 projections use Q8 group-32 (the
bundled draft is itself a block-wise FP8 checkpoint, decoded to values and re-quantized by the
recipe). The bundle is fully multimodal: text, image, and video, with the package's BF16 MTP head
and block-wise FP8 DFlash2 draft built in.

## Artifact

| Field | Value |
|---|---|
| Filename | `Qwen3.8-27B-FP8-EfficientThink-Uncensored-v3-full.ninfer` |
| Size | 30,274,702,592 bytes (28.20 GiB) |
| SHA-256 | `01090912172444632f1d5414dc73e010a943e0fa19a9c231da64d6c46ec4f0b7` |
| Container version | 3 |
| Artifact id | `3a9521595dca4c5ca070228a73f900ba` |
| Architecture | `Qwen3_5ForCausalLM` |
| Public model name | `qwen3.8-27b-fp8-efficientthink-uncensored` |
| Chat template | [qwen3_8.jinja](https://github.com/Neroued/ninfer/blob/98dada0e03cb073fd07f905400b5904bc6e82759/tools/chat_templates/qwen3_8.jinja); override with `--chat-template FILE` |
| Template defaults | thinking on; effort `xhigh`; closed-turn reasoning retained |
| Stored objects | 1,126 (1,120 tensors and 6 resources) |
| Row-scaled FP8 E4M3 tensors | 258 (embedding, output head, text projections) |
| Q8 group-32 (FP16) tensors | 28 (MTP, DFlash2 projections, Vision merger) |
| Q4/Q5/Q6 group-64 (FP16) tensors | 55 / 54 / 1 (Vision linear layers; the 55th Q4 is the proposal head) |
| BF16 tensors | 627 (Vision tower direct, text norms, GDN `a`/`b` gating + convolution, MTP norms, DFlash2 direct) |
| FP32 / INT32 tensors | 96 / 1 |

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  '01090912172444632f1d5414dc73e010a943e0fa19a9c231da64d6c46ec4f0b7' \
  'Qwen3.8-27B-FP8-EfficientThink-Uncensored-v3-full.ninfer' | sha256sum --check
```

The file contains Text, Vision, MTP, DFlash2, and the frontend resources. Vision weights come from
the package's full-BF16 visual tower; the MTP head is the package's BF16 head; the DFlash2 draft
is the package's block-wise FP8 draft. Vision and speculative weights are loaded only when selected
at startup.

## Conversion and verification

The checkpoint quantizes the text linear projections with E4M3 codes plus a `weight_scale_inv`
grid per 128×128 weight block. NInfer's native storage is per-row: one BF16 scale per output row.
How the conversion closes that gap:

**Block-wise to per-row, exactly.** The built-in compressed-tensors reader resolves a matrix
stored as `F8_E4M3` with a sibling `weight_scale_inv` to a block-wise value source: the grid is
taken from `quantization_config.weight_block_size`, validated against the code shape, and each
code is decoded in FP32, where an E4M3 mantissa times the block scale fits exactly. The decoded
values are then re-encoded with one BF16 scale per output row (`fp8_row_maxabs`). Row ranges
decode only the block rows they touch.

**Deliberate per-operator filtering.** The `qwfp8ovr.py` override intercepts only the token
embedding, the output head, and the standard text attention/MLP projections. The GDN `a`/`b`
gating projections are not suitable for row FP8 and are kept as direct BF16; Vision, MTP, and
DFlash2 fall through to the base recipe's grouped-int selections.

**Canonical tokenizer resources.** The package's `tokenizer.json` is a re-serialization of the
standard Qwen3.8 tokenizer — the 248,044-entry vocabulary, all 247,587 merges in the same order,
and the 33 added tokens are identical to the official Qwen3.8-27B checkpoint — but written by an
older tool: its Split pre-tokenizer regex omits the `\p{M}` mark class, and its
transformers-5.17 `tokenizer_config.json` carries a null `added_tokens_decoder` and no
`add_bos_token`. The engine's Qwen tokenizer validation rejects both, so the artifact stores the
canonical tokenizer pair from the official Qwen3.8-27B checkpoint instead. Tokenization itself is
unchanged: vocabulary, merges, and added tokens are identical between the two serializations.

**Error at the format's noise floor.** Every element of all 258 row-scaled FP8 objects
(4,396,032 rows) was compared against an independent decode of the checkpoint's block-wise
storage and stayed within the strict theoretical bound of its stored row scale — the E4M3
half-ulp at the element's own magnitude plus the source value's own BF16 rounding — with no
element reaching even the doubled bound. Per-object relative L2 is 2.05–2.66% (mean 2.48%),
the inherent E4M3 noise floor since both sides are E4M3 at different scale granularity. A
512-row band of every one of the 258 FP8 objects re-encodes bit-exact (codes and row scales)
under the converter's exact arithmetic, the 723 direct BF16/FP32 objects are bit-exact against
the source, and the 1,120 tensor objects hold no NaN or Inf.

**Hardware fit.** Every large text projection sits in the row-scaled FP8 bank, so prefill and
decode run the FP8-activation tensor-core routes directly instead of dequantizing to BF16
first, and `--kv-dtype fp8` keeps the attention matrices in the same dtype as the weights.

`python -m tools.artifact.inspect <artifact> --objects --bindings` validates configuration,
resources, bindings, and uses of all four components (text, vision, mtp, dflash2).

## Model overview

From the [source card](https://huggingface.co/yyrgzn/Qwen3.8-27B-FP8-EfficientThink-Uncensored-NInfer-v3);
full training details and per-tier benchmarks are published there.

- A model post-trained with several rounds of SFT and RLOO on top of
  [Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2),
  which is itself the official [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) post-trained
  with SFT and SimPO. Built to fix the original model's habit of failing to stop: it already has an
  answer, yet keeps re-deriving until the generation cap with no final answer.
- GPQA **178/198**, MMLU **450/500**, LCB **90/100** (static FP8, full suites, 100K context), vs.
  177 / 444 / 83 for the original Qwen3.8-27B (FP8);
- 94K-cap truncations drop 4 → 1 on GPQA and 13 → 3 on LCB under the same protocol;
- packaged speculative decoding on the source card (SGLang): about 180 tok/s per request with
  DFlash2 and 91 tok/s with MTP, vs. about 46 tok/s without speculation;
- architecture unchanged from the official 27B: causal language model with vision encoder; 27B
  parameters; hidden size 5,120; 64 layers; padded vocabulary 248,320; 16 × (3 × (Gated DeltaNet →
  FFN) + 1 × (Gated Attention → FFN)); Gated DeltaNet with 48 value heads, 16 QK heads, head
  dimension 128; Gated Attention with 24 query heads, 4 KV heads, head dimension 256, rotary
  dimension 64; FFN intermediate 17,408; MTP head trained; 27-layer vision encoder (hidden 1,152,
  patch 16, temporal patch 2); 262,144-token native context, extensible to 1,000,000 with RoPE
  scaling (YaRN).

## Requirements

- [NInfer-all](https://github.com/iamwavecut/ninfer-all) revision
  [`f118551f`](https://github.com/iamwavecut/ninfer-all/commit/f118551f) or later, built from
  source for the target architecture: `-DCMAKE_CUDA_ARCHITECTURES=89` for the RTX 40 series,
  `120a` for the RTX 50 series and the RTX PRO 6000 Blackwell;
- Applicable GPUs: NVIDIA GeForce **RTX 4090 (48 GB)** and NVIDIA **RTX PRO 6000 (96 GB)**, and
  other cards the engine supports. The verified startup line below (262,144-token pool, two
  lanes) is sized for 48 GB of VRAM; the 96 GB card runs the same line with headroom to raise
  `--max-context` and `--kv-capacity`;
- 64-bit Linux;
- CUDA Toolkit 13.1 or newer.

The upstream [NInfer](https://github.com/Neroued/ninfer) build targets `sm_120a` (RTX 50 series)
only, and the startup line below uses NInfer-all's `--gdn-state-fp16` and `--fast-prefill-kernel`.
NInfer does not provide an install target or packaged binary. See the
[repository build guide](https://github.com/iamwavecut/ninfer-all#building).

## Start the server

The verified startup line for the full FP8 + DFlash2 configuration (NInfer-all `ninfer-serve`):

```bash
export CUDA_MODULE_LOADING=LAZY

ninfer-serve models/Qwen3.8-27B-FP8-EfficientThink-Uncensored-v3-full.ninfer \
    --model-id qwen3.8-27b-fp8-efficientthink-uncensored \
    --max-context 262144 \
    --kv-capacity 262144 \
    --prefill-chunk 32768 \
    --prefill-cublas \
    --fast-prefill-kernel \
    --max-concurrency 2 \
    --kv-dtype int8 \
    --gdn-state-fp16 \
    --vision --vision-residency overlay \
    --default-reasoning-effort xhigh \
    --host "127.0.0.1" \
    --port 10111 \
    --spec dflash2 --draft-tokens 5
```

| Option | Value | Effect |
|---|---|---|
| `--model-id` | `qwen3.8-27b-fp8-efficientthink-uncensored` | public model name reported by the server (independent of the artifact's file name) |
| `--max-context` | 262,144 | per-request logical context ceiling (the model's native window) |
| `--kv-capacity` | 262,144 | shared KV pool in tokens; a request running alone may use the whole pool |
| `--prefill-chunk` | 32,768 | prefill chunk size in tokens (a multiple of 128); `--prefill-cublas` benefits from the larger chunk |
| `--prefill-cublas` | - | routes wide prefill GEMMs to cuBLAS: a large prefill speedup for a small perplexity cost |
| `--fast-prefill-kernel` | - | the fast prompt-attention kernel (FP16 PV per 64-key tile), available only for int8 or rk* KV caches; also rounds `--prefill-chunk` down to whole attention waves |
| `--max-concurrency` | 2 | two active request lanes |
| `--kv-dtype` | `int8` | int8 KV cache, required here because `--fast-prefill-kernel`'s fast prompt-attention kernel only applies to int8 or rk* KV caches — with `--kv-dtype fp8` the flag is silently ignored and prefill runs the slower standard FP8 prompt path; downgrade to `rk8v4` if VRAM is tight |
| `--gdn-state-fp16` | - | keeps the GDN recurrent state in FP16 (halves the size of each host-side state image) |
| `--vision` / `--vision-residency overlay` | - | accept image and video input; overlay keeps the vision tower in host memory and borrows device memory per image |
| `--default-reasoning-effort` | `xhigh` | reasoning effort default for requests that set none (the template default) |
| `--spec` / `--draft-tokens` | `dflash2` / 5 | DFlash2 speculative decoding with five drafts per round (1–15 supported) |

Variants: the same line runs MTP instead of DFlash2 by replacing
`--spec dflash2 --draft-tokens 5` with `--spec mtp --draft-tokens 3`. A 278,000-token,
four-lane configuration also fits 48 GB: set `--max-context`/`--kv-capacity` to 278000,
`--prefill-chunk` to 2048, `--max-concurrency` to 4, and run `--kv-dtype fp8 --mlp-a8-decode`
instead of `--kv-dtype int8 --fast-prefill-kernel --prefill-cublas`. That FP8-KV variant keeps
the attention matrices in the same dtype as the Text weights, but on this card it loses the fast
prompt-attention kernel (`--fast-prefill-kernel` only applies to int8 or rk* KV caches), which is
why the measured line above stays on int8.

See the [HTTP serving guide](https://github.com/iamwavecut/ninfer-all/blob/master/README.md) for
the API surface and the [resource scheduling reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/resource-scheduling-and-context-cache.md)
for cache and admission semantics.

## Measured on an RTX 4090 (48 GB)

Two runs of the same number-theory conversation on 2026-10-14: one under WSL2 with NInfer-all
build `v0.6.0-rtx3090-1553-g237bf65b` and one on the native Windows 11 host with build
`v0.6.0-rtx3090-1700-g11327e96` and the disk-KV tier. Both are single-card (built-in
`rtx-4090-sm89` device profile) and use the startup line above (INT8 KV, 262,144-token context,
32,768-token prefill chunk, two lanes, thinking at `xhigh` effort; the Windows run adds the
disk-KV tier).

### WSL2 (build `v0.6.0-rtx3090-1553-g237bf65b`, no disk-KV tier)

Startup:

| Stage | Value |
|---|---|
| Weights materialized | 27.2 GiB in 5.5 s (4.97 GiB/s) |
| Pinned host state / KV | 918.6 MiB / 8.00 GiB |
| CUDA Graph capture | 2.3 s |
| Engine ready, total | 15.0 s (warmup 518 ms) |
| KV pool | 262,144 tokens, INT8, pages 4,096/8,192 |
| VRAM after startup | 27.2 GiB weights + 17.4 GiB runtime, 2.14 GiB free of 48 GB |

Load: a ten-turn number-theory conversation — the same task run against the official FP8
artifact — over the OpenAI Chat Completions stream, thinking mode at `xhigh` effort, four
tools, prompts growing from 1,605 to 23,202 tokens, 23,868 committed output tokens in total.
The turn-1 answer was checked correct.

Turn 1 (1,605-token prompt, 15,970 committed output tokens, stopped on a tool call):

| Metric | Value |
|---|---|
| Time to first token | 7.3 s (cold prefill 221.4 tok/s) |
| Wall time | 2 m 9.3 s |
| Committed decode | 131.1 tok/s |
| Steady-state 5-second decode windows | 80.7–212.8 tok/s |
| Speculation accepted | 12,581 / 19,575 (64.3%); ngram 1,677 / 3,945, 263 rounds |

Session, turns 2–10: committed decode on the substantive turns (400+ output tokens) is
104.2–127.3 tok/s; turn 7 peaks at 306.3 tok/s with ngram drafting accepting 336 of 360.
Mixed speculation acceptance over the whole session is 18,744 / 29,799 (62.9%) — DFlash2
15,564 / 23,520 (66.2%), ngram 3,180 / 6,279 (50.6%) — with per-turn values from 50.3% to
89.1%. Turns 3–10 reused 95.7–99.8% of each prompt from the private endpoint cache and opened
in 155–527 ms, the fresh part (under 800 tokens) prefilled at 87.8–1.60k tok/s.

Two caveats on the WSL2 run. The first two turns prefilled cold at 221.4 and 246.0 tok/s
(turn 2: 17,619 tokens in 71.6 s with the host idle throughout the window), roughly ten times
slower than the warm prefill rate above and than the identical line on the Windows host
below, so the slowdown is WSL2-specific rather than an artifact or build effect. And the
disk-KV tier (`--disk-kv-path` plus 128 GiB plus restore) crashed the WSL2 VM during startup
on this host, so the WSL2 run omits the disk tier; the FP8-KV variant noted above is not the
one measured in either run.

### Windows 11 host (build `v0.6.0-rtx3090-1700-g11327e96`, disk-KV tier)

Startup:

| Stage | Value |
|---|---|
| Weights materialized | 27.5 GiB in 5.4 s (5.14 GiB/s) |
| Pinned host state / KV | 918.6 MiB / 812.6 MiB, the disk tier holding the rest |
| CUDA Graph capture | 1.6 s |
| Engine ready, total | 10.0 s (warmup 271 ms) |
| KV pool | 262,144 tokens, INT8, pages 4,096/8,192 |
| VRAM after startup | 27.5 GiB weights + 17.4 GiB runtime, 2.47 GiB free of 48 GB |

The captured log holds the conversation's first three requests. The run's first request
already hit 1,600 of 1,605 prompt tokens (99.7%) — a fresh engine can only serve that prefix
from `--disk-kv-restore`, so the restore path was exercised — and with the disk tier the
pinned host KV drops from 8.00 GiB (WSL2 run) to 812.6 MiB. The long turn (3,320-token
prompt, 75.9% cached, 14,513 committed output tokens, stopped on the stop token):

| Metric | Value |
|---|---|
| Time to first token | 489 ms (fresh 800 tokens at 1.69k tok/s) |
| Wall time | 1 m 41.2 s |
| Committed decode | 144.3 tok/s |
| Steady-state 5-second decode windows | 86.8–221.4 tok/s |
| Speculation accepted | 11,600 / 17,215 (67.4%); ngram 1,821 / 3,975, 265 rounds |

Prefill is normal on this host — requests 2 and 3, whose fresh parts prefilled at 1.33k and
1.69k tok/s, hit the rest from the disk tier — which isolates the WSL2 cold-prefill slowdown
above to the WSL2 host.

On both hosts the committed decode (131.1 / 144.3 tok/s) and mixed acceptance (64.3% /
67.4%) match the official FP8 artifact's numbers on this card (132.0 tok/s, 62.8%) within
run-to-run noise, although the two ETU runs use different builds. The rates count committed
decode tokens, net of rejected drafts, for one active request; ngram drafting runs alongside
`--spec` by default in this build, so the acceptance figures mix both drafters.

## One-shot CLI

```bash
ninfer models/Qwen3.8-27B-FP8-EfficientThink-Uncensored-v3-full.ninfer \
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
- Context allocation is subject to GPU memory and the selected KV-cache type; the verified 48 GB
  configuration caps the shared pool at 262,144 tokens (a 278,000-token variant is noted above).
- NInfer does not execute generated tool calls.

## Provenance

| Field | Value |
|---|---|
| Source repository | [yyrgzn/Qwen3.8-27B-FP8-EfficientThink-Uncensored](https://huggingface.co/yyrgzn/Qwen3.8-27B-FP8-EfficientThink-Uncensored-NInfer-v3) (static FP8 Block128 tier) |
| Training base | [nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2](https://huggingface.co/nerkyor/Qwen3.8-27B-EfficientThink-Uncensored-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2) (SFT + SimPO post-training of the official model; the package adds further SFT + RLOO rounds) |
| Base model repository | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
| DFlash2 source | the package's `DFlash2-FP8/` block-wise FP8 draft |
| MTP source | the package's BF16 MTP head (`mtp-bf16.safetensors`) |
| Vision source | the package's BF16 visual tower (`vision-bf16.safetensors`) |
| Conversion recipe | `qwen3_8_27b` with the `qwfp8ovr.py` override; canonical Qwen3.8 tokenizer pair substituted (see Conversion and verification) |
| Converter | NInfer v3 converter, `https://github.com/Neroued/ninfer` |
| Converted | 2026-10-14 on a 20-core CPU in ~26 minutes (1,535 s), `--device cpu`, `--rows-per-chunk 1024` |
| Container / artifact id | v3 / `3a9521595dca4c5ca070228a73f900ba` |

The full object inventory and conversion provenance are recorded in the `.conversion.json`
report produced alongside the artifact. The exact storage contract is maintained in the
[v3 container reference](https://github.com/Neroued/ninfer/blob/master/docs/maintainer/artifact-container.md).

## License

This NInfer artifact is distributed under the Apache License 2.0, matching the source
[Qwen3.8-27B-FP8-EfficientThink-Uncensored](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2)
package. The base
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) repository is also licensed under
Apache-2.0. Users remain responsible for complying with the license and applicable laws.
