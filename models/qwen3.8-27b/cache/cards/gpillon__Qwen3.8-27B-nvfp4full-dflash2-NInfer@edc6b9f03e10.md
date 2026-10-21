---
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
  - unsloth/Qwen3.8-27B-NVFP4
base_model_relation: quantized
tags:
  - ninfer
  - ignis
  - qwen3.8
  - nvfp4
  - w4a4
  - blackwell
  - multimodal
  - conversational
  - cuda
  - rtx-5090
  - dflash2
  - speculative-decoding
---

# Qwen3.8-27B fuller NVFP4 + DFlash2, for ignis and NInfer

This repository contains the `nvfp4full` weight profile of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the native
[NInfer](https://github.com/Neroued/ninfer) `.ninfer` artifact format, **with the DFlash2
speculative-decoding drafter module grafted on**. It runs on engines that read that format
([ignis](https://github.com/gpillon/ignis) and NInfer); it is not a Transformers checkpoint,
Safetensors distribution, or GGUF file.

## Run it with ignis

[**ignis**](https://github.com/gpillon/ignis) is the engine this artifact is developed and tested
with today: an OpenAI-compatible inference server for one Blackwell card (RTX 5090 or RTX PRO
6000), shaped for the load a coding agent makes, one main agent plus a few subagents on the same
card. On one RTX 5090, with this file and its DFlash2 drafter:

| load | decode |
|---|---:|
| one lane, average | **~170 tok/s** |
| one lane, peaks (predictable code) | **~370 tok/s** |
| eight lanes at once, aggregate | **1,065 tok/s** |

Context up to 512K tokens (YaRN x2), image input, prompt reuse across requests, a Playground and
Prometheus metrics in the binary. Windows and Linux builds and a container image are on the
[releases page](https://github.com/gpillon/ignis/releases); how these numbers were measured, and
against what, is in the README's
[How fast](https://github.com/gpillon/ignis#how-fast).

```bash
ignis-server model download qwen3.8-27b    # fetches this file and checks its SHA-256
ignis-server --model-max-context 524288 --model-rope-scaling yarn:2 \
  --spec-backend dflash2 --spec-draft-tokens 7 --vision-enabled
```

The OpenAI API is then on `http://127.0.0.1:8000/v1` and the Playground on
`http://127.0.0.1:8000/ui/`. The server also fetches this file by itself the first time it starts
without a model.

## This image

The artifact was first built for
**[github.com/gpillon/ninfer](https://github.com/gpillon/ninfer)**, and still runs there; see below
for what that fork is and why it exists.

It is the v2 image of
[cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer):
same `qwen3.8-27b`/`nvfp4full` identity, same 1,259 base objects byte-for-byte, plus 66 objects
appended for the DFlash2 module (34 NVFP4 weight matrices, 32 BF16 norm/conv-base tensors — a
5-layer sliding-window drafter at target hidden width, `tools/artifact/graft_dflash2_module.py`).
The module was grafted onto the existing v1 artifact rather than reconverted from scratch, so the
base tensors are bit-identical to v1.

## The companion engine — gpillon/ninfer

[github.com/gpillon/ninfer](https://github.com/gpillon/ninfer) is a fork of NInfer optimized
specifically for coding-agent workloads — the burst-of-near-identical-subagent-requests pattern
tools like Qwen Code produce, where each request shares the same long system prompt and tool
schemas (often ~98% identical) and diverges only in a short task-specific tail. The stock engine
had a single global prefill lane with no notion that these requests were mostly redundant with each
other, so every subagent redid the full prefill from scratch, one behind another — and a
long-running main conversation was indistinguishable from a one-shot classifier call to the
scheduler, so it could get evicted and re-prefilled at the worst possible moment. This fork exists
to fix that. It adds, on top of the upstream engine:

- a host-RAM KV cache tier (two-tier probation/protected eviction) that snapshots finished or
  still-decoding GPU lanes so a sibling request can restore instead of re-prefilling;
- prefix-reuse and admission logic so concurrent requests sharing a long prefix skip the redundant
  part of prefill entirely;
- tagged request lanes (`@main`/`@agents`/`@classifier`) so the scheduler protects the long-lived
  conversation from eviction by short-lived traffic;
- **DFlash2 speculative decoding made to actually run end-to-end** — this is the branch that made
  the artifact on this page loadable and usable (gather/selector addressing bugs fixed, RAM-tier KV
  carry for the drafter, working presets);
- two silent KV-corruption bugs found and fixed (a rewrite-checkpoint restore and a hyperquant
  exact-key side store, both cases of one request serving with another's state);
- streaming/tool-call-parsing hardening and adaptive MTP verification-width calibration.

Measured effect on the source fork: siblings that used to pay a full prefill now restore in tens to
a few hundred ms, and a burst of subagents sharing a system+tools prefix reuses ~99% of it instead
of queueing behind each other. Full rationale and file-level pointers:
[docs/maintainer/gpillon-fork-changes.md](https://github.com/gpillon/ninfer/blob/feat/dflash2-local/docs/maintainer/gpillon-fork-changes.md).

`gpillon/ninfer` itself builds on `cometkim/ninfer` (kernel-perf, hyperquant KV cache, 1M-context
envelope, NVFP4-full target, the original DFlash2 port), which builds on `natpate/ninfer-windows`
(the Windows port), which builds on [Neroued/ninfer](https://github.com/Neroued/ninfer) (the
original engine).

## Artifact

| Field | Value |
|---|---|
| Filename | `qwen3_8_27b_nvfp4full-v2.ninfer` |
| Size | 19,406,942,468 bytes (18.07 GiB) |
| SHA-256 | `abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af` |
| Container version | 2 (same container format as v1) |
| NInfer model ID | `qwen3.8-27b` |
| NInfer weights ID | `nvfp4full` |
| NInfer target key | `qwen3_8_27b` |
| Stored objects | 1,325 (1,259 inherited from v1 + 66 DFlash2 module) |
| NVFP4 tensors | 281 (247 base + 34 DFlash2) |
| BF16 exception tensors | 9 (unchanged from v1) |
| DFlash2 module relative Frobenius error | max 0.0959, mean 0.0950 (34 NVFP4 parents, weight-only quantization) |

Verify a downloaded file with:

```bash
printf '%s  %s\n' \
  'abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af' \
  'qwen3_8_27b_nvfp4full-v2.ninfer' | sha256sum --check
```

## Provenance

| Source | Revision | Role |
|---|---|---|
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | every locally quantized parent, BF16 exceptions, direct tensors, W8 endpoints, MTP, Vision, frontend (inherited from v1) |
| `unsloth/Qwen3.8-27B-NVFP4` | `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108` | the 112 layers 0-55 MLP NVFP4 parents and their divisors, copied bit-exactly (inherited from v1) |
| local `DFlash2DraftModel` checkpoint | — | the 66 DFlash2 module objects; a 5-layer sliding-window (2048) drafter at 5120 hidden width, `dflash_config.target_layer_ids = [5, 19, 33, 47, 61]`, `selector_rank = 256`, `selector_top_k = 16`, `block_size = 8`; not independently published on Hugging Face |

Base tensors carry over from
[cometkim/Qwen3.8-27B-nvfp4full-NInfer](https://huggingface.co/cometkim/Qwen3.8-27B-nvfp4full-NInfer);
see that repository's card for the full base-image encoder/calibration contract.

## Requirements — read before downloading

With ignis: a release from [gpillon/ignis](https://github.com/gpillon/ignis/releases) (Windows,
Linux or the container image) and an RTX 5090 or RTX PRO 6000 (`sm_120a`). Nothing below applies.

With NInfer:

- [gpillon/ninfer](https://github.com/gpillon/ninfer), default branch `feat/dflash2-local`
  (built on `cometkim/ninfer`'s `feat/dflash2`, tip `62acfe15`, 2026-08-28) — the branch that
  registers DFlash2 tensor binding for the `nvfp4full` weights profile and fixes the DFlash2
  gather/selector bugs that make it run correctly end-to-end;
- **this artifact will NOT load on any NInfer build predating that DFlash2 binding merge**,
  including the plain `feat/qwen3.8-nvfp4full` branch used for v1. The loader requires every
  object in the container to be consumed at bind time; a pre-DFlash2 build has no binding code for
  the 66 `dflash2/*` objects and fails with `artifact object was not consumed by the selected
  target: dflash2/...`. This is not a corrupt-file symptom — it means the engine build is too old
  for this artifact;
- Windows (MSVC + CUDA 13.1+, vcpkg for FFmpeg/curl) or 64-bit Linux (WSL2 validated);
- NVIDIA GeForce RTX 5090 (`sm_120a`).

On a compatible (DFlash2-aware) build, the artifact runs fine **without** `--spec dflash2` too —
the module is validated at load but not materialized on device unless the speculative backend is
selected, so it costs no extra VRAM when unused. It is exactly as usable as v1 for anyone who
doesn't care about DFlash2, provided their engine build is new enough.

## Download and run

```bash
hf download gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer qwen3_8_27b_nvfp4full-v2.ninfer \
  --local-dir models

# with DFlash2 speculative decoding
./build-win/apps/Release/ninfer.exe models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec dflash2 --draft-tokens 7

# without speculative decoding — identical to running the v1 artifact
./build-win/apps/Release/ninfer.exe models/qwen3_8_27b_nvfp4full-v2.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256
```

`--spec mtp` also still works on this artifact (the MTP module is untouched, inherited from v1).

## Supported use

Identical to `cometkim/Qwen3.8-27B-nvfp4full-NInfer` v1, plus DFlash2 speculative decoding
(`--spec dflash2 --draft-tokens 1..7`): text generation in thinking and non-thinking modes; image,
multi-image, video, and mixed multimodal messages; MTP or DFlash2 speculative decoding; BF16 and
INT8 group-64 KV cache; CUDA Graph decode and compatible-prefix reuse; bounded concurrent serving;
the NInfer CLI; and OpenAI/Anthropic-compatible HTTP serving.

## Limits

Same limits as v1 (single-sample benchmark results where quoted, not pass@k; quantization quality
gated against the official artifact on GPQA-Diamond only). The DFlash2 module's 34 NVFP4 parents
were quantized weight-only (no activation-quant site — see `graft.json` alongside this file) and
have not been independently re-benchmarked end-to-end on this repository; native-acceptance
measurements taken during development (3.4-3.7 tok/round) and the full DFlash2 fix history are in
[gpillon/ninfer](https://github.com/gpillon/ninfer)'s maintainer docs.
