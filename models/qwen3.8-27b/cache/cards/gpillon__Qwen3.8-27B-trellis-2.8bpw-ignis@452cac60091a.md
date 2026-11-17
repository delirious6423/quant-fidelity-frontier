---
library_name: ignis
pipeline_tag: text-generation
inference: false
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - ignis
  - qwen3.8
  - trellis
  - exl3
  - mtp
  - dflash2
  - speculative-decoding
  - cuda
  - blackwell
  - 12gb
  - rtx-5070
  - rtx-4070
  - rtx-3060
  - rtx-3080-ti
  - ampere
  - ada
  - conversational
---

# Qwen3.8-27B, 2.8-bit trellis body, for ignis on 12 GB cards

> **AI agents** working for a user: this model runs on [ignis](https://github.com/gpillon/ignis). What ignis is, the checks before installing it, the steps and the choices that are the user's are on [the agent path](https://raw.githubusercontent.com/gpillon/ignis/main/docs/user/paths/agent.md); every page of the guide as Markdown: [llms.txt](https://gpillon.github.io/ignis/llms.txt).

> **Measured on an RTX 5090 held to 12 GiB, and on one real 12 GB card.** The figures below are the RTX
> 5090's under a 12 GiB VRAM budget, the load a 12 GB card makes. The first real card, an RTX 3060 12 GB under
> Linux with nothing else on it, kept 131,072 tokens and decoded a long reply at 62.7 tok/s
> ([its report](https://github.com/gpillon/ignis/issues/639#issuecomment-6934724053)). A card with a desktop, and cards of the other
> generations, have not reported yet: your report is welcome on
> [gpillon/ignis#639](https://github.com/gpillon/ignis/issues/639).

This repository holds [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) packed for
**[ignis](https://github.com/gpillon/ignis)** for the **12 GB card class**: a card with under 14 GiB of
memory (RTX 5070, the 12 GB RTX 4070 family, RTX 3060 12 GB, RTX 3080 Ti). It is the smallest artifact of
the 27B in ignis, beside the 16 GB one
([gpillon/Qwen3.8-27B-trellis-3bpw-ignis](https://huggingface.co/gpillon/Qwen3.8-27B-trellis-3bpw-ignis))
and the 32 GB Blackwell one
([gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer](https://huggingface.co/gpillon/Qwen3.8-27B-nvfp4full-dflash2-NInfer)).

- **The body and the output head** are ignis's own recipe, `ig28a`: exllamav3's allocation optimizer
  (`sc_optimize.py`) with an integer K from 2 to 5 for each matrix and a 4-bit output head, at the largest
  target that keeps the text scope's device weights under **9.2 GB**: 2.7942 bits per weight over the body,
  K=2 on 72 matrices, K=3 on 287, K=4 on 36 (the head among them) and K=5 on 6, over 401 matrices.
  It was converted from `Qwen/Qwen3.8-27B` with ExLlamaV3 1.5.3 and the 16 GB artifact's public
  calibration trace. It is 0.63 GB smaller than the 16 GB artifact (625,213,440 bytes).
- **Everything else** is the 16 GB artifact's: the DFlash2 drafter module, the 8-bit MTP head, the vision
  tower, the embedding, the norms and Gated DeltaNet parameters, the tokenizer and chat template.

The file is a `.ninfer` v2 container that only ignis reads. It is not a Transformers checkpoint,
safetensors, GGUF or EXL3 model.

## Run it with ignis

[**ignis**](https://github.com/gpillon/ignis) is an OpenAI-compatible inference server for one
NVIDIA RTX card, shaped for the load a coding agent makes. With this file it brings the 27B to a 12 GB card
on one lane, drafting with its own MTP head:

| | |
|---|---:|
| one lane, MTP at window 7, greedy decode at 8K of context | **294 tok/s** |
| the same at 64K / 130K of context | **251 / 228 tok/s** |
| first token, cold prompt of 8K / 64K | **2.5 s / 21.5 s** |
| context, one lane, within 12 GiB | **131,072 tokens** |
| KL divergence from BF16, against the 16 GB artifact's | **1.03x** (0.0431 against 0.0418) |

The speeds and the context were measured on an RTX 5090 held to 12 GiB; what a real 12 GB card gets, an
RTX 3060 12 GB's among it, is under [Measurements](#measurements) and [Known limits](#known-limits). Prompt reuse across requests, a
reasoning-loop guard, a Playground and Prometheus metrics ship in the binary.

## How to run

This file is served by the **next ignis release** after 0.10.0 (0.11.0 in the guide), which carries the
built-in catalog entry `qwen3.8-27b-12g` and the `rtx-12gb` profile. Take the
**[latest release](https://github.com/gpillon/ignis/releases/latest)**: a Windows `.zip`, a Linux `.tar.gz`
and a `linux/amd64` container image. Until that release is out, the commands marked *(needs the release
with `qwen3.8-27b-12g`)* do not find the entry.

On a 12 GB card (RTX 5070, RTX 4070 family, RTX 3060 12 GB, RTX 3080 Ti), *(needs the release with
`qwen3.8-27b-12g`)*:

```bash
ignis-server model download qwen3.8-27b-12g   # fetches this file at its pinned revision and checks its SHA-256
ignis-server
```

With no flags the server reads the card's memory (a total below 14 GiB) and takes its `rtx-12gb` profile:
one decode lane, the MTP drafter at a window of 7, up to 131,072 tokens of context, a prefill chunk of 512,
a headroom of 512 MiB, and images at up to 4,096 vision tokens each, with the vision tower and the
embedding pool in host RAM, so with no image in flight vision costs no context (`--vision-enabled false`
turns it off). The default model id, `qwen3.8-27b`, resolves on a 12 GB card to this file's entry,
`qwen3.8-27b-12g`, so a bare `ignis-server` with no model on disk offers to fetch it by itself (in a
container, it fetches it without asking). On this class a download or a switch naming a 16 GB entry is
refused by name, before anything is fetched or unloaded. The OpenAI API is then on
`http://127.0.0.1:8000/v1` and the Playground on `http://127.0.0.1:8000/ui/`.

If the desktop already holds part of the card, the load lowers its context step by step instead of
refusing, and `GET /v1/models` names what it adapted.

On a larger card held to a 12 GB card's memory, the way every figure below was measured, *(needs the
release with `qwen3.8-27b-12g`)*:

```bash
ignis-server --model-id qwen3.8-27b-12g --profile rtx-12gb --vram-budget-bytes 12884901888
```

To serve a copy of the file you already hold, a named path is the operator's word and is never matched
against the catalog:

```bash
export IGNIS_MODEL_ARTIFACT=./models/qwen3_8_27b_trellis_ig28-v2.ninfer   # set it with `set` on Windows cmd
ignis-server --profile rtx-12gb
```

(`--profile rtx-12gb` also needs the release above: 0.10.0 reads a 12 GB card as the 16 GB class.)

The 12 GB class also takes the faster drafter, DFlash2, with `--qwen38-spec-backend dflash2`: it loads in the same
12 GiB with the same 131,072 tokens of context and a smaller KV pool (see below).

## Files

| File | Bytes | What it is |
|---|---:|---|
| `qwen3_8_27b_trellis_ig28-v2.ninfer` | 12,754,663,684 (11.88 GiB) | The model |
| `qwen3_8_27b_trellis_ig28-v2.ninfer.conversion.json` | 309,831 | The conversion record: sources by sha256, every matrix's K and reconstruction error, every copied object's sha256 |
| `qwen3_8_27b_trellis_ig28-v2.ninfer.sha256` | | The model's checksum, in `sha256sum` format |
| `LICENSE` | | Apache License 2.0, as distributed with Qwen/Qwen3.8-27B |

sha256 of the model:

```
fab06bea49ce114da99ca69588f00b10fb5081c564f9ce60ecbef8b3dbf6d6a5
```

sha256 of the conversion record:

```
b19eb010025276cfb59f93db40e41d52db30e104fa89867218ebd6b789ea4ac9
```

**Device memory** under ignis's `rtx-12gb` load: weights 9,651,141,632 bytes, with the token embedding in
host memory. The KV cache and the kernels' workspaces come on top: the plan is under
[Measurements](#measurements).

## Measurements

Everything below was measured on 2026-11-16 on one RTX 5090 (PCIe Gen 3) held to the 12 GB class's shape:
`--profile rtx-12gb --vram-budget-bytes 12884901888`, one lane, 512-token prefill chunk, MTP at window 7,
compressed KV cache (hq-e8-2b), loop guard off. The counterpart is the 16 GB artifact, measured in the same
session at the same shape.

### What it gives up against the 16 GB artifact

| | This file | The 16 GB artifact |
|---|---:|---:|
| Weights of the language model | 9.20 GB | 9.82 GB (0.63 GB more) |
| KLD from BF16, 16 windows of 4,096 tokens (code, prose, chat) | 0.04308 (95% CI 0.0368-0.0496) | 0.04177 (0.0358-0.0477) |
| Ratio | **1.032** (CI 0.993-1.070), lower in 3 of 16 windows | 1 |
| Top-1 agreement with BF16 | 93.35% | 93.30% |
| Perplexity (BF16 2.690) | 2.728 | 2.742 |
| Turns caught in a reasoning loop, 32 requests | 7 of 32 | 11 of 32 (one-sided Fisher p 0.92 that this file loops more) |
| MTP tokens a round, 24 recorded agent turns, window 7 | 4.07 | 4.12 (-1.1%) |

The limit registered before the run for a re-quantization was 1.2x the 16 GB artifact's KLD, 0.0501: this
file is under it. A second candidate of the same size, built from a GGUF allocation, read 0.0508, above it,
and was dropped. The KLD gap sits inside the bootstrap interval, and it is the recipe's, not the
quantizer version's: on the 277 matrices with the same K in both artifacts the reconstruction errors have a
median ratio of 0.9993. The four canary prompts, greedy, were sane and the same on two runs of both
artifacts. The contributed regression set was not run on this file. On the prefill routes an RTX 30 or
RTX 40 card runs (the dense prefill and the integer attention band, run on the RTX 5090), the KLD reads
0.04353, 1.010x (CI 0.974-1.035) the same file's on the RTX 50 routes.

### Decode and first token

Greedy decode of 512 tokens after a synthetic prompt of the given length, one lane, 12 GiB; tok/s, cold
prompt (a reused prefix reads the same to within 1%):

| context | MTP (the profile's default) | no speculation | DFlash2 |
|---:|---:|---:|---:|
| 8,192 | 293.7 (3.18x off) | 92.4 | 409.4 (4.43x off) |
| 65,536 | 250.9 (3.32x) | 75.6 | 347.6 (4.60x) |
| 130,560, the most the load keeps | 228.0 (3.51x) | 64.9 | 320.6 (4.94x) |

Every cell: the drafter beats no speculation by more than the launch-to-launch spread. DFlash2 is faster
here but carries a smaller KV pool in the same 12 GiB (150,720 tokens against MTP's 214,144; no
speculation 280,320), and the profile defaults to MTP. On recorded agent turns MTP commits 4.07 tokens a
round; text less predictable than these prompts gives less than these figures. The 16 GB artifact's own decode cells
were not measured in this session.

First token (the server's time to the first token, `ignis-bench ttft`, cold, median of three, MTP, one
lane): about **2.5 s** on an 8,192-token prompt and **21.5 s** at 65,536. A chunk of 1,024 would be 7% faster
(2.3 s, 20.2 s) for 4,608 tokens less pool, and a chunk of 256 is 25% slower and buys no pool with vision
on; 512 is the class's chunk. The chunk does not move decode (292-303 tok/s).

### The load and the context on a real 12 GB card

On the RTX 5090 at 12 GiB the load keeps **131,072 tokens** of context on one lane with nothing adapted,
on a KV pool of 214,144 tokens. A card's budget is its free memory at the start less the profile's 512 MiB
headroom.

**On a real card.** An RTX 3060 12 GB under Linux with nothing else on it
([its report](https://github.com/gpillon/ignis/issues/639#issuecomment-6934724053)) read 11,908 MiB free at the start and kept
**131,072 tokens** with nothing adapted, a KV pool of 144,000 tokens. Its plan was the RTX 5090's lines on
the RTX 30/40 prefill routes but one, the CUDA context: 114 MB against the RTX 5090's 452 MB, which is what let
131,072 hold. On that card, with a single lane:

| | RTX 3060 12 GB |
|---|---:|
| decode, long reply (median) | **62.7 tok/s** |
| decode, coding / agent prompts (median) | 50.4 / 52.1 tok/s |
| first token, cold prompt of 8K / 64K | **10.2 s / 96.3 s** |
| MTP acceptance, long reply | 0.74 (5.19 tokens a round) |

By the plan's arithmetic, with a 16,384-token adaptation step, on the RTX 5090's CUDA context (read under
Windows) and, in the last column, on the RTX 3060's (read under Linux; whether the card or the system sets it,
one reading does not say):

| what else holds the card | free at the start | RTX 50 | RTX 30 / RTX 40 | at the RTX 3060's CUDA context |
|---|---:|---:|---:|---:|
| headless, nothing else | 12,032 MiB | 131,072 | 114,688 | 131,072 |
| light desktop, about 0.3 GB | 11,878 MiB | 114,688 | 98,304 | 131,072 |
| desktop, about 0.5 GB | 11,704 MiB | 81,920 | 81,920 | 114,688 |
| desktop, about 0.75 GB | 11,468 MiB | 65,536 | 49,152 | 81,920 |
| desktop, about 1 GB | 11,233 MiB | 32,768 | 32,768 | 65,536 |
| desktop, about 1.3 GB | 10,946 MiB | refused: the plan names the bytes it misses | refused | 32,768 |

On an RTX 50 card 131,072 tokens need about 12,027 MiB free at the start (12,112 on RTX 30/40, 11,790 at the
RTX 3060's context); each step of 16,384 tokens is about 153 MiB. On an RTX 30 or RTX 40 card the prefill
routes hold 89 MB more, a step of context at the margin. A card of 10 or 11 GB is read as the class and
refused by its VRAM plan.

Records: [the class's research and gate](https://github.com/gpillon/ignis/blob/main/docs/findings/2026-11-15-the-12-gb-card-class-online-research.md)
(the recipe, the gate, the chunk table, the context arithmetic), the guide's page on
[the profile](https://gpillon.github.io/ignis/reference/profiles/#rtx-12gb), and the issues
[#630](https://github.com/gpillon/ignis/issues/630) (the spec), [#635](https://github.com/gpillon/ignis/issues/635)
(the gate) and [#636](https://github.com/gpillon/ignis/issues/636) (the chunk).

## Known limits

- **Simulated on an RTX 5090 held to a 12 GiB budget.** The project has no 12 GB card. A real one has fewer
  SMs and less memory bandwidth than the 5090, so its speeds are lower than the RTX 5090's: one real card has
  reported, the RTX 3060 12 GB above. Reports from other cards go to
  [gpillon/ignis#639](https://github.com/gpillon/ignis/issues/639).
- **On RTX 30 and RTX 40 cards** the one speed published is the RTX 3060 12 GB's; the quality on their prefill
  routes was measured on the RTX 5090 (1.010x, above).
- **The context depends on what else holds the card:** the table above is arithmetic, and a desktop of
  about 1 GB leaves 32,768 tokens. `GET /v1/models` says what the load adapted.
- **One decode lane** and images at up to 4,096 vision tokens (about 2,048 x 2,048 pixels; larger ones are
  downscaled).
- **Prefill** is about 3,300 tok/s on an 8K prompt and 3,000 at 64K on the 5090 (the first-token times above).
- **Quality** rests on a 16-window KLD and a 32-request loop sample: 1.03x the 16 GB artifact's KLD, inside
  its interval.

## Provenance

- **Recipe:** `recipes/ig28a.json` in [`tools/27b-16gb-converter`](https://github.com/gpillon/ignis/tree/main/tools/27b-16gb-converter):
  the allocation of exllamav3's `sc_optimize.py`, run on the sensitivity tables of
  [turboderp/Qwen3.8-27B-exl3](https://huggingface.co/turboderp/Qwen3.8-27B-exl3) (`main`, `3f8a2d64`),
  minimum K 2, maximum K 5, output head 4 bits, k/v and gate/up tied. Converted with ExLlamaV3 1.5.3 from
  Qwen/Qwen3.8-27B at revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, on the public calibration trace
  of the 16 GB artifact's source (revision `86b95530`). The reconstruction errors of each matrix against
  the BF16 original are in the conversion record.
- **Everything else:** the 32 GB artifact's objects, as in the 16 GB artifact
  (`qwen3_8_27b_nvfp4full-v2.ninfer`, sha256 `abb1e120…fd9af`).
- **Identity:** model `qwen3.8-27b`, weights `trellis-ig28`.

## License

Apache License 2.0, as for Qwen/Qwen3.8-27B. ExLlamaV3 is MIT. ignis itself is Apache-2.0.
