---
license: apache-2.0
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: halogen
# The widget is off deliberately: these are .hgn weights, so nothing on the
# Hub can load them and a widget could only fail. The pipeline tag stays,
# because it is what puts the model in the text-generation filter.
inference: false
tags:
  - halogen
  - halogen-flash
  - qwen
  - moe
  - mixture-of-experts
  - long-context
  - quantized
  - 4-bit
  - strix-halo
  - gfx1151
  - rocm
  - amd
  - radeon
  - ryzen-ai
  - local-llm
---

<p align="center">
  <img src="halogen.jpg" alt="halogen" width="760">
</p>

# halogen-flash, Qwen3.8-Flash-Next for AMD Strix Halo

Weights for
[halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server), a
dedicated inference engine for Qwen3.8-Flash-Next on AMD Strix Halo
(gfx1151). One model family, one GPU, every kernel written for this silicon.

**These weights are in halogen's own `.hgn` format and will not load in
transformers, vLLM, or llama.cpp.** They exist to be mounted into the
halogen-flash-server container.

## What is here

```
qwen38-flash-next-w4b.hgn              115.55 GiB   checkpoint (1198 tensors)
qwen38-flash-next-w4b.overlay.hgn        2.40 GiB   quality sidecar, KEEP THIS
qwen38-flash-next-w4b.overlay-speed.hgn  2.31 GiB   optional: the speed arm
qwen38-flash-next-vision.hgn             0.84 GiB   optional: reads images
tokenizer/                                          flat tokenizer directory
```

**The checkpoint is TWO files.** The sidecar is a patch overlay: a subset of
tensors re-quantized more carefully, read in place of the base file's copies.
The engine loads `<checkpoint>.overlay.hgn` automatically when it sits beside
the checkpoint, you do not configure anything, you just keep both files. Drop
it and the model still runs, roughly 5–9% worse on perplexity, saying so in one
line of startup output.

Download both. Together they are ~118 GiB, and the model needs ~68 GiB resident
plus the memory-mapped n-gram table.

**The vision file is separate and optional.** `qwen38-flash-next-vision.hgn`
lets the server read images, and without it the server is text-only and refuses
an image with a message naming the setting that turns it on. Take it if you
want to send screenshots; skip it and nothing else changes, because with no
file the image path is absent rather than disabled and text behaviour is
byte-identical either way. Put it beside the checkpoint and start the server
with `HALOGEN_VISION_TOWER=1`.

**Plan on a machine of its own.** Loaded at the defaults the server holds most
of a 128 GB host, and what is left is mostly not in the large contiguous pieces
another big process needs. If you run application containers or another model
beside it they compete for the remainder, and when it runs out both they and
the server can stop for minutes at a time at full CPU with no output, which is
not a crash and needs no restart. The server reports what it leaves behind on
its last line of startup, and the
[server repository](https://github.com/peonist-ai/halogen-flash-server) has the
settings that give memory back.

## Use

```bash
hf download peonist-ai/halogen-qwen3.8-flash-next --local-dir ~/halogen-models

podman run --rm -p 8731:8731 \
  --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --ipc=host --ulimit memlock=-1:-1 \
  -v ~/halogen-models:/models:ro \
  ghcr.io/peonist-ai/halogen-flash-server:0.6.3
```

This repo carries the tokenizer, so one `-v` is all you need. On Docker rather
than Podman, replace `--group-add keep-groups` with
`--group-add video --group-add render`: `keep-groups` is a Podman keyword that
Docker resolves against the container's `/etc/group` and fails to find.

An OpenAI-compatible endpoint comes up on `:8731`: `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, and `/v1/responses`. The last of those is the
Responses API, so the **OpenAI Codex CLI** works against this server directly,
tool calls included, by pointing a `model_providers` entry at it with
`wire_api = "responses"`.

Full documentation is in the
[server repository](https://github.com/peonist-ai/halogen-flash-server).

## Precision

4-bit weights are a **correctness precondition, not an optimization**:
Qwen3.8-Flash-Next is 125B total parameters plus a 51B n-gram embedding table,
and neither BF16 (335 GiB) nor FP8 (173 GiB) fits in Strix Halo's 124 GB of
unified memory. The question was never whether to quantize but where to spend
the bits.

| | |
|---|---|
| base file | 115.55 GiB, 1198 tensors |
| trunk + experts | Q4C-P (4-bit, per-column groups) |
| n-gram embedding table | FP8, 47.7 GiB, a lookup, paged rather than resident |
| rank-1 / conv1d / PLE projections | BF16 pass-through |
| quality sidecar | 723 non-expert tensors re-quantized activation-aware, **plus the twelve `o_proj` at 8-bit**, plus (0.6.0) the MTP head's 18 dense projections at 8-bit |

The sidecar is where most of the quality lives, and it is small because the
loss was concentrated. Measuring each tensor family against its own BF16
ceiling put nearly all of the non-expert quantization cost in **twelve
`o_proj` tensors, 106 MB, 0.09% of the file.** Those twelve now ship at 8
bits, which measures as a statistical tie with full BF16 rows. The mechanism
is calibration: a 4-bit trunk is over-confident, and the whole gain sits in
the hardest quartile of predictions.

**0.6.0 adds the MTP head's own dense projections at 8 bits** (18 tensors,
0.09 GiB) to both sidecars. They change no answer: the head only proposes
tokens and the trunk verifies every one, so this is a speed change. Its
proposals are accepted more often (51% to 59% on prose at the same trunk),
which is worth about 4% of decode speed on prose and nothing on code. An
older image reads the same file and gains the same; a 0.6.0 image with the
older file says so at startup and, started with `HALOGEN_DOWNLOAD` set, fetches
just this file.

## Speculative decoding

The checkpoint carries the model's **MTP head** (a full Flash-Next layer , 
512 experts and its own indexer). Speculative decoding with it is
**byte-identical to serial greedy decode**: the head only proposes, and a
token is emitted only if the trunk would have produced it. It is a speed
optimization with no quality cost, on by default, and disableable per
request.

Since 0.6.0 the server also drafts from the request's own text: when the last
few tokens of the answer already occur earlier in the conversation, the tokens
that followed them are proposed as a chain and verified in one step. Tool-call
arguments, file paths and code that quotes the file being edited are mostly
such copies, and that is where an agent's output spends its tokens. Same
guarantee: every emitted token is the model's own greedy choice, and the same
rule: it runs while the request is the only one generating.

## Provenance and license

Derived from **Qwen3.8-Flash-Next**. These weights are a derivative work and
are governed by the license of the original model, see the
[base model](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for those terms.
Verify them yourself before commercial use; nothing here grants rights the
upstream license does not. Unlike halogen's 27B checkpoint, no third-party
quantization was imported: every quantized value here was produced by us from
the BF16 originals.

The halogen-flash **engine** is separate, closed source, and distributed under
its own terms, see the
[server repository](https://github.com/peonist-ai/halogen-flash-server).
