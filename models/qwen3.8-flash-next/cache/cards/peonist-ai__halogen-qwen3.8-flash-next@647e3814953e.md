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
Questions and numbers from your own machine go to the
[Discord](https://discord.gg/bcm6QknaV6). Bugs go to the server repository's
issues.

**These weights are in halogen's own `.hgn` format and will not load in
transformers, vLLM, or llama.cpp.** They exist to be mounted into the
halogen-flash-server container.

## What is here

```
qwen38-flash-next-v2.hgn                62.1 GiB   the checkpoint (0.15 on)
qwen38-flash-next-ngram.hgn             47.7 GiB   its lookup table, KEEP THIS
qwen38-flash-next-ht43.hgn              53.7 GiB   optional (0.16 on): a smaller checkpoint, opt in
qwen38-flash-next-vision.hgn             0.84 GiB  optional: reads images
qwen38-flash-next-mtp.hgn                1.42 GiB  optional: the draft head, for running a GGUF
qwen38-flash-next-w4b.hgn              115.55 GiB  0.14's checkpoint (still loads)
qwen38-flash-next-w4b.overlay.hgn        2.40 GiB  w4b's quality sidecar
qwen38-flash-next-w4b.overlay-speed.hgn  2.31 GiB  w4b's speed arm
tokenizer/                                         flat tokenizer directory
```

**From 0.15 the checkpoint is `qwen38-flash-next-v2.hgn`, with its lookup
table beside it.** Download both (about 110 GiB). The model needs about
62 GiB resident plus the memory-mapped table. The server fetches exactly
these when `HALOGEN_DOWNLOAD` names this repo on a new volume.

**A smaller checkpoint, opt in (0.16 on).** `qwen38-flash-next-ht43.hgn`
stores the experts in fewer bits and holds about 8 GiB less in memory than
v2, with the same lookup table. Prompts are read somewhat slower with it,
and decode with the draft head a few percent slower. Start the server with
`HALOGEN_CHECKPOINT=/models/qwen38-flash-next-ht43.hgn` to use it. v2 stays
the default.

**0.14's checkpoint is still here and still loads.** `qwen38-flash-next-w4b.hgn`
is two files: the checkpoint and its quality sidecar, which the engine
reads from beside it. A volume that already holds them keeps serving them
after an upgrade, and the startup log says how to move to v2.

**The vision file is separate and optional.** `qwen38-flash-next-vision.hgn`
lets the server read images, and without it the server is text-only and refuses
an image with a message naming the setting that turns it on. Take it if you
want to send screenshots; skip it and nothing else changes, because with no
file the image path is absent rather than disabled and text behaviour is
byte-identical either way. Put it beside the checkpoint and start the server
with `HALOGEN_VISION_TOWER=1`.

**The draft-head file is for running someone else's GGUF.** Since 0.7.0 the
server opens a llama.cpp GGUF of this model directly (unsloth's `UD-IQ4_XS`,
or any `llama-quantize` output whose experts are `IQ4_NL` / `IQ4_XS` / `IQ3_S`
/ `Q4_0` and whose dense layers are `Q8_0`), repacking it losslessly into RAM
at startup. A GGUF carries no draft head this engine can run, so
`qwen38-flash-next-mtp.hgn` supplies it: the checkpoint's own 31 head tensors,
with the head's 18 dense projections at 8 bits as in the quality sidecar. With
a GGUF the checkpoint and the two sidecars above are not needed at all, only
this file and `tokenizer/`; the server fetches both when `HALOGEN_DOWNLOAD`
names this repo. The server repository's README has what that path costs and
buys, measured.

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
  ghcr.io/peonist-ai/halogen-flash-server:0.18.14
```

This repo carries the tokenizer, so one `-v` is all you need. On Docker rather
than Podman, replace `--group-add keep-groups` with
`--group-add video --group-add render`: `keep-groups` is a Podman keyword that
Docker resolves against the container's `/etc/group` and fails to find.

An OpenAI-compatible endpoint comes up on `:8731`: `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, and `/v1/responses`. The last of those is the
Responses API, so the **OpenAI Codex CLI** works against this server directly,
tool calls included, by pointing a `model_providers` entry at it with
`wire_api = "responses"`. Since 0.8.0 a `response_format` of `json_schema`
or `json_object` (`text.format` on the Responses route) is enforced while
the model decodes, so the reply is valid under the schema by construction,
at no cost to speed. Since 0.9.0 an opt-in **composable context** preview
(`HALOGEN_COMPOSABLE_CONTEXT=1`) lets a compacted conversation reuse its kept
tool results without reading them again; it is off by default and not
byte-identical (a reused answer is very close to, not the same as, reading the
text fresh), so every exactness guarantee holds with it off. Since 0.9.1 the
model's sparse-attention budget can be raised at startup
(`HALOGEN_INDEXER_BUDGET=4096`, default 2048): in the server's retrieval
battery 4096 recovered two planted facts at 16k that the default missed, for
about 7% of prefill speed at 32k; off by default, and not byte-identical
either.

Full documentation is in the
[server repository](https://github.com/peonist-ai/halogen-flash-server).

## Precision

4-bit weights are a **correctness precondition, not an optimization**:
Qwen3.8-Flash-Next is 125B total parameters plus a 51B n-gram embedding table,
and neither BF16 (335 GiB) nor FP8 (173 GiB) fits in Strix Halo's 124 GB of
unified memory. The question was never whether to quantize but where to spend
the bits.

**The v2 checkpoint (0.15 on):**

| | |
|---|---|
| checkpoint | 62.1 GiB, 4.16 bits a weight on average |
| experts and the rest of the model | 4-bit |
| the mixing layers between them | 6-bit |
| draft head's dense projections | 8-bit |
| n-gram lookup table | its own file, FP8, 47.7 GiB, paged rather than resident |

**0.14's checkpoint, w4b:**

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

"halogen" is a trademark of Peonist, LLC (U.S. application pending); see
`TRADEMARKS.md` in the server repository for permitted use.
