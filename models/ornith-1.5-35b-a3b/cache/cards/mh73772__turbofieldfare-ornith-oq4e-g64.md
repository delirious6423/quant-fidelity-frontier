---
license: mit
base_model:
- ornith-ai/Ornith-1.5-35B-A3B
- scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp
- shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY
base_model_relation: quantized
pipeline_tag: text-generation
inference: false
tags:
- turbofieldfare
- gturbo
- apple-silicon
- metal
- moe
- linear-attention
- mtp
- speculative-decoding
language:
- en
---

# TurboFieldfare — Ornith-1.5 35B-A3B (oQ4e-g64, grafted MTP head)

Prebuilt weights for [TurboFieldfare](https://github.com/drumih/turbo-fieldfare),
a Swift + Metal inference runtime for Apple Silicon. The Mac app downloads this
repository and verifies every file against a SHA-256 pin compiled into the app.

> **Not a `transformers`, MLX, or GGUF checkpoint.** These files are
> TurboFieldfare's own packed format and nothing else reads them. For the
> upstream weights, see [Provenance](#provenance).

## What is different from upstream

Two changes were baked in before packing, and both are why this is published
prebuilt rather than repacked on the user's machine:

- **`q_norm` is folded** into the full-attention layers.
- **The MTP head is replaced.** The `mtp.*` tensors shipped in
  `ornith-ai/Ornith-1.5-35B-A3B` are random initialization rather than trained
  weights, in the upstream bf16 originals as much as in any quantization of
  them. This build grafts the KL-distilled head from
  [`shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY).
  The evidence and the graft are written up in
  [`docs/qwen35moe/30-MTP-HEAD-GRAFT.md`](https://github.com/drumih/turbo-fieldfare/blob/main/docs/qwen35moe/30-MTP-HEAD-GRAFT.md).

The head sits in `mtp-head/` as a sidecar, not as a section of the pack. If it
is missing, the engine loads with speculative decoding silently off.

## Contents

21 GB. The repository layout is the install layout.

| Path | Size | |
| --- | --- | --- |
| `manifest.json` | 11 KB | architecture, quantization, file index |
| `model_weights.bin` | 2.3 GB | shared core: embeddings, lm_head, attention, norms, routers |
| `packed_experts/` | 18.1 GB | routed experts, one file per layer, streamed from SSD |
| `mtp-head/` | 503 MB | grafted MTP head |
| `tokenizer/` | 13 MB | tokenizer and chat template |

## Model

35B parameters, ~3B active. 40 layers in `[linear, linear, linear, full] × 10`,
so full attention lands on layers 3, 7, … 39; 256 experts per layer, top-8
routed. No vision path. Context 4K–128K, 32K by default — 128K loads but
roughly halves decode throughput unless `iogpu.wired_limit_mb` is raised, so
32K and 64K are the operating points this project uses.

Routed experts are 4-bit affine, group 64; embeddings, attention and the shared
expert are 8-bit; the router stays bf16. KV cache is FP16.

## Use

```bash
git clone https://github.com/drumih/turbo-fieldfare.git
cd turbo-fieldfare
swift build -c release
.build/release/TurboFieldfareMac      # Download, then Load Model
```

Or place the files yourself:

```bash
hf download mh73772/turbofieldfare-ornith-oq4e-g64 \
  --local-dir ~/Library/Application\ Support/TurboFieldfare/ornith-oq4e-g64.gturbo
```

Inside a checkout the app looks in `scratch/ornith-oq4e-g64.gturbo` instead. A
manual copy carries no install receipt, so the app re-verifies the hashes once.

The app's defaults: the checkpoint's official sampler, temperature 0.6 /
top-k 20 / top-p 0.95, pinned rather than editable; thinking on; speculative
decoding on at width 2, which is the only width the head has; prompt-cache
reuse on strict extensions of the previous request.

## Provenance

- Base model — [`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B), MIT
- Quantized checkpoint this was packed from — [`scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp`](https://huggingface.co/scottlowry/Ornith-1.5-35B-A3B-oQ4e-mtp), no license declared
- MTP head — [`shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY`](https://huggingface.co/shisa-ai/Ornith-1.5-35B-A3B-MTP-ONLY), apache-2.0
- Bake, graft and packing format — [TurboFieldfare](https://github.com/drumih/turbo-fieldfare), Apache-2.0

The weights follow their upstream terms; this repository changes their packing.

## Limitations

- Apple Silicon, macOS 15+, TurboFieldfare only.
- Speculative decoding is a trade, not free speed: it wins on tasks whose plain
  decode is slow enough for the expert fetches to be shared, and loses on the
  rest. The Mac app turns it on, the CLI and HTTP server leave it off.
- MTP output is not bit-identical to plain decode — the wide kernel sums in a
  different order.
