---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: halogen
# The widget is off deliberately: these are .hgn weights, so nothing on the
# Hub can load them and a widget could only fail. The pipeline tag stays,
# because it is what puts the model in the text-generation filter.
inference: false
tags:
  - halogen
  - qwen
  - quantized
  - 4-bit
  - vision
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

# halogen, Qwen3.8-27B for AMD Strix Halo (4-bit)

Weights for [halogen](https://github.com/peonist-ai/halogen-flash-server), a
dedicated inference engine for AMD Strix Halo (gfx1151). This repository holds
Qwen3.8-27B in halogen's current 4-bit format, with its vision tower.

**Pre-release.** These files are published ahead of the halogen release that
serves them. No published halogen image loads them yet; this card will say
which version does when it ships. The earlier 27B product,
[halogen-server](https://github.com/peonist-ai/halogen-server), uses a
different checkpoint from
[halogen-qwen3.8-27b](https://huggingface.co/peonist-ai/halogen-qwen3.8-27b)
and does not load these files.

**These weights are in halogen's own `.hgn` format and will not load in
transformers, vLLM, or llama.cpp.**

## What is here

```
qwen38-27b-ht4.hgn      12.84 GiB   the checkpoint, 4-bit, with the model's draft head
qwen38-27b-vision.hgn    0.86 GiB   optional: the vision tower, reads images
```

| file | bytes | sha256 |
|---|---|---|
| `qwen38-27b-ht4.hgn` | 13,786,218,048 | `cd2570dc985174ecbea82b65b2ed5c3125d56512fdf88ddd43f7b17c37cd5eff` |
| `qwen38-27b-vision.hgn` | 921,514,496 | `cae5552fc4e34ad0f74c54c11d4f3b91d55fa3652e1b0ff52dbd65859daa83fd` |

The tokenizer is the base model's, unchanged. A flat copy is in
[halogen-qwen3.8-27b](https://huggingface.co/peonist-ai/halogen-qwen3.8-27b/tree/main/tokenizer).

## Precision

The checkpoint stores the language model at about 4 bits a weight: 12.84 GiB
against 52 GiB at BF16, which leaves most of the machine's memory for context.

The vision tower is not quantized: `qwen38-27b-vision.hgn` holds the base
model's BF16 tower tensors bit for bit. It is a separate file because vision
is optional: a server started without it loads nothing for images and is the
text-only engine exactly.

## Provenance and license

Derived from **Qwen3.8-27B**. These weights are a derivative work and are
governed by the license of the original model, see the
[base model](https://huggingface.co/Qwen/Qwen3.8-27B) for those terms. Verify
them yourself before commercial use; nothing here grants rights the upstream
license does not. No third-party quantization was imported: every quantized
value here was produced by us from the BF16 originals.

The halogen **engine** is separate, closed source, and distributed under its
own terms, see the
[server repository](https://github.com/peonist-ai/halogen-flash-server).

"halogen" is a trademark of Peonist, LLC (U.S. application pending); see
`TRADEMARKS.md` in the server repository for permitted use.
