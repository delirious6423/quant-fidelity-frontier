---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
tags:
  - android
  - on-device
  - gptq
  - 4-bit
  - geode
  - qwen
---

# Geode · Qwen3.8-Flash-Next for phones

A model pack for **[Geode](https://geode.tifu.xyz)**, an Android app that runs language models entirely on the phone:
no cloud, no account. [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is the Qwen team's
180 B-parameter mixture of experts: 48 layers (Gated DeltaNet and attention with a sparse indexer), four
hyper-connection streams, 512 experts of which 10 answer each token, and a 51 B n-gram embedding table read a few
rows a token. It is by far the largest model Geode runs and it is slow on a phone: an extra for the biggest phones,
there because it can be.

## Pack

| Folder | Weights | Size | For |
| --- | --- | --- | --- |
| [`Q4/`](./Q4) | routed experts and attention / DeltaNet / shared-expert matrices 4-bit (Geode's q4: an FP8 scale per 16), quantized by GPTQ straight from Qwen's BF16 checkpoint; lm_head and the embedding 8-bit from BF16; routers, hyper-connections, the indexer, the n-gram projections bf16; the n-gram table in one of two variants (below); the vision encoder and the MTP draft layer | 125 GB with the FP8 table, 103 GB with the q4 table | 64-bit phones with 12 GB of RAM or more and that much free storage |

The n-gram table comes in two variants; the app downloads one:

| Variant | Files | Size | |
| --- | --- | --- | --- |
| FP8 (default) | `ngram-f8-*.bin` | 51.2 GB | Qwen's own FP8 bytes, bit for bit |
| q4 | `ngram-q4-*.bin` | 28.8 GB | saves 22 GB; the answers are slightly less accurate |

Only a few rows of the table and ten experts a layer are read for each token, so the pack runs from flash with a
small part of it in RAM; reading those experts from storage is what sets the speed.

The packs are Geode's own format (`geode.json` describes the files). They are not GGUF and do not load in other
runtimes.

## Fewer refusals (optional add-on, off by default)

`Q4/refusal/` holds a small optional add-on (272 KB, not part of the base download): `refusal.gguf`, one direction of
2,560 values for each of layers 8-33 (llama.cpp control-vector format), and `refusal.json`, its settings and scores.
With the model's *Fewer refusals* switch on (or `"fewer_refusals": true` in a request to Geode's local API), the engine
removes that layer's direction from every hyper-connection stream after each of those layers (h -= (h . v) v), and the
model declines far fewer requests. It acts on the activations, so it needs no other weights and serves any
quantization; the projection costs 0.2-0.4% of the decode speed (measured on the PC build). Off, nothing of it is loaded and the model runs exactly as without it.

Measured on the PC build of the same weights (an NVFP4 GPTQ quant), against its stock model (teacher-forced, 5,153 rows of code, prose and the model's own replies): KL 0.026,
the same top token 94.4%; held-out harmful requests refused 30/30 (EN) and 29/30 (UK) without it, 0/104 and 0/30 with
it (thinking on). The vector was made from a GPTQ quant of Qwen's BF16 weights, so the Qwen Community License 1.0
applies to it as to the rest of the pack. Removing refusals removes a safety behaviour of the model: you are
responsible for what it writes with the switch on.

## Use

1. Install Geode (Google Play beta: [geode.tifu.xyz](https://geode.tifu.xyz)).
2. Models → *Download from Hugging Face* → `Maximilian228/Geode-Qwen3.8-Flash-Next`, choose the n-gram table.

Requirements: Android 11+, a 64-bit ARM processor with the dot-product instructions (ARMv8.2), 12 GB of RAM or more,
and free storage for the pack.

## License and attribution

The weights are the Qwen team's [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), under the
Qwen Community License 1.0 (`LICENSE`, copied unchanged); this pack is a derivative work under the same license, and
its terms (the notice; the separate license Qwen requires for commercial Model-as-a-Service or AI Work Assistant use)
apply to it. Changes made here: the model converted into Geode's file layout and quantized - the routed experts and
the attention / DeltaNet / shared-expert matrices quantized to 4 bits by GPTQ from the BF16 checkpoint, lm_head and
the embedding rounded to 8 bits, the n-gram table kept as FP8 or rounded to 4 bits. Answers come from an
AI model and can be wrong.
