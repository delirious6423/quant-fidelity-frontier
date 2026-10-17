---
base_model:
- Qwen/Qwen3.8-27B
license: apache-2.0
tags:
- unsloth
- gguf
- archival-mirror
---

# Archival mirror: the five `unsloth/Qwen3.8-27B-GGUF` files our published comparison cites, at revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`

**This repository exists so that a published measurement stays reproducible.** It is a byte-identical mirror of five files from [`unsloth/Qwen3.8-27B-GGUF`](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) at one pinned revision — nothing requantized, re-exported, or edited. Use the upstream repository for anything current; it has the full quant ladder and this has only the five files a published table depends on.

## Why this exists, stated precisely

At the time of writing, revision `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` **still resolves upstream** and is in fact current HEAD. So unlike our [NVFP4 archival mirror](https://huggingface.co/malaiwah/Qwen3.8-27B-NVFP4-archival-9c73e2da), this one is precautionary rather than a recovery.

The precaution is not hypothetical. The sibling repository `unsloth/Qwen3.8-27B-NVFP4` was super-squashed on 2026-08-15, and the revision our published comparison cited stopped resolving — the Hub now answers `{"error":"Invalid rev id"}` for it. Our local snapshot turned out to be the only verifiable copy. Squashing history is a routine maintainer action and entirely upstream's call; the lesson is ours: **a Hugging Face revision id is not a durable citation, so anything a published number depends on gets mirrored while it still exists rather than after it does not.**

Mirroring is also nearly free, which is what makes it a sensible default rather than a special measure. Hugging Face Xet storage is content-addressed globally, so re-uploading bytes that already exist on the Hub transfers almost nothing: our 22.5 GB NVFP4 mirror moved about 205 MB on the wire.

## Files and digests

Every file was verified locally against the published `lfs_sha256` of revision `f1bfb127` before upload, and re-verified from the Hub's own tree API afterwards.

| file | size (B) | sha256 |
|---|---|---|
| `Qwen3.8-27B-Q8_0.gguf` | 29,047,086,048 | `a680f44a06920e5d689774823782006aa3acc8db95750323373b24139b67e348` |
| `Qwen3.8-27B-Q6_K.gguf` | 22,884,408,288 | `562fbf760503008f118e5df38de5b3e97992d1f693f475815631198547486727` |
| `Qwen3.8-27B-UD-Q5_K_XL.gguf` | 20,218,178,624 | `176a6a3f034e9cdc447c10cd00329fc9b31002e6589b9295f2ad4f1eefe0f6ab` |
| `BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf` | 49,986,159,616 | `b9966e82b7a4d87028b5eae061d578ee826305ebf8baea5bfc6e09bad0ba191f` |
| `BF16/Qwen3.8-27B-BF16-00002-of-00002.gguf` | 4,671,576,000 | `92e3943c4f9bd6292a7bef82369f65fed9bfed088b9df0fb2fa2ce17c9edfa02` |

Total 126.8 GB. Upstream paths are preserved, so the two-part BF16 set still loads as `BF16/Qwen3.8-27B-BF16-00001-of-00002.gguf`.

## What we measured with these files

They are the GGUF rows of a teacher-forced distribution-fidelity comparison of Qwen3.8-27B quantizations: KL(BF16 reference ‖ candidate) over a held-out suite of 512 contexts and 1,048,064 scored positions, full 248,320-entry vocabulary, every candidate read at the post-final-norm state and projected through **one shared BF16 LM head**, so no candidate's own head quantization is counted.

- `Q8_0` — 0.001087 mean KLD, the fidelity leader on that suite
- `Q6_K` — 0.002035
- `UD-Q5_K_XL` — 0.004444
- the **two-part BF16 set** is not a quantization row at all: it measures the *cross-engine floor*. Captured through llama.cpp and compared against the same weights served by vLLM, it answers "what do two engines disagree by on identical unquantized weights?" The answer is **0.000507 mean KLD, 99.07 % top-1 agreement**, and every GGUF row above contains that term while the vLLM-side rows do not. Without this file pair, none of the GGUF numbers could be read net of engine numerics — which is exactly why it is mirrored alongside them.

Full method, intervals and caveats are in the `receipts/` tree of the qwen38-27b project: `receipts/cross-engine-comparator.json`, `receipts/gguf-report-q8_0.json`, `receipts/gguf-report-q6_k.json`, `receipts/gguf-report-q5_k_xl.json`, `receipts/gguf-report-engine-floor.json`.

## Attribution and licence

Quantization and packaging by [unsloth](https://huggingface.co/unsloth); base model [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). Licence `apache-2.0`, inherited unchanged from upstream, as are the `base_model` and `unsloth` tags. All credit for these artifacts belongs upstream; this repository contributes nothing but durability.
