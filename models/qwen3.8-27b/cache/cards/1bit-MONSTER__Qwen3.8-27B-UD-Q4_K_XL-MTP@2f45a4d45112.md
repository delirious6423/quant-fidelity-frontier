---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
- unsloth/Qwen3.8-27B-GGUF
tags:
- gguf
- mtp
- speculative-decoding
- qwen3.8
---

# Qwen3.8-27B — UD-Q4_K_XL + MTP

A ready-to-run bundle of the quantization/draft-head combination we measured as the
speed/quality "sweet spot" for Qwen3.8-27B on consumer AMD hardware (Strix Halo iGPU,
Vulkan): Unsloth's `UD-Q4_K_XL` dynamic quant paired with its matching MTP (multi-token
prediction) draft head, used together via speculative decoding.

This repo doesn't introduce new weights — it packages two files Unsloth already
publishes separately (`unsloth/Qwen3.8-27B-GGUF`) as one bundle, documented with the
specific combination and flags we measured to work best together.

## Contents

- `Qwen3.8-27B-UD-Q4_K_XL.gguf` — the base model, Unsloth's Dynamic 2.0 Q4_K_XL quant.
- `MTP/mtp-Qwen3.8-27B-Q4_0.gguf` — the matching MTP draft head, same base model.

## Why this combination

Measured on Strix Halo (Vulkan backend), across the UD quant ladder (IQ4_XS through
Q8_0) with and without MTP speculative decoding:

- **UD-Q4_K_XL** was the accuracy/speed sweet spot: KLD 0.0081 and 95.3% top-1 agreement
  against the reference, with IQ4_XS matching its speed at 2.3x the KLD, and Q5/Q8
  costing meaningfully more for little accuracy gain.
- Pairing it with the **MTP draft head** (`--spec-type draft-mtp -md MTP/mtp-Qwen3.8-27B-Q4_0.gguf`,
  default max draft length of 3) took decode from ~12 tok/s to 29–36 tok/s in raw
  `llama.cpp`, and 35.0 / 28.3 / 28.9 tok/s (code / prose / short prompts) through our
  own engine's `1bit serve --mtp`, vs 12.2 tok/s without it. Vulkan was consistently the
  better backend for this pairing over ROCm.

## Running it

With plain `llama.cpp` (`llama-server`, Vulkan build):

```bash
llama-server -m Qwen3.8-27B-UD-Q4_K_XL.gguf \
  --spec-type draft-mtp -md MTP/mtp-Qwen3.8-27B-Q4_0.gguf \
  -ngl 99 --jinja
```

With the [1bit engine](https://github.com/1bit-MONSTER/engine) (`1bit serve`):

```bash
1bit serve -m Qwen3.8-27B-UD-Q4_K_XL.gguf --device vulkan \
  --mtp MTP/mtp-Qwen3.8-27B-Q4_0.gguf
```

`--spec-draft-n-max` (or the engine's default) controls the draft length; 3 was the
measured best for this model — 5 was slower.

## DFlash2 drafter (alternative to the MTP head)

The same target file also runs with the DFlash2 block drafter, converted in
[1bit-MONSTER/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/1bit-MONSTER/Qwen3.8-27B-DFlash2-GGUF)
(Q8_0 and Q4_K). On the 1bit engine's HRX route:

```bash
1bit serve -m Qwen3.8-27B-UD-Q4_K_XL.gguf --device hrx --dflash Qwen3.8-27B-DFlash2-q8_0.gguf
```

`--mtp` and `--dflash` are exclusive. DFlash2 drafts longer blocks, so it pays most on code and
structured output; the MTP head above stays the better default for short chat turns.

## Attribution

- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0).
- Quantization and MTP draft head: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
  (Unsloth Dynamic 2.0 GGUFs). All credit for the actual weights goes to the Qwen team
  and Unsloth — this repo only documents and bundles a specific combination we found
  and measured.
- License: Apache 2.0, inherited from the base model and the upstream GGUF repo.
