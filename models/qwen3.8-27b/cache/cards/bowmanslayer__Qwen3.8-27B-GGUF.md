---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
tags:
  - qwen3.5
  - gguf
  - llama.cpp
  - imatrix
  - quantized
  - speculative-decoding
  - mtp
---

# Qwen3.8-27B · GGUF (imatrix)

**Qwen3.8-27B for llama.cpp — importance-matrix quantized from the original BF16.
Runs from 16 GB VRAM up.**

The main weight files carry the language model. **Vision** ships as a separate
`mmproj` file and the model's **multi-token-prediction head** as a separate
speculative draft — llama.cpp keeps both outside the main file by design, not
because anything was cut down. Take either, both, or neither.

> **Renamed 2026-08-21** from `Qwen3.8-27B-Text-Only-GGUF`. The old name described the layout of one
> file but read as "this model cannot see", which was never true — the vision projector
> has been in this repo since it was first published. Old links redirect automatically.

> The **file names** still say `Text-Only`. Hugging Face redirects a renamed *repo*
> but not individual file URLs, so renaming the files would 404 every existing direct
> link and download script. They stay as they are.

> Community quantization. **Not** an official Qwen release; **not** endorsed by or
> affiliated with the Qwen team or Alibaba Cloud. "Qwen3.8" identifies the upstream
> model this artifact derives from (Apache-2.0 §6).

---

## Which file do I download?



| File | Size | pp512 | **tg128** | **Max context on one 3090** | |
|---|---|---|---|---|---|
| `Q8_0`   | 26.6 GiB | — | — | **does not fit** | Near-lossless. Needs 32 GB, or partial offload on 24 GB |
| `Q6_K`   | 20.6 GiB | 1088 | 32.3 | **32,512** | Best quality on one 24 GB card — but look at that context |
| `Q5_K_M` | 17.9 GiB | 1110 | 36.9 | **73,472** | |
| **`Q4_K_M`** | **15.4 GiB** | **1183** | **41.4** | **111,872** | **Fastest, and 3.4× the context of Q6_K.** The default pick |
| **`IQ4_XS`** | **14.0 GiB** | 1042 | 31.9 | **133,632** | **Pick for a 16 GB card** — see the caveat below |
| `Q3_K_M` | 12.4 GiB | 992 | 38.5 | **158,976** | Quality drops noticeably |
| `mmproj-*` | 0.9 GiB | — | — | — | Optional vision, pairs with any of the above |
| `mtp-*-Q4_K_M` | 1.9 GiB | — | — | — | Optional MTP speculative draft, **+36 % on CUDA** — see below |

All three columns measured on **one RTX 3090 (24 GB)**, Vulkan, full offload. Context
figures come from `llama-fit-params`, i.e. what actually fits — not a calculation.

**The size/context trade is steeper than the size/quality trade.** Dropping from `Q6_K`
to `Q4_K_M` costs a little quality and buys **3.4× the context** — on the same card.

**The counter-intuitive part: `IQ4_XS` is 9 % smaller than `Q4_K_M` but ~23 % slower.**
IQ-family quants cost more compute to dequantize. Take `IQ4_XS` because you need the
*size*, not because you want speed. If `Q4_K_M` fits your card, it is both faster and
higher quality.

**16 GB card:** `IQ4_XS`. `Q4_K_M` technically loads but leaves almost nothing for KV.
**24 GB card:** `Q4_K_M` for speed, `Q6_K` for quality.
**Apple Silicon:** unified memory is the budget — 32 GB → `Q5_K_M`/`Q6_K`, 64 GB → `Q8_0`.

*Vulkan numbers. CUDA builds are typically faster.*

---

## Why the KV cache is unusually small

Qwen3.8 is a **hybrid-attention** model. Of its 64 layers only **16 are full attention** —
the other 48 are Gated DeltaNet linear attention and hold no KV cache. With
`num_key_value_heads = 4`, `head_dim = 256`:

| KV dtype | Per token | 32K ctx | 128K ctx |
|---|---|---|---|
| f16 | 64 KiB | 2.0 GiB | 8.0 GiB |
| q8  | 32 KiB | 1.0 GiB | 4.0 GiB |

A comparable dense-attention 27B needs roughly **four times** this — which is why the
context figures in the table above are as large as they are.

*The table is measured on a 24 GB card. For a 16 GB card with `IQ4_XS` (14.0 GiB) the
same arithmetic gives roughly 32K at f16 KV or 64K at q8 — **estimated, not measured**,
as the author has no 16 GB card to test on.*

---

## Vision

```bash
llama-mtmd-cli -m Qwen3.8-27B-Text-Only-Q4_K_M.gguf \
               --mmproj mmproj-Qwen3.8-27B-f16.gguf \
               --image photo.jpg -p "Describe this image."
```

The projector is the general-purpose tower from `Qwen3.8-27B`, unchanged. Dedicated
`Qwen3-VL-*` models will still do better on dense OCR and small-object counting.

---

## Speculative decoding (MTP)

Qwen3.8 was trained with a multi-token-prediction head. llama.cpp's converter publishes
the target model and the MTP draft as **two files by design**, so the head is here as its
own optional download rather than inside the main weights.

```bash
llama-cli -m Qwen3.8-27B-Text-Only-Q4_K_M.gguf \
          -md mtp-Qwen3.8-27B-Q4_K_M.gguf \
          --spec-type draft-mtp --spec-draft-n-max 1 \
          -p "..."
```

Works with `llama-cli` and `llama-server`. **`llama-completion` does not accept `-md`.**
Requires build `b10502` or newer — `draft-mtp` does not exist before it.

### The numbers

One RTX 3090, `Q4_K_M` target, 512 tokens, greedy, single stream. `no draft` and
`n-max 1` are the mean of three runs; the rest are single runs.

**CUDA**

| `--spec-draft-n-max` | tok/s | Draft acceptance |
|---|---|---|
| *no draft* | 41.2 | — |
| **1** | **56.1** | **75 %** |
| 2 | 54.6 | 63 % |
| 3 *(default)* | 48.6 | 50 % |
| 4 | 45.1 | 41 % |

**+36 %** at `n-max 1`. Every setting helps on CUDA, but the default of 3 leaves a third
of the gain on the table.

### The backend matters more than anything else here

The same card, same model, same prompt, on the **Vulkan** build:

| `--spec-draft-n-max` | tok/s | Draft acceptance |
|---|---|---|
| *no draft* | 39.1 | — |
| **1** | **40.4** | 77 % |
| 2 | 33.7 | 62 % |
| 4 | 28.3 | 44 % |

**+3 %** at best, and *negative* at the default. Acceptance is the same 77 % — the draft
is doing its job either way. What differs is the cost of batched verification, which
Vulkan does not make cheap enough to pay for the draft's own forward pass.

**Metal** (M1 Max, 64 GB unified, same files, same method):

| `--spec-draft-n-max` | tok/s | Draft acceptance |
|---|---|---|
| *no draft* | **11.17** | — |
| 1 | 9.96 | 77 % |
| 2 | 8.80 | 66 % |

**Negative.** −11 % at `n-max 1`, −21 % at 2.

So: **+36 % on CUDA, +3 % on Vulkan, −11 % on Metal — from identical files, at the same
75–77 % acceptance on all three.** The draft does its job everywhere; what differs is
what batched verification costs on each backend. **On Apple Silicon, do not attach the
draft.** If you are on ROCm, SYCL, or anything else, measure before assuming any of these
numbers applies to you.

*Apple Silicon users: llama.cpp is not the fastest path for this model anyway. An MLX
build of the same weights ran **15.9 tok/s** on the same machine — 42 % faster than
llama.cpp Metal, and no draft involved.*

### Only one draft file, and why

`Q8_0` (2.9 GiB) was built and tested too: **55.0 tok/s at 73.8 % acceptance**, versus
**56.1 at 75.0 %** for this `Q4_K_M` (1.9 GiB) — and the two produced **byte-identical
output**. The bigger draft was slower and no more accurate, so it is not published. A
draft cannot change what the target accepts, so there is no quality argument for it.

### It is not bit-identical to running without a draft

Speculative decoding preserves the output **distribution** — every drafted token is
verified against the target, so nothing gets through that the target would not have
produced. It does **not** reproduce the same string. Batched verification changes
floating-point reduction order, so a near-tie between two candidate tokens can land on
the other one.

Measured: each configuration is perfectly reproducible with *itself* (two greedy runs
byte-identical), but greedy output with the draft diverged from greedy output without it
after ~338 characters on CUDA and ~121 on Vulkan, continuing as an equally coherent
answer. **If you need byte-reproducible output, do not attach a draft.**

---

## How these were made

Quantized **from the original BF16 weights**, not re-quantized from an existing INT4
release — so no compounding loss.

1. Vision tower and MTP block split off at the safetensors level;
   `model.language_model.*` promoted to `model.*`
2. `convert_hf_to_gguf.py --outtype bf16 --no-mtp` → BF16 GGUF (lossless from source).
   The `mmproj` and `mtp-*` files come from second passes over the *unmodified*
   source with `--mmproj` and `--mtp` — the pairing llama.cpp's converter documents
3. **Importance matrix** over 300 chunks of the same calibration corpus used for this
   author's W4A16 releases (512 passages, pile-val news text), on 2×RTX 3090
4. Every level quantized **with that imatrix**, K-quants included

**Every file in this repo was loaded and generated with before publishing** — including
both `mmproj` files, which were checked against a synthetic image with known content.
Not just checksum-verified.

---

## Requirements

A llama.cpp build that knows the `qwen35` architecture. Build `b10502` or newer works.

> Older builds fail on the **main** files with `check_tensor_dims: tensor 'blk.64...'
> not found`. That is a converter that counted an MTP block the main files do not carry —
> not a corrupt download. The `mtp-*` files are the opposite case: they *are* block 64,
> and `--spec-type draft-mtp` only exists in `b10502` and newer.
