---
library_name: llama.cpp
base_model:
  - Qwen/Qwen3.8-27B
tags:
  - gguf
  - speculative-decoding
  - dflash2
  - mtp
inference: false
---

# pplx-computer-qwen-3-8-27b-dflash2-gguf-20260826

llama.cpp GGUF build of the pplx-computer qwen3.8 27B checkpoint
(`Qwen3_5ForConditionalGeneration`, dense hybrid GDN linear-attention + full
attention every 4th of 64 layers), converted 2026-08-26 from the raw bf16
checkpoint. Ships **two speculative-decoding drafts — pick one**: the
checkpoint's native MTP head, or the DFlash2 block-diffusion drafter.

## Files

| File | Size | What |
|---|---|---|
| `qwen38-27b-Q4_K_M.gguf` | 16 GB (4.92 BPW) | target model |
| `draft/qwen38-27b-mtp-Q8_0.gguf` | 3.0 GB | native MTP draft (exported from the checkpoint's own MTP layer) |
| `draft/Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | 1.1 GB | DFlash2 draft ([z-lab/Qwen3.8-27B-DFlash2-GGUF](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2-GGUF)) |
| `qwen38-27b-mmproj-f16.gguf` | 928 MB | vision tower (mmproj), optional |

## Option 1 — native MTP (works on stock llama.cpp master)

```bash
llama-server \
  -m qwen38-27b-Q4_K_M.gguf \
  -md draft/qwen38-27b-mtp-Q8_0.gguf \
  --spec-type draft-mtp \
  -ngl 999 -ngld 999 -c 32768 --host 0.0.0.0 --port 8090
```

## Option 2 — DFlash2 (needs llama.cpp PR #27342)

DFlash2 is **not** in llama.cpp master: build
[PR #27342](https://github.com/ggml-org/llama.cpp/pull/27342) (`z-lab:dflash2`,
verified at `f7aadef`). On master the draft fails with
`done_getting_tensors: wrong number of tensors; expected 81, got 58`.

```bash
llama-server \
  -m qwen38-27b-Q4_K_M.gguf \
  -md draft/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  -ngl 999 -ngld 999 -c 32768 --host 0.0.0.0 --port 8090
```

## Which one?

Measured on CPU (identical prompt, temp 0, 2026-08-26):

| draft | acceptance | mean accepted len | llama.cpp build |
|---|---|---|---|
| native MTP Q8_0 | 0.83 | 3.50 | stock master |
| DFlash2 Q4_K_M | 0.67 | 5.71 | PR #27342 only |

DFlash2 drafts longer blocks per verification step (higher throughput
ceiling); native MTP has a higher acceptance rate and runs on unpatched
llama.cpp. Default to MTP unless you already run the PR build.

## Docker compose

A ready-to-run compose file (DFlash2, image built from the pinned PR commit)
is in [`serving/`](./serving) together with step-by-step instructions.
