---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- gguf
- quantization
---

# Qwen3.8-27B — per-type GGUF store (qalloc store-v2)

This repo is a **byte-addressable quantization store**, not a single quant.
It holds Qwen3.8-27B quantized at ~41 different types (mainline llama.cpp
and ik_llama.cpp families unified; per-type `backend` flag in the manifest):

- `types/<TYPE>.gguf` — the whole model at one type. Every file is a normal,
  **directly runnable** uniform quant (a mainline-typed file runs on stock
  llama.cpp; ik types need ik_llama.cpp).
- `main.gguf` — header KV metadata + all non-quantizable tensors (with the
  upstream chat-template fix applied).
- `manifest.json` (`qalloc-manifest/v2`) — the byte index: for every tensor
  × type, the exact `[offset, offset+bytes)` range inside its per-type file.
  A client can assemble any **mixed-precision** allocation with plain HTTP
  Range requests — no server-side work.
- `calib/`, `eval/` — the calibration and evaluation corpus files used by
  the damage-model program that drives allocation.

Assembly tooling and the allocation picker (exact MCKP solver over a
measured damage model) live in the `qalloc` project; the interactive picker
runs in the browser via WASM. The former v1 stores (per-tensor files, two
repos) are deprecated: this repo IS the store.
