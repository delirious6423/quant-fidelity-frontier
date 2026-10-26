---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- deepseek
- gguf
- quantized
- strix-halo
- speculative-decoding
---

# DeepSeek-V4.1-Flash-1017 — Strix dense U6 overlay (+ drafter aux files)

A **partial-weight overlay**, not a standalone model. `c-u6.gguf` holds 321 dense tensors of DeepSeek-V4.1-Flash-1017:
attention (`q_a`, `q_b`, `kv`, `wo_a`, `wo_b`), the shared expert, and the output head. They are re-quantized to a
6-bit uniform format (**U6**, 6.5 bits per weight, imatrix-weighted). At load time the overlay replaces the matching
Q8_0 tensors of
[kyuz0/deepseek-v4.1-flash-gguf](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf) `DeepSeek-V4.1-Flash-1017-Q2.gguf`.
The routed experts stay as they are in that file (IQ2_XXS gate/up, Q2_K down).

It only works with that GGUF and with the engine in
[github.com/hyukudan/strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab), where it is the
production dense format of the two-node AMD Strix Halo (gfx1151) DeepSeek profile. The repository also ships the
small auxiliary files that the engine's DSpark speculative drafter uses.

Companion repo: [hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3](https://huggingface.co/hyukudan/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3).
It holds the trellis (EXL3-style) packs that production adds for the shared-expert and head decode sites.

## Files

| Path | Size (bytes) | SHA-256 | What |
|---|---:|---|---|
| `c-u6.gguf` | 7,588,516,384 | `67e6e10ed97bff78c125e4e1433c3cb8dba4c0e3e2b2d42f9595e251b6fa2122` | U6 overlay (identity `c-u6-imat`, content FNV `e53b8712ae30ebf8`, 321 tensors) |
| `c-u6.json` | 43,325 | `24487214c9f3c04d2cd34fa0997bca5e28e426f9e067114a9257b85db63e3201` | recipe and per-tensor report (spec `attn=u6,sh=u6,head=u6`, relative MSE per tensor) |
| `calib/cpu.imat` | 16,798,870 | `a779a7ca48f472b0654d0b2f407b370c1aa21b95219f1fbfd90dd86a25d059b6` | importance matrix used for the requantization |
| `aux/draft-keep-stats.jsonl` | 16,122 | `747e42c29ef39ee2488204f8e3172567cef8d1c7c247a10b4194a17dd535d4a5` | DSpark drafter route statistics (keep-64 expert lists) |
| `aux/vocab/*.bin` (28 files) | 2,490,732 total | see `SHA256SUMS` | DSpark draft-vocabulary token-ID lists |

`SHA256SUMS` lists every file. Check it with `sha256sum -c SHA256SUMS`.

Required upstream files (not included here, download them from kyuz0 at revision `20408f699619df7071aa7cda6034e9cef5bf804b`):

| File | Size (bytes) | SHA-256 |
|---|---:|---|
| `DeepSeek-V4.1-Flash-1017-Q2.gguf` | 365,713,686,528 | `5365629079214ff28becbbab7c0900ef9865b79c0f0c64d951924bb362ae5b67` |
| `DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf` (drafter) | 7,966,294,016 | `d3efe0b1f6b6fea54f1c58c5cbeb3fded5967b15b64476aed00c95806ad3e3e9` |

## How it was made

- **Tool:** `tools/ds41_requant/requant.py` in the engine repo.
- **Input:** it reads the **Q8_0** tensors of kyuz0's Q2 GGUF (not the original FP8 checkpoint).
- **Encoding:** each tensor is written in a Q8_0-embeddable U6 form. Per block, the tool searches the scale (±12 % in
  2 % steps), refits by least squares, then rounds the scale to f16. The engine's unchanged Q8_0 kernels therefore
  compute the overlay exactly, and its packed kernels (`STRIX_DS41_CB4=1`) reproduce the same bits.
- **Imatrix (`calib/cpu.imat`):** per-channel mean x² of the inputs of `q_a`/`kv`, `q_b`, `wo_a` (per group), `wo_b`
  and the shared gate/up. It was collected by the engine's host-forward tap over 8 calibration documents × 128 tokens,
  disjoint from the evaluation set. It lowers KL by about 10 %.
- **Selection:** several formats and per-class mixes were swept: u3–u6, nl3/nl4 codebooks, and learned 16-entry
  codebooks, which did not pay. `c-u6` was the winner under a pre-registered quality gate.

### Auxiliary drafter files

- `aux/draft-keep-stats.jsonl`: per-stage expert route counts of the DSpark drafter, measured on 18 held-out chats.
  The 64 most-routed experts per stage carry 95.4 / 99.8 / 100 % of the routes. With keep-64 pruning, the drafter
  keeps only those experts resident. This cuts its device footprint from about 8.0 GB to 4.4 GB, and measured
  +16 % decode throughput on code.
- `aux/vocab/*.bin`: draft-vocabulary token-ID lists (`core16k`…`core64k`, `core-ranked`, per-language packs
  `pack-<lang>[16k]` for code, de, en, es, fr, it, ja, ko, pt, ru, zh). They are plain little-endian uint32 token-ID arrays,
  built from token frequencies over a mixed corpus of model-generated answers and Wikipedia text. They restrict
  which ids the drafter proposes.

## How to use (strix-halo-engine-lab)

Build the engine as described in its README (`strix-deepseek41-rank`, `strix-deepseek41-serve`).
Production-identity environment for **both ranks** of the two-node group (`docs/models/deepseek-v4.1/production-defaults.md`):

```sh
A=/path/to/DeepSeek-V4.1-Flash-1017-strix-dense-U6          # this repo
B=/path/to/DeepSeek-V4.1-Flash-1017-strix-trellis-EXL3      # companion trellis repo
export STRIX_DS41_OVERLAY=$A/c-u6.gguf STRIX_DS41_CB4=1
export STRIX_DS41_TRELLIS=$B/ours-K4 STRIX_DS41_TRELLIS_MIX=head=$B/ours-K5
export STRIX_DS41_TRELLIS_SITES=shexp,head STRIX_DS41_TRELLIS_HEADOUT=1

strix-deepseek41-serve DeepSeek-V4.1-Flash-1017-Q2.gguf --rank <0|1> --transport usb4gpu \
  --group-manifest <group.json> --layers 40 --tp-attention --engram-direct \
  --prefill-path device --prefill-rows 2048 --slots 8 --context 4096 \
  --dspark DeepSeek-V4.1-Flash-1017-DSpark-MXFP4.gguf --dspark-vocab auto:$A/aux/vocab
```

- **Overlay only.** Drop the three `STRIX_DS41_TRELLIS*` lines. This runs the plain `c-u6` identity, which needs no
  external library.
- **Trellis sites.** They need an engine build linked against
  [github.com/hyukudan/strix-trellis](https://github.com/hyukudan/strix-trellis). Pass
  `-DSTRIX_DS41_TRELLIS_LIBRARY=<archive> -DSTRIX_DS41_TRELLIS_INCLUDE_DIR=<dir>`; see `docs/models/deepseek-v4.1/trellis-library-api.md`.
- **Draft vocabulary.** `--dspark-vocab auto:<dir>` picks a core list plus a language pack per request. Leaving it out
  (full vocabulary) is also valid, just slower.
- **Keep-64 pruning.** The CUDA DeepSeek runner for the RTX PRO 6000 uses the keep-64 lists through
  `--draft-keep 64 --draft-keep-from $A/aux/draft-keep-stats.jsonl`. Builds that embed these lists turn keep-64 on by
  default, and `--draft-keep 0` disables it.
- **Other modes.** The RTX pipeline split, batch and long-context command lines are in
  `docs/models/deepseek-v4.1/` of the engine repo.

## Exactness and quality notes

- **The overlay changes the model's logits relative to the Q8_0 dense tensors.** It is a weight change, not a lossless
  repack. It was measured by teacher forcing on two nodes, 40 layers, full-vocabulary, against the Q8_0-dense model:
  - **KL 0.0121, top-1 agreement 96.9 %**, every category ≤ 0.015 (prose en/es, code, math);
  - long context (7,168 positions) 0.012, falling with position; greedy 44/64 identical tokens.
  - This sits at the measured perturbation floor: re-quantizing only `attn_kv` to 6 bits already gives KL ≈ 0.010.
  - Against the original FP8 checkpoint, with the same Q2 experts and A8 activations, the c-u6 model scores KL 0.0447,
    against 0.0441 for the Q8_0-dense model.
- **Speed and bytes.** On the two-node pair it read 1.11 GB less per token per node and decoded 6.6 % faster when it
  was adopted.
- **Drafter files never change the output.** `draft-keep-stats.jsonl` and `vocab/` affect only what the drafter
  proposes. Speculative decoding in the engine is verified by the target, so the greedy token IDs equal those of
  plain greedy decoding with the same target identity.
- **Bitwise identity depends on the engine build and environment.** The engine announces the identity it runs
  (e.g. `c-u6-imat`, `ds41-trellis-mix-v1{...}`) and gates bitwise against its own references. The SHA-256 values
  above are the ones the engine pins.

## License and credits

- **License: MIT**, same as the base model. `LICENSE` reproduces DeepSeek's MIT notice ("Copyright (c) 2023 DeepSeek").
- **Base model:** [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
  revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, by DeepSeek (MIT).
- **GGUF source:** [kyuz0/deepseek-v4.1-flash-gguf](https://huggingface.co/kyuz0/deepseek-v4.1-flash-gguf),
  revision `20408f699619df7071aa7cda6034e9cef5bf804b`. The overlay is derived from kyuz0's Q8_0 tensors, and the
  DSpark drafter is kyuz0's GGUF of DeepSeek's draft model.
- **Engine and tools:** [strix-halo-engine-lab](https://github.com/hyukudan/strix-halo-engine-lab).
