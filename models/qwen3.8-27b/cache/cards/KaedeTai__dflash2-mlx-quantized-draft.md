---
license: mit
library_name: mlx
tags:
- mlx
- speculative-decoding
- dflash2
- apple-silicon
- quantization
- benchmark
base_model:
- Qwen/Qwen3.8-27B
---

# DFlash2 on Apple Silicon — quantized-draft loader + M5 Max measurements

Two things this repo provides, neither of which is model weights:

1. **`load_draft_q4.py`** — a drop-in loader that lets the
   [`cfontes/qwen3.8-27b-mlx-dflash2`](https://huggingface.co/cfontes/qwen3.8-27b-mlx-dflash2)
   server and benchmarks use a **quantized** DFlash2 drafter. The upstream loader builds a
   bf16 skeleton and calls `load_weights` directly, so a 4-bit draft fails on shape
   mismatch.
2. **`config.json`** — the draft config that
   [`ProCreations/Qwen3.8-27B-DFlash2-MLXFast-Q4`](https://huggingface.co/ProCreations/Qwen3.8-27B-DFlash2-MLXFast-Q4)
   ships without. That repo contains only `model.safetensors`, so no standard loader can
   instantiate it.

Plus the measurements that motivated both.

## Headline: quantizing the *drafter* is close to free throughput

Apple **M5 Max**, 128 GB, macOS 26.4.1, MLX 0.32.0 / mlx-lm 0.31.3. Target:
`tozp/Qwen3.8-27B-OBLITERATED-V2-mlx-4bit`. Greedy, 300 tokens, K=4, interleaved A/B with a
90 s cooldown before each measurement, two rounds.

| drafter | size | tok/s | accept |
|---|---:|---:|---:|
| `z-lab/Qwen3.8-27B-DFlash2` (bf16) | 3.85 GB | 59.8 / 60.3 | 87.9% |
| `ProCreations/...-MLXFast-Q4` (affine-4 g64) | **1.27 GB** | **76.1 / 76.0** | 86.9% |
| | −2.6 GB | **+26%** | −1.0 pp |

The drafter runs a forward pass every block, so shrinking it by 2.6 GB removes real
bandwidth from the hot loop. The prediction quality it costs — one point of acceptance — is
nowhere near enough to offset that.

Stacked against no speculation at all, on the same target and machine:

| configuration | tok/s | vs dense |
|---|---:|---:|
| dense (`mlx_lm.stream_generate`) | 33.3 | 1.00× |
| + bf16 drafter | 61.4 | 1.84× |
| **+ Q4 drafter** | **76.1** | **2.29×** |

A 27B **dense** model at 76 tok/s in **16.1 GB resident** (measured RSS with the server
loaded and generating; 14 GB target + 1.2 GB drafter on disk, plus KV cache — it grows
with context).

## Target-model comparison (same protocol)

| target | dense | DFlash2 (bf16 draft) | speedup | accept | greedy token-exact vs dense |
|---|---:|---:|---:|---:|---|
| `mlx-community/Qwen3.8-27B-mxfp4` | 35.4 | 54.0 | 1.52× | 82% | yes |
| Qwen3.8-27B abliterated w/ [Heretic](https://github.com/p-e-w/heretic) | 33.3 | 62.2 | 1.87× | 89% | one near-tie flip |
| `tozp/...-OBLITERATED-V2` | 33.3 | 61.4 | 1.84× | 88% | yes |

Round-to-round spread under 2% on every cell.

**The drafter transfers across quantization schemes and weight surgery.** It is trained
against the stock model; both alternatives are abliterated by unrelated methods (Optuna
search vs SVD+LEACE blend) *and* quantized differently (affine-4 g64 vs mxfp4 g32).
Acceptance went **up** 6–7 points rather than down. Two independent methods landing in the
same place suggests the cause is removing refusal directions as such — plausibly lower
next-token entropy, which is testable and untested here.

## Measurement traps on this hardware

- **Thermal drift.** Running targets back-to-back without cooldowns, the same mxfp4
  measurement read 34.9 t/s at the start and 26.8 t/s three minutes later — **−23%**, enough
  to invert a model-vs-model conclusion. Interleaving with 90 s cooldowns took spread from
  23% to under 2%.
- **Cold start.** The first model load of a session measures far below steady state; one run
  produced `dense 1.9 t/s / speedup 18.00x`. Warm up before timing.
- **`ioreg` "Device Utilization %" is useless here** — it reads 100% at rest. Check GPU
  memory in use and whether other engines hold models instead. A 37 GB workload on the GPU
  went undetected this way and produced a full set of wrong numbers.

## Usage

```bash
git clone https://huggingface.co/cfontes/qwen3.8-27b-mlx-dflash2 dflash2
cd dflash2
hf download ProCreations/Qwen3.8-27B-DFlash2-MLXFast-Q4 --local-dir models/draft-q4
cp /path/to/this/repo/config.json          models/draft-q4/config.json
cp /path/to/this/repo/load_draft_q4.py     bench/extra/load_draft_q4.py

# point the server at the quantized draft
python dflash2_mlx_server/server.py --main-dir models/<target> --draft-dir models/draft-q4
```

`server.py` calls `dflash_port.local_load.load_draft_from_dir`; swap that import for
`load_draft_any` from `load_draft_q4.py`. The loader detects a `quantization` block and
falls back to the original function when there isn't one, so bf16 drafts keep working.

Verified end-to-end: server loads in 0.9 s and reports `accept_rate 0.917, tok_per_s 75.4`.

## How the loader works

`nn.quantize` is applied to the draft skeleton **before** `load_weights`, with a predicate
that quantizes exactly the modules that have a `.scales` entry in the weight file — the same
test `mlx_lm` uses. It also normalizes the candidate-selector codebook keys, which the two
published drafts spell differently (`candidate_selector.predecessor_codebook` vs
`...codebook.weight`); the upstream rename raises `KeyError` on the Q4 layout.

## Credits

- [z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) — the DFlash2 drafter
- [cfontes](https://huggingface.co/cfontes/qwen3.8-27b-mlx-dflash2) — the MLX port and server this patches (MIT)
- [ProCreations](https://huggingface.co/ProCreations/Qwen3.8-27B-DFlash2-MLXFast-Q4) — the affine-4 quantization measured here
- [tozp](https://huggingface.co/tozp/Qwen3.8-27B-OBLITERATED-V2-mlx-4bit) and [Heretic](https://github.com/p-e-w/heretic) — the abliterated targets

No model weights are redistributed here. `config.json` is derived from z-lab's draft config
with a quantization block added.
