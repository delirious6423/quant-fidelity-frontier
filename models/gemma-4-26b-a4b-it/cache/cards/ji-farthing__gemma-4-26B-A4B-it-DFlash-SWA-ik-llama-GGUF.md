---
license: apache-2.0
base_model:
  - z-lab/gemma-4-26B-A4B-it-DFlash
  - google/gemma-4-26B-A4B-it
tags:
  - gguf
  - gemma4
  - dflash
  - speculative-decoding
  - sliding-window-attention
  - ik_llama
---

# Gemma 4 26B-A4B DFlash SWA draft for ik_llama

This repo contains an `ik_llama`-compatible DFlash draft GGUF converted from `z-lab/gemma-4-26B-A4B-it-DFlash`, carrying the per-layer sliding-window attention (SWA) pattern.

This is not a standalone chat model. Use it as a `--model-draft` file next to a matching Gemma 4 26B-A4B IT target GGUF, with DFlash speculative decoding. Gemma 4 needs `--jinja`.

## Sliding-window attention

The draft is sliding-window on every layer except a final full-attention (global) layer: `sliding_window_pattern = [true, true, true, true, false]`, `sliding_window = 2048`.

## Files

| File | Quant | Draft window |
| --- | --- | --- |
| `gemma-4-26B-A4B-it-DFlash-SWA-ik_llama-Q8_0.gguf` | Q8_0 | 2048 (4 sliding + 1 global) |

## Use

```bash
llama-server \
  -m /path/to/gemma-4-26B-A4B-it-<quant>.gguf \
  --model-draft /path/to/gemma-4-26B-A4B-it-DFlash-SWA-ik_llama-Q8_0.gguf \
  --spec-type dflash:n_max=4,cross_ctx=8192 \
  -c 8192 --jinja
```

SWA only engages once the DFlash cross-context exceeds the 2048 window, so set `cross_ctx` above the window for long-context prompts (the default 512 does not grow with `-c`).

## Validation (RTX 4070, ik_llama DFlash SWA branch)

Draft acceptance and throughput versus the same draft run with full attention, as the prompt overflows the 2048 window, where `clip = (prompt - 2048) / prompt`:

| prompt tok | clip | accept, full-attn | accept, SWA | acceptance gain | tok/s change |
| --- | --- | --- | --- | --- | --- |
| 1413 | 0% | 14.4% | 15.2% | +0.7 pp | +1.9% |
| 2751 | 26% | 17.1% | 17.1% | +0.0 pp | -1.1% |
| 3923 | 48% | 12.0% | 15.9% | +3.9 pp | +7.9% |
| 5426 | 62% | 13.8% | 14.7% | +0.9 pp | +0.5% |
| 10673 | 81% | 10.6% | 18.1% | +7.5 pp | +16% |

## Conversion

Converted from `z-lab/gemma-4-26B-A4B-it-DFlash` with `ik_llama`'s `convert_hf_to_gguf.py` DFlash draft converter (sliding-window support branch), then quantized to Q8_0. The per-layer SWA pattern is taken from the source `layer_types`. Conversion requires a `--target-model-dir` containing the target tokenizer merges.
