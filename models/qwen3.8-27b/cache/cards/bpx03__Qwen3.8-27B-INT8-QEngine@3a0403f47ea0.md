---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- qengine
- qwen3.8
- int8
---

# Qwen3.8-27B INT8 for QEngine

Qwen3.8-27B with INT8 layer weights, for [QEngine](https://github.com/bpx03/QEngine) 1.3.5.2 or later, a source-available
inference engine for 3× NVIDIA RTX 3090. This is an experimental option. QEngine's default for this model is still INT4.

These weights are closer to the original BF16 model than INT4 and they run slower, because every decode step reads twice
the weight bytes.

```bash
hf download bpx03/Qwen3.8-27B-INT8-QEngine --local-dir models/Qwen3.8-27B-INT8
server/run.sh 8080 --model models/Qwen3.8-27B-INT8     # Docker: -e QE_MODEL=/app/models/Qwen3.8-27B-INT8
```

Three cards are what I'd use. On 2 cards it needed 16 slots or fewer when I measured it, and one card can't hold the weights.

## What it costs and what it gives

3× RTX 3090. The INT8 KL and top-1 are from QEngine 1.3.5.2. The INT4 column and all speed and KV pool numbers are from November 1, 2026, on older code, same day and same build for both, and I haven't rerun those:

| metric | INT8 | INT4 | INT8 / INT4 |
|---|---:|---:|---:|
| KL vs BF16, mean (320 positions) | 6.66e-4 | 1.47e-2 | 22× closer |
| top-1 match with BF16 | 99.7 % | 95.3 % | |
| decode narrative, tok/s | 184.6 | 266.6 | 0.69 |
| decode code, tok/s | 276.7 | 387.7 | 0.71 |
| prefill 10K, tok/s | 4,816 | 6,077 | 0.79 |
| prefill 90K, tok/s | 4,098 | 4,866 | 0.84 |
| agent c=8, output tok/s | 810 | 1,215 | 0.67 |
| KV pool, 32 slots, 262K context, tokens | 566,272 | 932,864 | 0.61 |

I don't know yet whether the lower KL gives better answers in real tasks, because the 8-pack quality suite has never run
on these weights. On QEngine 1.3.5.2 they passed quick checks on 3 cards: batched output equal to single-request output with the KV pool under pressure, KV reuse on a follow-up turn, and correct replies in bursts of 8 parallel requests.

## How it was made

`tools/quant_int8.py` from the QEngine repo, from the official BF16 weights: symmetric INT8, group 128, no calibration
data. The FP8 KV scales come from QEngine's INT4 checkpoint, and the MTP head
(`model_mtp.safetensors`) is the stock BF16 one. The README of the repo shows how to build it yourself.

`quantization_config` in `config.json` only describes the files. QEngine reads the formats from the tensors, and these
files are laid out for QEngine.

## License

Apache 2.0, as the original [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
