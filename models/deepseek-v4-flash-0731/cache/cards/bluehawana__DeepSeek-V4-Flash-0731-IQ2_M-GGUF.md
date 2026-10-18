---
license: mit
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
quantized_by: AtomicChat
tags:
  - gguf
  - ollama
  - apple-silicon
  - mac
  - deepseek
---

# DeepSeek-V4-Flash-0731 · IQ2_M · single-file GGUF for Ollama

Single-file IQ2_M quant of DeepSeek-V4-Flash-0731 (284B MoE, 13B active per
token, MIT) merged from [AtomicChat's sharded GGUFs](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
so Ollama can pull it. ~33 tok/s on a 128 GB Apple Silicon Mac. Model by
DeepSeek, quantization by AtomicChat — we only merged and documented.

**Run it on a Mac with one command — no cloning, no merging:**

```bash
ollama run hf.co/bluehawana/DeepSeek-V4-Flash-0731-IQ2_M-GGUF
```

Or from the Ollama registry:

```bash
ollama run bluehawana/deepseek-v4-flash:iq2_m
```

~104 GB download. Tested on a MacBook Pro M5 Max (128 GB unified memory):
loads in ~25 s.

## Why this repo

The excellent [AtomicChat quants](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF)
are published as **sharded** GGUFs, which Ollama can't pull from the registry —
and (as of Ollama 0.32.12) `ollama create` fails on them locally too
(`split GGUF ... has 1 shards, expected 4`). This repo is the same IQ2_M quant
merged into a single file with llama.cpp's `llama-gguf-split --merge`, so
`ollama run hf.co/...` just works.

Scripts, the do-it-yourself recipe (works for any sharded GGUF, any quant),
and Mac-specific tips:
**https://github.com/bluehawana/ollama-deepseek-v4-flash-iq2_m-mac-m5max**

## Benchmark (MacBook Pro M5 Max, 128 GB)

![Benchmark: ~33 tok/s generation, 545 tok/s prefill, 3.9 s warm load](https://huggingface.co/bluehawana/DeepSeek-V4-Flash-0731-IQ2_M-GGUF/resolve/main/benchmark.svg)

| Run                 | Prompt tok | Gen tok | Prefill tok/s | Gen tok/s | Total  |
|---------------------|-----------:|--------:|--------------:|----------:|-------:|
| Short answer        |         15 |     128 |          51.9 |  **33.9** |  4.2 s |
| Short story         |         18 |     256 |          59.4 |  **33.6** |  8.0 s |
| Essay               |         25 |     512 |          64.8 |  **33.3** | 15.8 s |
| Long-prompt summary |      2,774 |      64 |     **544.9** |  **32.4** |  7.2 s |

Measured via Ollama's `/api/generate` (its own `eval_count`/`eval_duration`
fields), one pass per row, reasoning tokens included. These speed numbers are
our own measurements of this merged file; all *quality* metrics cited in this
card are AtomicChat's.

## Notes

- Reasoning model — expect `<think>…</think>` before answers.
- On 128 GB Macs, keep context modest (`/set parameter num_ctx 8192`) —
  103 GB of weights leaves little KV-cache headroom.

## Credits — this is not our model

We did **not** train or quantize this model. This repo's only contribution is
merging AtomicChat's four GGUF shards into one file (llama.cpp
`llama-gguf-split --merge`, a byte-preserving operation) so Ollama can pull it,
plus Mac-specific documentation. All quality metrics quoted anywhere in this
project (perplexity, KL divergence, 83.6% top-1 match) are AtomicChat's own
measurements from their model card.

- Discovery: [@rohanpaul_ai's post on X](https://x.com/rohanpaul_ai/status/2084458010307629118)
- Base model: [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) (MIT)
- Quantization + imatrix: [AtomicChat/DeepSeek-V4-Flash-0731-GGUF](https://huggingface.co/AtomicChat/DeepSeek-V4-Flash-0731-GGUF) (MIT) — all credit for the quant work goes to them; this is a 1:1 merge of their shards.
