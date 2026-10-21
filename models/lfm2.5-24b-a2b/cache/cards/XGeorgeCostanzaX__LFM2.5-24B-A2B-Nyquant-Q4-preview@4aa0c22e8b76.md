---
library_name: gguf
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
pipeline_tag: text-generation
tags:
- gguf
- llama.cpp
- quantized
---

# LFM2.5-24B-A2B — Nyquant Q4 (early preview)

GGUF quantizations of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), made with an importance-matrix, per-tensor bit allocation: the routed experts at ~4 bits, the more sensitive tensors (attention, short-conv, dense FFN, embeddings) at higher precision.

This is an early preview. It is an independent derivative and is not affiliated with or endorsed by Liquid AI. See [NOTICE](NOTICE) and [LICENSE](LICENSE).

## Files

| File | Size | Bits per weight |
|---|---:|---:|
| `LFM2.5-24B-A2B-mix-ar4match-preview.gguf` | 13.87 GB | 4.63 |
| `LFM2.5-24B-A2B-mix-iq4xs-preview.gguf` | 13.08 GB | 4.37 |
| `LFM2.5-24B-A2B-mix-F12000-preview.gguf` | 12.00 GB | 4.00 |

## Quality

KL divergence from the BF16 model (lower is better) and perplexity, measured on held-out chat, web and code text: 60 windows of 2,048 tokens per corpus, tokens 1,024-2,046 scored, 95 % bootstrap intervals.

Measurements are still running; the table will be filled in as they complete.

| Model | Size | KLD vs BF16 (mean of 3 corpora) [95 % CI] | PPL chat | PPL web | PPL code |
|---|---:|---|---:|---:|---:|
| BF16 (reference) | 47.96 GB | 0 | pending | pending | pending |
| **mix-ar4match** (this repo) | 13.87 GB | pending | pending | pending | pending |
| **mix-iq4xs** (this repo) | 13.08 GB | pending | pending | pending | pending |
| **mix-F12000** (this repo) | 12.00 GB | pending | pending | pending | pending |
| Q4_K_M | 14.53 GB | pending | pending | pending | pending |
| QAD-Q4_0 (official) | 13.58 GB | pending | pending | pending | pending |

Tool calling (36 greedy test cases vs BF16): pending.

## Usage

```bash
# llama.cpp
llama-cli -m LFM2.5-24B-A2B-mix-ar4match-preview.gguf -ngl 99 -c 8192 -cnv
llama-server -m LFM2.5-24B-A2B-mix-ar4match-preview.gguf -ngl 99 -c 32768
```

Use the chat template embedded in the GGUF.

## License

The [LFM Open License v1.0](LICENSE). These files are modified versions of the original weights; see [NOTICE](NOTICE).
