---
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: exllamav3
tags:
- exl3
- exllamav3
- quantized
- lfm2
- moe
---

# LFM2.5-24B-A2B, EXL3 3.25 bpw (`-hq`, 8-bit head)

An [ExLlamaV3](https://github.com/turboderp-org/exllamav3) quant of
[LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) (24B mixture-of-experts,
~2.4B active), sized so the model **runs entirely on one 12 GB GPU with a 128K context**.

| | |
|---|---|
| Size | 10.59 GB |
| Bits | 3.25 bpw average over the decoder, `-hq` (more bits for the attention and shared layers), lm_head 8 bits, embeddings unquantized |
| Bit allocation | 2 dense layers 4.2 bpw, 18 MoE layers ~3.55, 20 MoE layers ~3.07, lm_head 8.0 |
| Calibration | exllamav3's bundled corpus mix (a self-sampled trace was tried and did not do better, below) |
| Made with | exllamav3 1.6.1, `convert.py -b 3.25 -hq -hb 8` |
| Runs on | exllamav3 ≥ 1.6.1; **≥ 1.6.2 for a quantized KV cache** (1.6.1 has a quantized-cache decode bug with this architecture) |

## Quality vs the BF16 model

qbench (exllamav3 `eval/`), 40 × 2048 tokens of wikitext-2 with the chat template loosely applied, reference =
the BF16 source:

| Model | Perplexity | KLD mean | KLD median | KLD p90 |
|---|---|---|---|---|
| BF16 | 11.2459 | – | – | – |
| BF16 + rounding noise (noise floor) | 11.2530 | 0.0080 | 0.0024 | 0.0184 |
| **This quant (3.25 bpw)** | **11.2959 (+0.44%)** | **0.0918** | **0.0464** | **0.2071** |
| 6.0 bpw `-hq` (not published, for scale) | 11.2469 | 0.0162 | 0.0066 | 0.0369 |
| 3.25 bpw calibrated on a self-sampled trace (not published) | 11.3239 | 0.0998 | 0.0510 | 0.2278 |

Where the BF16 model is confident (top token ≥ 95%, 19% of tokens), this quant's median KLD is 0.0014, at
the noise floor. The divergence sits in tokens the BF16 model was itself unsure about. Wikitext is out of
domain for this chat model, so absolute KLD runs high; use the numbers to compare quants of this model.

Agentic quality (tool-eval-bench hard mode, 97 scenarios): **82 / 100**, also 82 with 4 scenarios run at
once. The 6.0 bpw reference scored 76 and the trace-calibrated 3.25 bpw 80: at this level single runs vary
by about ±5 points, so the quant costs no measurable agent quality. Needle in a haystack: 100% (20/20) to 64K.

## On a 12 GB card (RTX 3060)

Measured in TabbyAPI on an RTX 3060 (12 GB):

| Setting | Result |
|---|---|
| 128K context, KV cache 8,4 (K 8-bit, V 4-bit), 4 requests at once | 11.0 GB after load and warmup, 11.4 GB peak |
| Decode, one request | ~58-60 tok/s (barely drops with context: only 10 of 40 layers keep a KV cache) |
| Prefill | ~1,600-2,100 tok/s from 1K to 64K |
| 4 requests at once (~1K prompts) | ~108 tok/s in total, ~30 per request |
| 8 requests sharing the same 128K pool | ~137 tok/s in total; requests beyond the pool wait and then run, no out-of-memory |

Example TabbyAPI `config.yml` (model section):

```yaml
model:
  model_dir: /path/to/models
  model_name: LFM2.5-24B-A2B-exl3-3.25bpw
  backend: exllamav3
  max_seq_len: 131072
  cache_size: 131072       # one pool shared by all requests
  cache_mode: 8,4          # needs exllamav3 >= 1.6.2
  chunk_size: 2048
  max_batch_size: 4
  reasoning: true          # LFM2.5 always thinks: <think>...</think> -> reasoning_content
  tool_format: lfm2        # Pythonic calls between <|tool_call_start|> / <|tool_call_end|>
```

Use the sampling LiquidAI recommends for this model: temperature 0.1, top_k 50, repetition penalty 1.05.

## License

The original model is licensed under the **LFM Open License v1.0** ([`LICENSE`](LICENSE), copied unchanged
from the source repository). This repository contains a **modified (quantized) version** of the original
weights; the tokenizer, chat template and configuration files are copied from the source, with
`config.json` extended by exllamav3's quantization settings. Under section 5 of the license, commercial use by
an entity with annual revenue of US$10 million or more is not licensed. See the license for the full terms.

## Credits

[Liquid AI](https://huggingface.co/LiquidAI) for LFM2.5-24B-A2B, and
[turboderp](https://github.com/turboderp-org/exllamav3) for ExLlamaV3.
