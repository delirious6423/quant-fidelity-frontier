---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
library_name: gguf
tags:
  - gguf
  - gemma4
  - text-generation
  - conversational
  - text-only
  - endpoints_compatible
pipeline_tag: text-generation
---

# gemma-4-26B-A4B-it-OnlyText-GGUF

GGUF quants of [OnlyTextLLMs/gemma-4-26B-A4B-it-OnlyText](https://huggingface.co/OnlyTextLLMs/gemma-4-26B-A4B-it-OnlyText)
— the text-only derivative of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it). The file is the text
backbone alone: **no draft head is embedded**, so there is no in-file
speculative decoding — plain decoding is what the numbers below measure.

## Quant files

| File | Size |
|---|---|
| `gemma-4-26B-A4B-it-OnlyText-Q4_K_M.gguf` | 16.8 GB |
| `gemma-4-26B-A4B-it-OnlyText-Q6_K.gguf` | 22.6 GB |
| `gemma-4-26B-A4B-it-OnlyText-Q8_0.gguf` | 26.9 GB |

### Quantization damage (vs the F16 master)

llama-perplexity --kl-divergence against the F16 master's logits (ctx 1024, 40 windows, 20440 scored tokens). `Same top-1` is the share of positions where the quant keeps the F16
file's top token; `KLD` is the divergence of the full distribution, whose median is
the tail-free summary (a mean over a heavy-tailed loss says more about the tail than
about the quant).

| Quant | Mean KLD | Median KLD | Same top-1 | Max KLD |
|---|---|---|---|---|
| `Q4_K_M` | 3.1975 | 1.5861 | 45.42% | 45.66 |
| `Q6_K` | 1.1702 | 0.1979 | 68.00% | 36.12 |
| `Q8_0` | 0.8199 | 0.0890 | 73.17% | 30.30 |

**`Q4_K_M` is not interchangeable with the others here.** It changes the
top-1 token on 54.6% of positions, against 26.8% for `Q8_0`, so choose from the `Same top-1` column
rather than by size alone.

## Benchmarks

Measured 2026-09-24 on a single AMD Radeon AI PRO R9700 (gfx1201, 34 GB) with
llama.cpp `9d286e1b315c` (HIP/ROCm build, ROCm 7.14, `-c 32768 -fa on`). Baseline =
llama-bench PP512/1024/2048 + TG128/256; chat decode = llama-cli single-turn,
256 generated tokens, median conditions, `-np 1`.

| Quant | Device | Prefill t/s (PP512) | Gen t/s chat | Gen t/s llama-bench TG128 |
|---|---|---|---|---|
| Q4_K_M | single R9700 | 2585 | 83.1 | 93.6 |
| Q6_K | single R9700 | 2071 | 80.0 | 87.5 |
| Q8_0 | single R9700 | 2774 | 75.0 | 82.1 |

The two decode columns are different measurements (chat runs use the chat template
and a ~90-token prompt; TG128 is a bare 128-token generation) — do not compare them
against each other.



No published baselines are cited for this size — the numbers above are ours.

## Usage

Requires a llama.cpp build that knows the `gemma4` architecture.

```bash
# llama-server (OpenAI-compatible)
llama-server -m gemma-4-26B-A4B-it-OnlyText-Q4_K_M.gguf -ngl 99 -c 32768 -fa on

# llama-cli single-turn chat
llama-cli -m gemma-4-26B-A4B-it-OnlyText-Q4_K_M.gguf -ngl 99 -c 32768 -fa on \
    -st -p "Your prompt"
```

Notes:

- The chat template supports Gemma's thinking channel but defaults `enable_thinking`
to false, so plain chat answers directly. Pass
`"chat_template_kwargs": {"enable_thinking": true}` (server API) to let it
reason first.
- Dual-GPU: unset `*_VISIBLE_DEVICES` env pins; llama.cpp auto layer-splits
  (`--split-mode layer` is the default).

## Details

- Base: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) (apache-2.0): 25.23B params, 30 layers (25 sliding-attention + 5 full-attention), hidden 2816, vocab 262137, 128 experts (top-8), `Gemma4ForCausalLM`, `gemma4` GGUF architecture.
- Tokenizer: 17 text-only special tokens at ids 0–106 (EOS 1 = `<eos>`, turn end 106 = `<turn|>`); image/audio/video tokens removed.
- Conversion: `convert_hf_to_gguf.py` (llama.cpp `9d286e1b315c`) then
  `llama-quantize` with the file's named type; no draft head is embedded (the family ships none).
- Weights are untouched derivatives — no additional training.

## Attribution

This model is a derivative of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) by the
[Google team](https://huggingface.co/google), released under the
[apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
All credit for the underlying weights and capabilities belongs to the original
authors; this repository only removes modalities and quantizes, it does not add
new training.

## Accuracy against the source checkpoint

Scored per position — this file's logits against the source checkpoint at matching
precision, over the same 1533 tokens (ctx 1024, 3 windows of wikitext-2) — the two agree
on the top-1 token at **86.2%** of positions, median KLD
**0.009**. Both sides hold the same weights in the same precision
there, so that gap is llama.cpp against transformers, not quantization: on this family's
out-of-distribution corpus (untemplated prose, instruct-only model) the top-1 token sits
near a tie at many positions, and any numerical difference — kernel, dtype, device —
flips it, while the median says the two distributions are otherwise nearly identical.

Quantization costs a further **7.0 points** of top-1 agreement
(Q8_0 79.2%, measured the same way), which is the axis
the table above is for.

Reference conditions: this member's F16 (50.5 GB) does not fit one 32 GB card,
so its reference splits across both R9700s. Forcing a single card spills tensors
to CPU and is numerically equivalent (99.87% same top-1, mean KLD 0.000045) but
~22x slower.

