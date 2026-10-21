---
license: gemma
base_model: google/gemma-4-26B-A4B-it
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - moe
  - quantization
  - imatrix
  - asymmetric-quant
  - larger-than-ram
  - streaming-inference
---

# gemma-4-26B-A4B-it   |>   Asymmetric 2-bit Routed-Expert Quant

A GGUF quantization of `gemma-4-26B-A4B-it` built for **larger-than-RAM /
SSD-streaming inference**. Instead of quantizing every tensor to the same width,
it quantizes **asymmetrically**: the routed-expert weights (≈89% of the bytes
moved per token in a Mixture-of-Experts model) are pushed to ~2 bits, while
everything quality-sensitive (attention, the router, shared FFN, embeddings and
norms) stays at Q8_0.

The result matches the practical quality of the 4-bit baseline while being ~33%
smaller on disk and up to ~1.5× faster when streaming from SSD.

## Why this exists — intelligence that fits on basic hardware

This quant is built for **[Fono](https://github.com/bogdanr/fono)**, whose aim is to
run the most capable model that will actually fit on ordinary, memory-limited
hardware. The guiding principle is *intelligence per gigabyte*: spend precision only
where the model is genuinely sensitive, and compress everything else hard. That lets
a larger, smarter model drop into the same memory budget, so a modest machine gets
to punch well above its weight instead of settling for a smaller, weaker model.

## What "asymmetric quant" means here

A dense uniform quant (e.g. Q4_0) spends the same bits everywhere. But in an MoE,
decode speed under memory pressure is dominated by how many **routed-expert**
bytes must be pulled from SSD per token. So this recipe:

| Tensor group | Precision | Why |
|---|---|---|
| Routed expert `ffn_gate_up_exps` | **IQ2_XXS** (~2.06 bpw, imatrix) | ~89% of streamed mass gets compressed |
| Routed expert `ffn_down_exps` | **Q2_K** (imatrix), inner dim zero-padded 704→768 | smart 2-bit, no crude fallback |
| Attention / router / shared FFN / embeddings / norms | **Q8_0** | near-lossless where the model is sensitive |

Two design details make the aggressive part safe:
1. **imatrix-aware i-quants only.** An importance matrix steers the 2-bit codebook
   toward the weights that matter most. It was computed with `llama-imatrix` over
   Bartowski's `calibration_datav3` (generic web/wiki text, deliberately **disjoint**
   from our benchmark tasks so scores aren't inflated), 100 chunks × 512 tokens.
   Crude legacy 2-bit (`Q2_0`) at the same bit budget destroys the model (0–3/10 on
   coding); the imatrix-guided i-quant keeps 9/10.
2. **Zero-pad to 256-alignment.** Gemma's routed inner dim (704) isn't divisible by
   256, which normally forces a crude fallback. Padding to 768 with zeros is
   numerically exact (`gelu(0)·0 = 0`) and unlocks Q2_K on `down`.

## Why you might use it

There are **two** distinct wins:

1. **It fits in RAM where the full-size model won't.** At 9.60 GB the asymmetric
   quant can sit entirely in memory on a machine where the 14.4 GB baseline would
   have to spill to SSD (or fail to load). Running fully in RAM is far faster than
   streaming — so on that hardware the smaller-but-equally-smart model simply *is*
   the faster model, and you get 26B-class capability where you otherwise couldn't.
2. **When it must stream, it streams less.** Past the RAM budget it moves ~45% fewer
   cold bytes per token than the uniform 4-bit build, so decode stays quick.

Other reasons: the smallest gemma-4-26B GGUF that still passes real coding tasks, and
low **bytes-moved-per-token** rather than just a small file.

(If the *full-size* model already fits comfortably in your RAM, a uniform Q4_0 is
marginally faster there — fewer dequant ops. The asymmetric quant's advantage is
exactly the common case where it fits and the bigger one doesn't.)

## Measured results

Coding  = run-the-code pass rate (10 objective tasks)
polish  = instruction-formatting score
cold@6G = cold bytes streamed from SSD per token at a 6 GB RAM cap
t/s     = decode tokens/sec.

| variant | size | coding | polish | cold@6G | t/s@6G | t/s@max |
|---|---:|:---:|:---:|---:|---:|---:|
| QAT Q4_0 (uniform baseline) | 14.4 GB | 10/10 | 0.981 | 799 MB | 6.8 | 19.5 |
| **asym (this model)** | **9.60 GB** | **9/10** | **0.969** | **439 MB** | **10.5** | 16.1 |

vs the uniform 4-bit baseline at a 6 GB cap: **−45% cold bytes, ~1.5× decode
speed, −33% on disk**, at essentially intact quality.

## Files

- `gemma-4-26B-A4B-it-asym.gguf` (9.60 GB)

## Usage (llama.cpp)

```bash
llama-cli -m gemma-4-26B-A4B-it-asym.gguf -p "Write a prime sieve in Rust."
```

## Limitations

- ~2-bit routed experts trade a little robustness on the hardest reasoning/parsing
  tasks (one hard Rust task lost vs the 10/10 baseline). It was a small syntax error
  that prevented compilation. For general purpose tasks it _feels_ the same.
- Inherits all limitations and the acceptable-use policy of the base Gemma model.

