---
license: other
license_name: deepseek
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- deepseek
- deepseek-v4.1-flash
- gguf
- iq1_s
- quantization
- strix-halo
- diy-quant
library_name: gguf
---

# DeepSeek-V4.1-Flash-1017 DIY IQ1_S (v3bu)

An independent, fully reproducible IQ1_S-dominant dynamic GGUF quantization (~1.77 bpw) of
**DeepSeek-V4.1-Flash-1017** (552B backbone + 196B Engram), built end-to-end
from the official public weights on a single 128 GB RAM machine
(AMD Ryzen AI Max+ 395 / Strix Halo, Ubuntu).

**Backbone: 121.8 GB in 5 shards (~1.77 bpw)**

Type mix (per-tensor, like other dynamic quants): the routed experts are mostly
IQ1_S (88 tensors) with the more sensitive expert layers at IQ2_XXS (32); the
active path (attention, shared experts, routers) is kept at Q5_K\u2013Q8_0/F32. — designed to run fully resident in 128 GB RAM,
with the Engram conditional-memory tensor kept as a separate lazy-loaded SSD shard.

## Why this build exists

- Full pipeline transparency: every step (download → convert → extract → quantize → assemble → evaluate) is scripted and published.
- Measured, not guessed: quality is quantified with KL divergence against a Q8 reference, with a self-vs-self control at 0.000.
- A documented independent reproduction attempt of the official `UD-IQ1_S` layout, plus an investigation of what makes the official release better (see the build passport).

## Quality (our measurements)

| Metric (wiki1100, vs Q8_K_XL reference) | Value |
|---|---|
| KLD mean / median | 0.612 / 0.265 |
| PPL ratio vs reference | 1.485 |
| Top-1 agreement with official UD-IQ1_S (model vs model) | 75.3% |
| Control: self vs self | 0.000 |

For comparison, the official `UD-IQ1_S` release measured by the same pipeline:
KLD 0.414 mean / 0.155 median, PPL ratio 1.156, top-1 77.1%.
The remaining gap traces to the weight source (public FP8 vs internal
higher-precision weights), not to the quantization layout or tooling —
details in the [build passport](PASSPORT.md).

## Layout

- `DS41-DIY-v3bu-00001..00005-of-00006.gguf` — the quantized backbone (this repo)
- Shard 6 (the Engram table, `per_layer_token_embd`, IQ4_NL, 110.6 GB) is **not re-hosted**:
  take it from [`unsloth/DeepSeek-V4.1-Flash-1017-GGUF`](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`engram/*IQ4_NL*`) and stamp it as shard 6 — see below.

## How to run (llama.cpp, 128 GB APU)

```bash
# 1) fetch the engram shard from unsloth's repo
hf download unsloth/DeepSeek-V4.1-Flash-1017-GGUF --include "engram/*IQ4_NL*" "engram/set_split.py" --local-dir .

# 2) stamp it as shard 6 of 6
python3 engram/set_split.py engram/DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file 6 6
mv engram/DeepSeek-V4.1-Flash-1017-engram-IQ4_NL.gguf_file DeepSeek-V4.1-Flash-1017-DIY-IQ1_S-00006-of-00006.gguf

# 3) run (requires llama.cpp with DeepSeek-V4.1 support, e.g. PR #30817 branch or newer master)
llama-server -m DeepSeek-V4.1-Flash-1017-DIY-IQ1_S-00001-of-00006.gguf -ngl 99 -c 32768 \
  --temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0 \
  --load-mode mmap --lazy-mode on \
  --override-tensor 'per_layer_token_embd.weight=CPU' \
  --override-tensor 'blk\.([0-9]+)\.ffn_(up|down|gate)_exps\.weight=CPU'
```

The two `--override-tensor` flags are essential on 128 GB unified-memory APUs:
the first keeps the 110 GB engram table out of the GPU allocator (lazy SSD streaming),
the second keeps the MoE experts on the CPU side so the backbone fits GTT.

Sampling note: like other 1-bit builds, use `--temp 1.0 --top-p 1.0 --top-k 0 --min-p 0.0`;
lower temperatures can degrade into repetition loops.

Performance (Ryzen AI Max+ 395): ~1 min model load, ~5–6 tok/s generation,
backbone fully RAM-resident, 32k context.

## Reproducibility

The complete pipeline (all scripts, from corpus download to KLD evaluation)
and the llama.cpp patch branch are described in the [build passport](PASSPORT.md).

## Attribution & License

- Base model and license: [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
- The official quantization layout this build borrows: [unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (Unsloth Dynamic)
- Quantizer: llama.cpp `llama-quantize` (Unsloth build 11876)

This build is a derivative work of the base model and is distributed under the
base model's license terms.
