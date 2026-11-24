---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - nchapman/Qwen3.8-Flash-Next-Swift-1.5-LoRA
  - nazeshinjite/Qwen3.8-Flash-Next-Abliterated-LoRA
library_name: gguf
pipeline_tag: text-generation
tags:
  - gguf
  - qwen4exp
  - q5_k_m
  - abliterated
  - llama.cpp
---

# Qwen3.8-Flash-Next + Swift-1.5 + abliterated — GGUF (Q5_K_M)

GGUF form of the same weights published as a Basalt pack in
[`koldfrontier/qwen38-flash-next-swift15-abliterated-q5km-basalt`](https://huggingface.co/koldfrontier/qwen38-flash-next-swift15-abliterated-q5km-basalt). Use this repository if your runtime reads GGUF
(llama.cpp, LM Studio, ollama); use the other one only if you run the Basalt/Lapilli engine.

| file | bytes | sha256 |
|---|---|---|
| `qwen38-flash-next-swift15-abliterated-q5km-Q5_K_M.gguf` | 118,202,878,144 | `a66a4d7824262fcd64583b54751016be18cc52d1e3969f2832d2c081679c4071` |

## What it is

The base Qwen3.8-Flash-Next checkpoint with two published LoRAs applied — nchapman's Swift-1.5
(`llama-export-lora`) and nazeshinjite's rank-1 abliteration direction — then quantized once.

**Not a plain Q5_K_M.** llama.cpp refuses K-quants on rows whose width is not a multiple of 256, which
here hits the expert down-projections (640 columns) and `per_layer_token_embd` (160 columns). The recipe
pins those explicitly:

```text
blk.<N>.ffn_down_exps.weight   = q5_1      for N in 0..47
blk.<N>.ffn_down_shexp.weight  = q5_1      for N in 0..47
per_layer_token_embd.weight    = iq4_nl
```

Everything else keeps llama.cpp's Q5_K_M choices (gate/up `q5_K`, attention `q6_K`).

## Measured

`basalt-bench --quality 16` against a KL-divergence base from this model's own F16 weights, 16-chunk
wikitext-2 test set. These are the Basalt pack's numbers; the GGUF decodes to the same weights.

| pack | PPL | KLD mean | same top-1 |
|---|---|---|---|
| **this model (down experts q5_1)** | **2.1877** | **0.11384** | **91.52%** |
| same model, down experts q8_0 | 2.1984 | 0.11562 | 91.35% |
| same model, down experts iq4_nl | 2.2040 | 0.12574 | 91.27% |
| F16 reference | 2.1183 | — | — |

## What it is not

- **Abliteration is unbenchmarked.** The direction is a third party's published LoRA; four spot probes
  answered directly with no refusals, and that is the whole behavioural evidence. No HarmBench/XSTest run.
- **Only tested on 2x V100 with the Basalt/Lapilli engine.** The GGUF itself is ordinary; the numbers above
  came from that engine.
- **Not 16-bit.** The F16 source is 354 GB and is not published.

## Licence

The Qwen community licence applies and permits redistribution of derivatives with attribution; the
`base_model` field and `NOTICE` carry it. The two LoRAs keep their own licences.
