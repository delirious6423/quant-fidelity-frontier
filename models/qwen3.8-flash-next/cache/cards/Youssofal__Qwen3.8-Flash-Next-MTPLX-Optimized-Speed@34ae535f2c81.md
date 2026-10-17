---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: mtplx
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
tags:
- mlx
- apple-silicon
- macos
- speculative-decoding
- multi-token-prediction
- qwen
- qwen3.8-flash-next
- moe
- mtp
- mtplx
- local-ai
- chat
- qwen3.8
- qwen3-8
- qwen-3.8
- local-llm
- llm
- vision
- m5
- m5-max
- m4
- m3
- macbook-pro
- mac-studio
- opencode
- claude-code
- flash-next
- qwen3-8-flash-next
- qwen4
- 125b
---
**[MTPLX](https://mtplx.com): the fastest way to run Qwen 3.8 on a Mac. Native multi-token-prediction speculative decoding on Apple Silicon, two to three times the speed of plain decoding, exact at any temperature.**

# Qwen 3.8 Flash-Next Optimized Speed

**Dynamic 4-bit quant with 8-bit attention. Higher quality and slightly slower. Recommended.**

Qwen's 125B-A6B Flash-Next preview, the Qwen4-generation architecture with
GDN hybrid MoE, Qwen Sparse Attention, and the 51B-parameter n-gram memory, running natively on [MTPLX](https://mtplx.com) from day 0, with its native
multi-token-prediction head drafting through MTPLX's speculative path. This is
the recommended build: the Qwen Sparse Attention projections are kept at
8-bit, so the attention pathway that steers long contexts keeps its
precision. For the absolute fastest build, pick
[Bare Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed).

The 32 GB n-gram table streams from SSD, so only the weights stay in memory.
The pack peaks at 87 GiB while serving, so it is recommended for Macs with
128 GB or more. On a 96 GB Mac, use
[Bare Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed),
which peaks at 78 GiB.

## Faster prompts in MTPLX 2.12.0 (22 September 2026)

The same Mac, pack and prompts on both versions: MacBook Pro M5 Max with 128 GB,
one model loaded at a time, boots in the order 2.11.3, 2.12.0, 2.12.0, 2.11.3,
thinking off, 512-token answers, the model's own sampler, GPU memory limit
raised to 120 GiB.

| Flash-Next Optimized Speed | MTPLX 2.11.3 | MTPLX 2.12.0 |
|---|---:|---:|
| Time to the first token, 4,061-token prompt | 5.26 s | 2.88 s |
| Prompt processing, 4,061-token prompt | 786 tok/s | 1,453 tok/s |
| Time to the first token, 65,502-token prompt | 85.5 s | 60.1 s |
| Prompt processing, 65,502-token prompt | 768 tok/s | 1,094 tok/s |
| Decode after a 65,502-token prompt | 56.1 tok/s | 63.5 tok/s |

New prompt kernels added after this run read a 65,529-token prompt at
1,548 tok/s. [Release notes](https://mtplx.com/releases/2.12.0/)

## Measured on MTPLX 2.11.3 (16 September 2026)

This is the Qwen3.8-Flash-Next MLX pack for MTPLX, the fastest way to run Qwen 3.8 Flash Next on a Mac. MacBook Pro M5 Max with 128 GB, fans verified at maximum, sampled at the model's own settings (temperature 1.0, top-p 0.95, top-k 20). Conditions and sources for every row: [mtplx.com/benchmarks](https://mtplx.com/benchmarks/).

| Run | tok/s |
|---|---|
| One OpenCode request on the Optimized Speed pack: 1,301 tokens generated, 18,539-token prompt with 18,364 tokens served from cache, MTP depth 3 | 125.8 |
| 9k-token code prompt, 1,500 tokens generated, seeded sampler, thinking off (62.5 on MTPLX 2.11.2) | 79.3 |
| 109k-token OpenCode turn, mean of two runs | 61.8 |
| 200k-token OpenCode turn, warm, mean of two runs | 50.3 |
| Full 45k to 56k-token generations (Flappy Bird at effort xhigh), whole turn | 66.8 |

Exactness on this release: a thousand four-token draws from the fast path match a thousand from the plain path within the plain path's own noise at every joint length, at temperature 1, top-p 0.95, top-k 20. A 96,760-token conversation restored from the session cache in 8 ms. 261,120-token prompts decode. Details: [MTPLX 2.11.3 release notes](https://mtplx.com/releases/2.11.3/).

Recommended for Apple Silicon Macs with 128 GB of unified memory or more: MacBook Pro M4 Max and M5 Max with 128 GB, and Mac Studio M3 Ultra, M4 Max and M5 Max with 128 GB or more. From MTPLX 2.12.0 it is the first recommendation on Macs with 256 GB or more. Guide: [Run Qwen 3.8 Flash Next on a Mac](https://mtplx.com/models/qwen3.8-flash-next/).

## Speeds

Measured on an M5 Max, fans verified at max, single stream, real server
(`mtplx serve`), official Qwen 3.8 sampling (temperature 1.0, top-p 0.95,
top-k 20, sampled, not greedy).

| Run | tok/s |
|---|---|
| Coding task, MTP speculative decode (the default) | 73.5 |
| Same task, plain autoregressive | 43.8 |

That is a 1.7x speculative multiplier through the product serve path, on
sampled output that follows the model's own distribution.

## How it is built

- MoE experts and dense matrices at 4-bit with 64-weight groups; the Qwen
  Sparse Attention projections promoted to 8-bit (the quality edge over
  Bare Speed).
- The GDN convolution and recurrent-state parameters, every norm, the QSA
  indexer, and the MTP head stay 16-bit.
- The n-gram embedding table ships as a separate `ngram-table.safetensors`
  sidecar that MTPLX streams from SSD on every Mac. The vision tower is
  preserved in the weights.

| | |
|---|---|
| Download | 115.1 GB (includes the 32 GB n-gram table) |
| Resident weights (n-gram on SSD) | ~83 GB + working set |
| Peak while serving | 87 GiB |
| Recommended Macs | 128 GB+ unified memory (on 96 GB, use Bare Speed) |
| Context window | 262,144 tokens |
| MTP depth | adaptive, ceiling 3 |
| Sampling | temperature 1.0, top-p 0.95, top-k 20 (the official Qwen 3.8 contract) |

The serving contract ships inside `mtplx_runtime.json`. MTPLX reads it on
load. Drafts are accepted with the probability-ratio rule plus residual
resampling, so the output follows the model's own distribution at any
temperature.

## Use it

Mac app: download at [mtplx.com](https://mtplx.com), pick "Qwen 3.8
Flash-Next Optimized Speed".

Command line:

```bash
pip install mtplx
mtplx serve --model Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed
```

Sibling: [Bare Speed](https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Bare-Speed)
(flat 4-bit, the quickest build).

Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
(Qwen Community License; the upstream model card is preserved in this repo as
`README-upstream-qwen.md`).
