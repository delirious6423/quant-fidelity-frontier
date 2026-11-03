---
license: mit
license_link: LICENSE
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: paiton
pipeline_tag: text-generation
inference: false
tags: [paiton, amd, rocm, cdna4, gfx950, mi350, mi350p, mxfp4, fp4, 3-bit, moe, expert-offload, cpu-offload, engram, speculative-decoding]
---

# DeepSeek-V4.1-Flash-1017 for Paiton on one AMD Instinct MI350P (CDNA4)

Our quantised files and expert placement for running
[deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (revision
`1bae0e10`) on **a single AMD Instinct MI350P (144 GB)** with Paiton's native CDNA4 (`gfx950`) engine.

The original release is 475 GiB: 269 GiB of routed experts (40 layers x 384 experts, MXFP4: E2M1 with an E8M0 scale per
32 weights), 189 GiB of engram tables (two n-gram lookup tables, E4M3 with an E8M0 scale per 32) and about 17 GiB of
everything else. It does not fit on one card, so the experts are split between the GPU and the host:

- **GPU, 5,812 experts in 97.6 GB:** the 3,000 most used experts in their **original MXFP4 format, bit for bit**, and
  the next 2,812 in our 3-bit format `ds41-mx3-v1` (3.31 bits per weight, 14.65 MB per expert instead of 18.8 MB),
  cut by GPTQ from the original FP4 values. 1,024 of the GPU slots follow the experts the current requests use.
- **Host, the other 9,548 experts in their original MXFP4 format.** For decode, the host CPUs compute these experts
  in place; only the activations (a few KB per token) cross the PCIe link. For long prompts, cold experts are streamed
  to the GPU while it computes.
- **Engram tables in host memory, in row-FP4** (E2M1 with the E8M0 scale per 32): 105 GB instead of 203 GB, gathered
  row by row from the last four token ids.
- Trunk (attention, indexer, compressor, shared experts, head), the DSpark drafter (3 stages, 128 experts each) and
  the KV cache stay on the GPU in their original formats.

Only the 2,812 3-bit experts and the engram tables differ from the original release. This repository holds just those,
plus the placement map and their documentation (135.8 GiB in total). The engine reads every other tensor from the
original checkpoint at revision `1bae0e10`, which you download from deepseek-ai; on first start, it builds its own layout
of those tensors from that snapshot. Total download: the original checkpoint (475 GiB) plus this repository
(135.8 GiB).

## Host profiles

The GPU side is identical in all profiles; they differ only in where and how the engram tables are kept.

| profile | for | host memory used | host experts (9,548) | engram tables | status |
|---|---|---:|---|---|---|
| **B (this release), all in RAM** | servers with about 320 GB or more | **287.5 GB** (179.5 experts + 105 engram + 3 pinned) | original MXFP4 | row-FP4 (E2M1 + E8M0 per 32), 105 GB, in this repository | **served and measured on the MI350P at 450 W** |
| A, large host | servers with about 512 GB or more | ~490 GB | original MXFP4 | original E4M3 + E8M0, 203 GB | coming |
| C, small host | hosts that cannot hold B | less RAM + NVMe | original MXFP4 | on NVMe, gathered per token | coming |

All numbers below are profile B. Converting the engram tables to row-FP4 costs about +0.003-0.004 KL in our emulator.

The profiles differ only in the engram tables. The host's memory bandwidth, not its size, sets decode speed at
16 users and more (see Limits), so a host with fewer or slower memory channels than ours will be slower in every profile.

## Accuracy

KL divergence from the original model's next-token distribution (top-64 bucket KL, zero = identical), on the thinking and
answer positions of held-out conversations; top-1 = how often the most likely next token agrees. Discrete routing and
indexer top-k selections make this model chaotic: two correct implementations of the original model already differ by
the noise floors below, so read every row as excess over its floor.

**Served engine on the MI350P at 450 W, profile B (release build)**, reference = the original model served by SGLang on two MI355X:

| prompt set | server | KL | top-1 agreement | noise floor: same server measured twice |
|---|---|---:|---:|---|
| chat (1,767 positions) | interactive (64 x 70,000) | **0.0307** | **95.0 %** | 0.0039 |
| chat (same set) | token factory (512 x 12,288) | **0.0311** | **94.5 %** | 0.0006 |
| long context (40 conversations, 1,000 positions) | interactive | **0.0438** | **94.2 %** | 0.0033 |
| long context (40 conversations, 1,000 positions) | token factory | **0.0418** | **94.2 %** | 0.0086 |
| reference: original vs itself (SGLang run twice), a different set of 58 short conversations | - | 0.0172 | 95.7 % | - |

With 512 chat requests at once, the token factory scores 0.0306 KL and 94.5 % top-1 agreement.

Long-context recall: 12 of 12 hidden-passphrase tests passed on the interactive server (prompts of 8K to 68,160 tokens,
passphrase at 10 %, 50 % and 90 % depth), 6 of 6 on the token factory (up to 11,033 tokens). 0 invalid token ids.

**The quantisation alone, in our emulator of the engine's arithmetic** (same code on both sides, all 171,200 positions
of the long-context set, free routing):

| | long-context KL [95 % CI] | top-1 |
|---|---:|---:|
| noise floor (two equally correct arithmetic variants) | 0.0154 | - |
| **profile B** | **0.0444** [0.0295, 0.0633] | 92.1 % |

On the same 1,000 positions as the served rows, the emulator gives 0.0426 (interactive context list) and 0.0404 (token
factory list): serving adds about 0.001 KL, less than the served noise floor.

The 3-bit tier's cost is the format's price: GPTQ recovered only 10-17 % of it, so the placement keeps the most used
3,000 experts lossless and puts the 3-bit tier on the next 2,812.

DSpark (the model's own block drafter; this release drafts 2 tokens per step, for up to 12 running requests) is greedy only and verified against plain decoding: at one
user the output is identical; at 3-4 users a verify batch can take a different (deterministic) summation order and
diverge at a near-tie.

## Stock software on one MI350P

We tried the stock engines on the same MI350P at 450 W with the original checkpoint, following their documentation for
very large models on one GPU:

| one MI350P, 450 W | result |
|---|---|
| vLLM v0.33.0, experts and engram tables offloaded to host memory, every documented setting for very large offloads | loads in 51 minutes (with 619 GB of host memory free), then runs out of GPU memory: the AMD MoE backend's weight preparation copies the offloaded experts back onto the GPU |
| vLLM v0.33.0 with its reference MoE path, GPU graph capture off, one request at a time | loads in 35-47 minutes and answers a one-line prompt at about 0.2 tokens a second, but a 1K-token prompt crashes the server (out of GPU memory), so no decode, prefill or concurrency number exists. Not usable for serving. Its fast MoE paths run out of GPU memory while loading, as above |
| SGLang 0.5.23 (release) and the SGLang ROCm preview of 21 Oct 2026, 300 GB of CPU offload | do not load: out of GPU memory after 13-17 minutes, before serving. The generic offloader builds each layer on the GPU before moving it to host memory, so the model never gets under 144 GB |
| the only out-of-the-box solution for this model | doesn't support CDNA (wave64) GPUs |

The stock setup keeps every expert it cannot place and both engram tables in host memory, about 375 GiB, against 287.5 GB
for profile B. Stock vLLM does run this model on one GPU with more memory (a 252 GB DGX Station GB300); this result is
for one 144 GB MI350P.

The files in this repository cannot be used by a stock engine either: every 3-bit expert row has an fp32 row scale that
no stock format represents exactly (see FORMAT.md), so there is no lossless conversion, and stock engines do not read the
placement map.

## Speed: Paiton engine (proprietary)

The same placement served by Paiton's native CDNA4 engine. That engine is proprietary and is **not distributed**: it is
not in this repository, these numbers are not reproducible from the files here, and these weights are served only by
that engine. Same card, same **450 W** limit, host 2x AMD EPYC 9654 (24 DDR5 channels), profile B.

LocalMaxxing speed test (three full runs, 30-second cells; we report the run with the median single-request decode), one MI350P at 450 W, profile B, interactive server
(64 requests x 70,000 tokens, DSpark up to 12 running requests):

| | c1 | c2 | c4 | c8 | c16 | c32 | c64 | prefill 8K | prefill 64K |
|---|---|---|---|---|---|---|---|---|---|
| **Paiton engine, LocalMaxxing, 450 W** | **77.9** | **112.7** | **166.9** | **241.4** | **377.9** | **518.4** | **709.6** | **4,366** (1.86 s) | **4,227** (15.1 s) |

Decode cells are tokens per second over all streams; prefill cells are prompt tokens per second (time to first token).

Serving sweep, one MI350P at 450 W (our own client, every request decodes its full length, all requests start
together), total output tokens per second, first-token wait (median) in parentheses. The token-factory server holds
512 requests x 12,288 tokens and runs without DSpark:

| server | shape | c1 | c4 | c16 | c64 | c256 | c384 | c512 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| interactive | 1K in / 1K out | 90.7 (0.7 s) | 154.3 (0.8 s) | 203.8 (8.9 s) | 297.2 (28 s) | | | |
| interactive | 8K in / 1K out | 79.5 (1.8 s) | 129.8 (3.9 s) | 151.1 (17 s) | 201.6 (70 s) | | | |
| interactive | 1K in / 8K out | 109.0 (0.7 s) | 199.3 (3.0 s) | 243.6 (11 s) | 370.5 (29 s) | | | |
| token factory | 1K in / 1K out | | | | | 542.8 | 620.8 | 682.5 |
| token factory | 8K in / 1K out | | | | | 299.1 | 317.8 | 328.9 |
| token factory | 1K in / 8K out | | | | | 813.2 | 953.4 | 1,107.6 |

With 512 requests at once each request writes about 1.7 tokens a second (1K in / 1K out) and 2.3 (1K in / 8K out) once its first token arrives, after a median wait of 168 s and 181 s for that first token: the token factory
is for batch work, not chat. GPU board power at 512 requests: 212.5 W (1K out) and 214.7 W (8K out) average under the
450 W cap, about 3,200 and 5,200 output tokens a second per kilowatt (host CPU power not counted).

KV cache next to the weights, as the engine reports it at start-up: **4.48M tokens** on the interactive server
(64 requests x 70,000 tokens) and **6.32M tokens** on the token factory (512 requests x 12,288), one MI350P at 450 W.

## What is in the repository

```
config.json                     repository descriptor (not a Transformers config; see below)
manifest.json                   formats, base revision, calibration provenance, export checks, every file's sha256
residency/residency-MI350P-P3000fp4.npy  int8 [40, 384]: -1 host (original MXFP4), 0 GPU original MXFP4, 1 GPU ds41-mx3-v1
residency/residency-MI350P-P3000fp4.json  legend, counts, GPU expert bytes
residency/rank-v1.npy           float32 [40, 384] expert heat ranking the map and the online tier start from
mx3/layer-00..39.safetensors    the 2,812 3-bit experts: codes, E8M0 scales, set flags, fp32 row scales per matrix
engram/engram-L01.rowfp4.bin    layer-1 engram table, row-FP4, 384,006,168 rows x 136 bytes
engram/engram-L14.rowfp4.bin    layer-14 engram table, row-FP4, 384,016,682 rows x 136 bytes
FORMAT.md                       byte-level formats of ds41-mx3-v1, ds41-residency-v1 and ds41-engram-rowfp4-v1
FILES.md, SHA256SUMS            every file with its size and sha256
LICENSE                         MIT, as the base model
```

`config.json` exists for tooling and download statistics; Transformers cannot load this repository.

## Recipe

Expert heat from 8M teacher-forced calibration tokens and 185K self-generated decode tokens (thinking mode) over nine
domains; placement by heat (rank-v1) under the GPU budget left after trunk, drafter, KV cache and work buffers.
3-bit tier `ds41-mx3-v1`: per group of 32 weights the original E8M0 scale (optionally shifted by a power of
two) and a choice among four 4-magnitude subsets of the E2M1 grid (2 flag bits), so every stored value is an exact MXFP4
value and the GPU runs it on the native FP4 matrix instructions; sign-magnitude rounding with alternating ties, a gain
gate per matrix, GPTQ with FP8 block-32 activation Grams (pooled gate/up per layer, per-expert down), fp32 row scales.
Not adopted, measured: 2-bit tiers (3.3x the 3-bit tier's cost), per-expert gate/up Grams (no gain), 3-bit or 2-bit
engram tables (+0.013 / +0.039 KL), 3-bit or 2-bit host experts (about +0.001 KL per 1 % of host memory traffic saved). The exporter decodes every exported
matrix and checks it equals the quantised values bit for bit. Engram row-FP4: each original row is re-quantised per
32 values with the model's own FP4 activation quantiser (power-of-two scale, round to nearest even).

## Serving

These files are read by Paiton's native CDNA4 engine for MI350 / MI355 behind an OpenAI-compatible server. The engine is
proprietary and is not part of this repository; these weights are served by the proprietary Paiton engine.
Transformers, stock vLLM, SGLang and llama.cpp cannot load the 3-bit format or the row-FP4 engram tables, and no lossless conversion to a stock
format exists (see "Stock software on one MI350P").

## Limitations

- Lossy in the 3-bit tier (tables above). Task benchmarks (math, code, knowledge), multilingual and tool-calling quality
  were not measured for this release; in our KL sets multilingual and knowledge text carry 2-3x the average cost.
- Text only: the vision tower is not served.
- Speculative decoding (DSpark) is greedy only; sampled requests decode without it. Prefix caching, logprobs,
  presence/frequency/repetition penalties, logit bias, structured output and n > 1 are not supported.
- Maximum context: 70,000 tokens per request on the interactive server (64 requests), 12,288 on the token factory
  (512 requests); longer prompts are refused with an error.
- **Host-bound speed.** At 16 and 64 users the step is set by the host CPUs reading cold experts from DRAM (about
  420 GB/s on our two EPYC 9654 sockets, which measure about 456 GB/s in a STREAM-type read test, so there is little
  headroom left on this host). 8K prompts are limited by the cold experts computed on the CPUs; 32K prompts by GPU
  compute. The PCIe Gen5 x16 link (55.8-57.5 GB/s measured) is not the limit in any cell.
- Needs the MI350P on PCIe Gen5 x16, the CPUs' memory on all channels, and the profile's host memory free for the engine.
- Tied to checkpoint revision `1bae0e10` and to the CDNA4 (`gfx950`) runtime.

## License and attribution

The base model [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) is
released by DeepSeek under the [MIT License](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE);
these files are a derivative and are distributed under the same license. The quantisation method builds on GPTQ.

[Paiton](https://eliovp.com/products/paiton) · [Public Paiton plugin](https://github.com/Eliovp-BV/paiton-vllm-plugin)
