---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: paiton
pipeline_tag: text-generation
inference: false
tags: [paiton, amd, rocm, cdna4, gfx950, mi350, mi355, fp4, 4-bit, w4a8, gptq, hadamard, moe, mtp, speculative-decoding]
---

# Qwen3.8-Flash-Next 4-bit (FP4 experts, W4A8) for Paiton on AMD CDNA4 (MI350 / MI355)

4-bit weights for [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `de4b8e4d`),
quantised with Paiton's own GPTQ run for the native CDNA4 (`gfx950`) runtime. The 512 routed experts of all 48 MoE
layers are stored as E2M1 (FP4) codes, the native FP4 MFMA operand of CDNA4, with one signed bf16 scale per 128 weights
(4.125 bits per weight) in a Hadamard-128 rotated basis; the trunk is 8-bit (E4M3 per output row, bf16 activations), the
language-model head int8, the n-gram embedding table E4M3. The MTP (speculative decoding) layer with a drafter fine-tuned
on this target, the draft heads and the bf16 vision tensors are included.

Container: `A5-fp4g128-guld4m` (format `fnq-container-1`, experts `fp4g128`). The 16-bit checkpoint is 335 GiB; this
repository is 120.4 GiB (65.9 GiB of 4-bit text weights plus 54.6 GiB of shared tables in `common/`).

## Accuracy (emulation of the served arithmetic, bf16 reference = the original checkpoint)

Full-vocabulary KL divergence from the bf16 model's next-token distribution to this model's, averaged over the text and
assistant positions of a held-out 294,912-token evaluation set (KL t+a; zero = identical, two correct bf16 implementations
score about 0.008). "Free" lets the quantised model choose its own experts; "routing locked" forces bf16's expert choices
and isolates the arithmetic error. Top-1 is how often the most likely next token agrees with bf16.

| | KL t+a free [95 % CI] | KL t+a routing locked | top-1 agreement | KL all positions |
|---|---|---|---|---|
| **this release** (per-expert gate/up and down Hessians) | **0.0265** [0.0239, 0.0292] | **0.0131** | **93.78 %** | 0.1207 |
| same format, pooled gate/up Hessians (A3) | 0.0275 [0.0251, 0.0300] | 0.0145 | 93.61 % | 0.1288 |
| same format, 2M calibration tokens (A2) | 0.0283 [0.0258, 0.0309] | 0.0151 | 93.51 % | 0.1271 |
| INT4-g128 experts (the RDNA4 4-bit recipe, same tool) | 0.0288 | 0.0135 | 93.51 % | |

Paired per-token bootstrap of this release against A3 (rows as the independent units): free -0.0010 [-0.0022, +0.0003],
routing locked -0.0014 [-0.0022, -0.0005]. A replicate of A3 cut on a different 4M-token calibration draw scores
0.0310 / 0.0146, so the free-routing difference between calibration draws is about 0.0035 and the locked metric is the
sharper one.

Long context (teacher-forced NLL in nats per token, difference to the bf16 model; rows = documents):
- long-v1 (44 held-out documents, 8-32K tokens, text + assistant positions): +0.06 % (0-2K), +0.03 % (2-8K),
  +0.19 % (8-32K) vs bf16; against A3 every bucket's paired 95 % interval includes zero.
- Needles: 8/8 prompts at 131K tokens and 4/4 at 258K tokens reproduced exactly (156/156 and 78/78 answer tokens), as bf16.
- 4 x 262K-token natural Python source: within +2.5 % of bf16 in every position bucket up to 262K (0.37-0.40 nats;
  +0.5 / +2.4 / +0.7 / +0.7 / +1.9 / +0.1 / +0.9 % for 0-2K, 2-8K, 8-32K, 32-64K, 64-128K, 128-195K, 195K-262K). Against
  A3 one bucket (2-8K) is +0.0022 [+0.0003, +0.0041] worse, the others are within their intervals.
- Assistant continuation after a 254K-token user turn of code (32,764 positions): 0.2469 vs bf16 0.2422 (+1.9 %; A3 0.2456).

MTP (speculative decoding), expected accepted drafts per step at depth 4, teacher-forced, greedy, text + assistant
positions: 2.256 with the shipped W8A16 drafter and the exact int8 head (2.213 with the int2 draft head, top-8 rescored);
2.262 with the bf16 drafter and the FP4 MTP experts; 2.042 before the drafter fine-tune.

## Default speed (stock engine, 450 W)

The reference for "default" speed on this card is a supported model of similar precision: AMD's Quark MXFP4 checkpoint
of the same Qwen3.8-Flash-Next (4-bit experts), served by stock vLLM 0.32.0 with AITER kernels, no Paiton code. One
MI350P, GPU power capped at **450 W** (the card's default is 600 W), 262,144-token context, no speculative decoding.
LocalMaxxing (localmaxxing.com) speed test: decode cells at 1-64 concurrent streams with no shared context, 30 s each,
median of three runs; prefill at 8K and 64K prompt tokens. Measured with LocalMaxxing's open-source lmx speed test
(v0.1.54); the raw result files, commands and a run recording are published with the [blog post](https://eliovp.com/blog/paiton-qwen38-flash-next-amd-instinct-mi350p).

| | c1 | c2 | c4 | c8 | c16 | c32 | c64 | prefill 8K | prefill 64K | KV cache (tokens) |
|---|---|---|---|---|---|---|---|---|---|---|
| Quark MXFP4, vLLM 0.32.0 + AITER (tok/s) | 122.1 | 195.4 | 384.7 | 602.8 | 1,029.0 | 1,544.1 | 2,185.2 | 7,009 | 7,224 | 1,795,019 |

## Paiton speed (proprietary engine)

The same weights served by Paiton's native CDNA4 engine (the optimised MI350 kernels, the `.so`). That engine is
proprietary and is **not distributed**: it is not in this repository, and these numbers are
not reproducible from the files here; these weights are served only by that engine.

Same card, same 450 W cap, same test and protocol as above; MTP speculative decoding (3 draft tokens) on for up to 4
concurrent requests, 64 concurrent sequences, 262,144-token context. The comparison rows are the publicly available MXFP4 (Quark) quantisation
of the same checkpoint (the default-speed model above) on the stock ROCm engines, each at its best configuration we
found, on the same card and cap:

| | c1 | c2 | c4 | c8 | c16 | c32 | c64 | prefill 8K | prefill 64K | KV cache (tokens) |
|---|---|---|---|---|---|---|---|---|---|---|
| **this model, Paiton engine** | **273.1** | **453.4** | **676.5** | **839.6** | **1,265.8** | **1,886.7** | **2,508.6** | **13,709** | **14,286** | **2,190,540** |
| vLLM 0.32.0 + AITER | 122.1 | 195.4 | 384.7 | 602.8 | 1,029.0 | 1,544.1 | 2,185.2 | 7,009 | 7,224 | 1,795,019 |
| SGLang 0.5.22, NEXTN x3 | 118.6 | 223.1 | 397.0 | 639.2 | 634.4 | does not fit | does not fit | 12,069 | 11,585 | 367,424 |
| SGLang 0.5.22, NEXTN x3, 16 running requests | 119.6 | 219.0 | 403.5 | 629.9 | 829.3 | does not fit | does not fit | 11,535 | 13,827 | 64,704 |

Decode cells are tokens per second over all streams; prefill cells are prompt tokens per second. KV cache is the number
of context tokens that fit on the card next to the weights (SGLang's KV pool size depends on its server config, 8 vs
16 running requests; values are what each config allocated on this card). SGLang's NEXTN configs do not start with
room for 32 or more requests on this card; without speculative decoding it reached 1,088.6 / 1,093.0 tok/s at 32 / 64. Every row was checked for corrupt output (no invalid token
ids, no repetition loops) before it was kept. At 450 W the engine is 2.24x the best stock row at one stream, 1.31x at
eight, 1.23x at sixteen, 1.22x at 32 and 1.15x at 64, and 1.14x / 1.03x the best stock row at 8K / 64K prefill.

## What is in the repository

```
manifest.json                 format of every tensor, calibration provenance, in-run bit-exactness checks, file sha256
layers/L00..L47.safetensors   one decoder layer each: FP4 experts (codes + bf16 scales), E4M3 trunk, bf16 rest
model.safetensors             embed_tokens (bf16), final hyper-connection mixer, lm_head (int8-g128 + E4M3 fallback rows)
mtp/mtp-experts.safetensors   the MTP layer's 512 routed experts, same FP4 format, re-cut on the fine-tuned layer (+ mtp-manifest.json)
mtp/mtp-ft-nonexpert.safetensors  the fine-tuned non-expert MTP tensors (bf16 overlay over common/bf16-mtp-vision.safetensors)
mtp/mtp-draft-e4m3.safetensors    the same drafter linears as served: E4M3 per output row + fp32 row scales (W8A16)
mtp/draft-head-int2.safetensors   int2-g128 draft head of the served int8 head (+ draft-head-int2.json)
mtp/SHA256SUMS
common/ple-e4m3/shard_000..127.safetensors   n-gram embedding table (E4M3 + fp32 row scale)
common/bf16-mtp-vision.safetensors           MTP layer (non-expert parts) and vision tower in bf16, as in the checkpoint
FORMAT.md                     byte-level format of the expert, trunk, head and table tensors
FILES.md, SHA256SUMS          every file with its size and sha256
```
Sizes: layers 61.5 GiB (48 x 1.28 GiB), model.safetensors 2.4 GiB, mtp 1.9 GiB, common 54.6 GiB.

## Recipe

GPTQ of the routed experts in the Hadamard-128 basis, sequential over layers with each layer calibrated on the quantised
model's own activations, 4,001,792 calibration tokens, per-expert gate/up and down-projection Hessians with Ledoit-Wolf
shrinkage, scale refit after GPTQ; E2M1 grid with an MSE-searched signed bf16 scale per (row, 128). Trunk, head and
n-gram table are the ones of the RDNA4 4-bit release (GPTQ W8 E4M3 rows, int8-g128 head, E4M3 table). The MTP drafter
was fine-tuned by self-distillation on this target's served states (2M tokens, one epoch), then its experts re-cut in the
same FP4 format and its linears stored as E4M3 rows. The exporter verifies per layer that dequantising the stored codes
reproduces the emulated weights bit-exactly (`checks` in manifest.json).

Selection: this container was chosen among candidate cuts (8M calibration tokens, coordinate descent over the codes,
per-expert gate/up Hessians) by a paired per-token bootstrap against A3 on free and locked KL, then the long-context
gates above. FP4 expert activations (MXFP4 shape) and an fp8 per-token trunk were measured on the same weights and not
adopted: +113 % and +25 % KL respectively.

## Serving

This container is read by Paiton's native CDNA4 engine for MI350 / MI355 (FP4 expert GEMV/GEMM with online FWHT-128
activations, fp8 per (token, 128) expert inputs; see `served_sites` in manifest.json). The optimised MI350 kernels are
proprietary and are not part of this repository; these weights are served by the proprietary Paiton engine.

**Transformers, stock vLLM and llama.cpp cannot load this format.**

## Limitations

- The weights are lossy (tables above). Task benchmarks (math, code, knowledge), multilingual and tool-calling quality
  were not measured for this release.
- Tied to checkpoint revision `de4b8e4d` and to the CDNA4 (`gfx950`) runtime.

## License and attribution

The base model [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is released by Qwen under the
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE); these weights are a
derivative and are distributed under the same license. The quantisation method builds on GPTQ.

[Paiton](https://eliovp.com/products/paiton) · [Public Paiton plugin](https://github.com/Eliovp-BV/paiton-vllm-plugin)
