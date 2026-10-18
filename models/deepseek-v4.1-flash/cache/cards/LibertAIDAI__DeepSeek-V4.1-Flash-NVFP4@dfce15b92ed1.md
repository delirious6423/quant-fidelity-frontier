---
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
base_model: deepseek-ai/DeepSeek-V4.1-Flash
base_model_relation: quantized
tags:
  - nvfp4
  - fp4
  - deepseek
  - moe
  - multimodal
  - libertai
---

# DeepSeek-V4.1-Flash-NVFP4

**475.2 GiB → 399.9 GiB (−15.8%)**, with the routed experts converted to NVFP4 **bit-exactly**
and the Engram memory tables taken from FP8 to FP4.

An FP4 re-pack of [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
by [LibertAI](https://libertai.io).

## ⚠️ Read this first: V4.1-Flash already ships in 4-bit

If you came here expecting the usual ~70% NVFP4 cut, **it does not exist for this model.**
DeepSeek quantized V4.1-Flash themselves. From the upstream `config.json`:

```json
"quantization_config": {
  "quant_method": "fp8", "weight_block_size": [32, 32],
  "scale_fmt": "ue8m0", "expert_dtype": "fp4"
}
```

Measured across all 48 upstream shards / 96,085 tensors (475.2 GiB):

| component | size | share | dtype as shipped |
|---|---:|---:|---|
| routed experts | 268.9 GiB | 56.6% | **E2M1 packed 2/byte + E8M0 scales, block 1×32** |
| Engram tables | 189.1 GiB | 39.8% | E4M3 + E8M0 scales, block 1×32 |
| MTP (3 layers) | 7.4 GiB | 1.6% | E2M1, as above |
| attention / dense | 4.9 GiB | 1.0% | E4M3, block 32×32 |
| embeddings, head, vision | 4.0 GiB | 0.8% | BF16 |

This is the break from V4-Flash. `DeepSeek-V4-Flash-0731` shipped FP8 block-128×128 with no
FP4 anywhere — which is why the NVFP4 conversions of *that* model were worth ~2×. Here the
57% of the checkpoint that is routed experts is already at 4 bits, so quantizing it again
buys nothing. **The Engram tables are the only large block left at 8 bits, and they are what
this repo is really about.**

## What this repo does

### 1. Experts: MXFP4 → NVFP4, bit-exactly

Upstream experts are MX-style: 32-element blocks with a power-of-two E8M0 scale. NVFP4 is
16-element blocks with an E4M3 scale plus an FP32 per-tensor global scale.

The transcode is **lossless**, for a structural reason: NVFP4's 16-element blocks are a
strict refinement of the source's 32-element blocks, so both NVFP4 half-blocks inherit one
source scale `s`. Emit the nibbles **unchanged** and force `weight_scale × weight_scale_2 == s`,
and every reconstructed value is identical. E8M0 scales are powers of two, and E4M3
represents 2^k exactly for k ∈ [−9, 8], so the identity holds for every block whose scale
lands in that window. The global scale is chosen per tensor to centre it.

Measured over the **whole checkpoint**:

| | |
|---|---|
| expert weights transcoded | **47,232** |
| block scales inside E4M3's exact window | **34,823,208,960 / 34,823,208,960 (100%)** |
| independent reconstruction check | **0 mismatches over 2.36 × 10⁹ elements, `max\|delta\| = 0`** |

**This costs size.** NVFP4 is 4 + 8/16 = 4.5 bits/weight against MXFP4's 4 + 8/32 = 4.25, so
the experts *grow* — one shard went 7,389,759,032 → 7,814,559,376 bytes, **+5.75%**. Take
the expert half only if your engine's Blackwell MoE kernels want NVFP4 and have no MX path.

Prior art: this losslessness is not something we invented, it is what any careful
implementation gets. We checked
[`s-zaizen/DeepSeek-V4.1-Flash-NVFP4`](https://huggingface.co/s-zaizen/DeepSeek-V4.1-Flash-NVFP4),
published the same day, by reconstructing `layers.1.ffn.experts.0.w1` from their checkpoint
and from the source: **their nibbles are identical and all 11,796,480 elements match
exactly.** Their repo is **491.1 GiB — larger than the 475.2 GiB original** — because the
expert transcode grows the file and Engram is left at FP8.

### 2. Engram: FP8 → FP4 — the actual size lever

The Engram conditional-memory tables are two tensors of ~384M rows × 256, 94.6 GiB each.
We take them to E2M1 **keeping the source's own block-32 / E8M0 scale layout**, so a table
row is stored exactly the way DeepSeek stores an expert weight and any kernel that
dequantizes an expert can dequantize the table. Scales are recomputed as a power of two
≥ amax/6 rather than reused, so **no block clips**.

| table | source | output | mean cosine (1.5M sampled rows) |
|---|---:|---:|---:|
| `layers.1.engram.embed` | 94.6 GiB | **48.8 GiB** | **0.993415** |
| `layers.14.engram.embed` | 94.6 GiB | **48.8 GiB** | **0.993445** |

⚠️ **This half is lossy and has not been evaluated end-to-end.** Cosine ≈ 0.9934 is lower
than we would accept on expert weights, and no benchmark has been run — no engine can run
this architecture yet (see below). If you need a checkpoint with untouched Engram, use
upstream or s-zaizen's repo. We publish this because the size and I/O win is large and the
measurement is stated honestly, not because the quality is established.

## Net effect

| | upstream | s-zaizen NVFP4 | this repo |
|---|---:|---:|---:|
| routed experts + MTP | 276.3 GiB | ~292 GiB | ~292 GiB |
| Engram | 189.1 GiB | 189.1 GiB | **97.6 GiB** |
| everything else | 8.9 GiB | 8.9 GiB | 8.9 GiB |
| **total** | **475.2 GiB** | **491.1 GiB** | **399.9 GiB** |

143,317 tensors across 48 shards. The index is rebuilt from the shard headers, and every
tensor the source had is present.

## Running Engram off SSD

Engram is a hashed n-gram **gather**, not a GEMM. With `engram_layer_ids [1,14]`,
`engram_max_ngram_size 4` and `engram_n_heads 8`, each position looks up 3 n-gram sizes ×
8 heads = 24 rows per Engram layer, so **48 rows per token**:

| | per row | per token |
|---|---:|---:|
| upstream FP8 | 264 B | 12.4 KB |
| this repo FP4 | 136 B | **6.4 KB** |

At decode that is ~48 random reads per token — trivial for an NVMe. Prefill is where it
bites: a 64K-token prompt is ~3.1M small random reads. **So the FP4 Engram halves the SSD
traffic per token as well as the footprint.** Upstream anticipates this: their `convert.py`
already shards `engram.embed` along dim 0 across ranks.

**But it does not make small boxes work.** Everything except the two Engram tables is
**302.3 GiB (324.6 GB)** and must be resident. Two 120 GB unified-memory boxes cannot hold
it, with or without Engram on disk. A 4×B200 or an 8×96 GB node can.

## On pruning

We looked at whether V4.1-Flash could be expert-pruned to fit smaller hardware without
calibration data. **It cannot**, and the checkpoint says so three ways:

* the `noaux_tc` router bias is nearly uniform — **std 0.0627** against absolute values of
  9–57, top/bottom-decile gap ~0.13. Load balancing has equalised expert usage, so there
  are no cold experts to drop.
* router directions are not redundant — nearest-neighbour cosine averages **0.32–0.45**,
  and 0–2 experts per layer exceed 0.9. Nothing to merge.
* ⚠️ **`gate.bias` and `gate.bias_vl` are effectively uncorrelated** (mean **−0.05**, min
  −0.45, mean absolute difference 6.4). The model carries a *separate* router bias for
  vision-language tokens, so expert specialisation is modality-dependent and **a text-only
  REAP calibration would prune the vision path.** If you are pruning this model, calibrate
  on multimodal data.

Any real prune needs activation-based REAP, which needs a working engine.

## Engine support

As of 2026-09-10 **no released engine serves `deepseek_v41`.** vLLM has open PRs
([#56214](https://github.com/vllm-project/vllm/pull/56214),
[#56228](https://github.com/vllm-project/vllm/pull/56228),
[#56227](https://github.com/vllm-project/vllm/pull/56227),
[#56208](https://github.com/vllm-project/vllm/pull/56208)) filed the morning the model
dropped; SGLang has merged a cookbook plus initial support.

⚠️ **DeepSeek's own `inference/` stack is bundled here for reference but will NOT load this
checkpoint as-is.** It expects the upstream layout: `convert.py` asserts routed experts are
`torch.int8` and reads their scales from `<name>.scale`, whereas this repo stores them as
`uint8` with `<name>.weight_scale` (E4M3, block 16) plus a `<name>.weight_scale_2` global
scale, and stores the Engram tables at E2M1 rather than E4M3. Loading it needs a reader that
understands the `quantization_config` below. We ship the reference code unmodified so the
delta is inspectable, not because it works out of the box.

Because nothing reads this architecture yet, the `quantization_config` in `config.json`
**describes** what we did rather than matching an existing convention:

```json
"expert_dtype": "nvfp4", "expert_block_size": 16, "expert_scale_fmt": "e4m3",
"expert_global_scale": true,
"engram_dtype": "fp4", "engram_block_size": 32, "engram_scale_fmt": "ue8m0"
```

Dense/attention weights are untouched (fp8, block 32×32, ue8m0). **No throughput, latency
or quality numbers are published here, because we have not been able to run the model.**

## About the base model

DeepSeek-V4.1-Flash is a multimodal MoE with a **552B-parameter backbone** plus **196B of
sparsely-accessed Engram memory**, activating **8B parameters per token during prefill and
16B during decode**, with a **1M-token** context.

It uses a **Causal Encoder-Decoder** layout (40 layers = a 20-layer causal encoder feeding a
20-layer decoder), **Compressed Sparse Attention 2** with per-layer Full/Reindex/Reuse modes
and a hierarchical sparse indexer, **FP4 main KV caching** for a global KV footprint of
**890 bytes/token**, Single-Pass mHC, and **DSpark** speculative decoding. 384 routed experts
+ 1 shared, 6 routed per token. Reasoning effort is continuously controllable from 1 to 100.

See the [technical report](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf)
for evaluations. We have run none of our own.

## Provenance

Built with [`dsv41_fp4_stream.py`](./dsv41_fp4_stream.py), a shard-streaming re-packer.
The E2M1 table and nibble order are taken verbatim from the checkpoint's own
`inference/convert.py`, not reconstructed from the spec. The Engram shards are written as
hand-built safetensors containers so a 94.6 GiB table is never held in RAM.

Also included: [`verify_transcode.py`](./verify_transcode.py) (the bit-exactness check),
[`test_engram_path.py`](./test_engram_path.py) and
[`test_engram_shard.py`](./test_engram_shard.py) (cpu/cuda agreement and container
round-trip), and [`analyze_router.py`](./analyze_router.py) (the pruning analysis above).

License: MIT, inherited from the base model.
