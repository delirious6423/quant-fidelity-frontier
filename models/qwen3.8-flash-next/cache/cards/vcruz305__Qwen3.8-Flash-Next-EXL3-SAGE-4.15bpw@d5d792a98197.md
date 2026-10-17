---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
tags:
  - exl3
  - exllamav3
  - quantized
  - moe
  - sage
  - 4-bit
  - multimodal
language:
  - en
  - zh
---

# Qwen3.8-Flash-Next — EXL3 (SAGE) 4.15 bpw

**Experimental artifact; uploaded and byte-verified (109.04 GB, 9 shards + n-gram). Quality gates have not yet run. Nothing below is a quality claim; unknowns are marked _pending_ rather than estimated.**

## What this is

An EXL3 (exllamav3) quantization of **`Qwen/Qwen3.8-Flash-Next-FP8`**, built with **SAGE**, our mixed-precision quantization method for EXL3, instead of a uniform-bitrate pass.

### Spec

| Field | Value |
|---|---|
| Base | `Qwen/Qwen3.8-Flash-Next-FP8` — 185.50 GB, 131 shards, 152,089 tensors |
| Architecture | `Qwen4ExpForConditionalGeneration` — 48 layers, hidden 2560, 24 heads / 2 KV heads, head_dim 256, 512 experts / 10 active |
| Attention | hybrid: 1 full-attention layer per 4, remainder linear attention |
| Context | 262,144 tokens |
| Modalities | text + vision (27-block tower, hidden 1152 → 2560) |
| Pack format | EXL3 mixed-K, **4.14999 bits/param** decoder target |
| Special bits | `head_bits 6`, `ngram_bits 6`, `vision_bits 6`, `mtp_bits 4` |
| Codebook | `mul1`, output scales always |
| Sharding | 8192 MB target → 9 shards + `ngram_embedding.safetensors` |
| Total pack size | 109.04 GB (Hub-verified) |

## How it was built

An EXL3 mixed-K build, allocated for a 4.14999 bits/param decoder target, then encoded with exllamav3. The tokenizer, chat template and configuration ship with the pack.

Toolchain: `vcruz305/exllamav3` @ `f6e42ec` (recipe kernels: int8 mixer, affinity, MTP, fused MoE).

## Quality gates — _pending_

Gates have not run yet. Until they do, this pack carries no quality claim of any kind.

## Serving

**Serving recipe: [vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe)** — the DGX Spark (GB10) route: the `vcruz305/exllamav3` fork as the runtime under TabbyAPI, with setup, serve and chat scripts. Its quick start fetches turboderp's 3.05 bpw pack; to serve this pack instead, point it at this folder (`MODEL_DIR`).

Needs exllamav3 with the recipe kernels (fork `f6e42ec`). Validated configuration on a DGX Spark / GB10 (121 GB budget):

- `EXL3_MIXEDK_LEGACY=1` and `EXL3_MK_BPS=3` — legacy grouped coop path, materially faster on `sm_121`
- `EXL3_NGRAM_STREAM=1` — required for the K6 n-gram table
- `-topk 1` for benchmarking (greedy), `-dc 0.6`, `EXL3_MTP_HEAD_N=65536`, `-cs 65536`

Throughput for this pack: _pending_.

## Provenance and protection level

- **Experimental research artifact.** The included Qwen Community License 1.0 governs use and redistribution of this derivative, subject to its stated conditions; see `LICENSE` and the base repository.
- **Not an official Qwen release.** Independent quantization by `vcruz305`.
- **No quality claims until the gates pass.** Unknown values read _pending_; nothing here is an estimate presented as a measurement.
- **Evaluation text was not used to calibrate the pack.**
- **Verifiable:** the pack is byte-verified against what was encoded, and the checkpoint it was quantized from is public and unchanged.
