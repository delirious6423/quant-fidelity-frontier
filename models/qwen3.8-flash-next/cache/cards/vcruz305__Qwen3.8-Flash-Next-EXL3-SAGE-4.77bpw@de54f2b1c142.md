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

# Qwen3.8-Flash-Next — EXL3 (SAGE) 4.77 bpw

**1x DGX Spark (128 GB) profile (~101.5 GiB / 109.0 GB total pack). All release quality gates passed — Headline KLD `0.0079` (`0.007898`), Top-1 Agreement `98.01%`, Top-5 Overlap `91.65%`.**

## What this is

An EXL3 (`exllamav3`) quantization of **`Qwen/Qwen3.8-Flash-Next`**, built with **SAGE**, our mixed-precision quantization method for EXL3, instead of a uniform-bitrate pass.

### Spec

| Field | Value |
|---|---|
| Base | `Qwen/Qwen3.8-Flash-Next` |
| Architecture | `Qwen4ExpForConditionalGeneration` — 48 layers, hidden 2560, 512 experts / 10 active |
| Attention | hybrid: 1 full-attention layer per 4, remainder linear attention |
| Context | 262,144 tokens |
| Modalities | text + vision (27-block tower) |
| Pack format | EXL3 mixed-K, **4.77 bits/param** routed MoE target |
| Special bits | `head_bits 6`, `ngram_bits 4`, `vision_bits 6`, `mtp_bits 4` |
| Codebook | `mul1`, output scales always |
| Sharding | 11 model shards + `ngram_embedding.safetensors` |
| Total pack size | ~101.5 GiB (~109.0 GB) |

## How it was built

An EXL3 mixed-K build, allocated for the 4.77 bpw profile (`ngram_bits 4`), then encoded with `exllamav3`. The tokenizer, chat template and configuration ship with the pack.

SAGE is our own mixed-precision quantization method for EXL3. The method is not published. If a tensor is EXL3, its K is self-describing on the trellis.

Toolchain: `vcruz305/exllamav3` @ `f6e42ec` (recipe kernels: int8 mixer, affinity, MTP, fused MoE).

## Quality gates — PASS

Full-vocabulary `float64` KL divergence, Total Variation (`TV`), teacher-forced NLL delta, Top-5 Overlap, and unrounded Top-1 Agreement against `Qwen/Qwen3.8-Flash-Next` (`BF16`):

| Pack | Pack size | KL vs BF16 (mean) | Top-1 Agreement | Top-5 Overlap | Mean TV | NLL Delta |
| --- | --- | --- | --- | --- | --- | --- |
| **this pack, 4.77 bpw** | ~101.5 GiB (~109.0 GB) | **0.0079** (`0.007898`) | **98.01%** (`18,319 / 18,690`) | **91.65%** | **0.0209** | **+0.0047** |
| BF16 reference | native | 0 | 100% | 100% | 0 | 0 |

### Per-category breakdown

| Category | KL vs BF16 (mean) | Top-1 Agreement | Top-5 Overlap | Mean TV | NLL Delta |
|---|---:|---:|---:|---:|---:|
| **Code (`3,986` tok)** | **0.0027** (`0.002721`) | **99.45%** (`3,964 / 3,986`) | **89.94%** | **0.0064** | **+0.0029** |
| **Reasoning (`4,662` tok)** | **0.0035** (`0.003508`) | **98.67%** (`4,600 / 4,662`) | **93.55%** | **0.0141** | **+0.0032** |
| **Math (`4,113` tok)** | **0.0061** (`0.006076`) | **98.81%** (`4,064 / 4,113`) | **91.17%** | **0.0163** | **+0.0034** |
| **General (`5,929` tok)** | **0.0161** (`0.016092`) | **95.99%** (`5,691 / 5,929`) | **91.64%** | **0.0391** | **+0.0081** |

## Serving

**Serving recipe: [vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe)** — the DGX Spark (GB10) route: the `vcruz305/exllamav3` fork as the runtime under TabbyAPI, with setup, serve and chat scripts. To serve this pack, point it at this folder (`MODEL_DIR`).

Needs `exllamav3` with the recipe kernels (fork `f6e42ec`). Validated configuration on a DGX Spark / GB10 (121 GB budget):

- `EXL3_MIXEDK_LEGACY=1` and `EXL3_MK_BPS=3` — legacy grouped coop path, materially faster on `sm_121`
- `EXL3_NGRAM_STREAM=1` — streams the n-gram table
- `-topk 1` for benchmarking (greedy), `-dc 0.6`, `EXL3_MTP_HEAD_N=65536`, `-cs 65536`

## Provenance and protection level

- **Experimental research artifact.** The included Qwen Community License 1.0 governs use and redistribution of this derivative, subject to its stated conditions; see `LICENSE` and the base repository.
- **Not an official Qwen release.** Independent quantization by `vcruz305`.
- **No quality claims until the gates pass.** Unknown values read _pending_; nothing here is an estimate presented as a measurement.
- **Evaluation text was not used to calibrate the pack.**
- **Verifiable:** the pack is byte-verified against what was encoded, and the checkpoint it was quantized from is public and unchanged.
