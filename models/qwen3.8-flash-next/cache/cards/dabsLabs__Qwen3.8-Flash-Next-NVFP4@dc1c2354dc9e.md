---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - nvfp4
  - compressed-tensors
  - blackwell
  - vllm
  - qwen4_exp
  - moe
  - multimodal
---

# Qwen3.8-Flash-Next — NVFP4 (experts-only, 4/6-optimal scales)

## TL;DR — the plain-English version

**What this is:** Qwen3.8-Flash-Next — Qwen's fast 125B mixture-of-experts model (only ~6B
parameters active per token, so it's quick) — shrunk from **360 GB to ~174 GB**, with the
quality-critical parts left completely untouched.

**How:** the model's "experts" are stored in **NVFP4**, a 4-bit format NVIDIA Blackwell
GPUs run natively. Attention, the vision system, the speculative-decoding head, and the
model's big 51 GB n-gram memory (PLE) stay at full precision.

**Is the quality still good?** Measured per-tensor error is ~8.7 % relative — good for
4-bit — using a scale-selection trick that cuts error ~16 % vs how most quants are made.
As an extra safety check, our conversion was cross-verified against **Qwen's own official
FP8 release**, expert by expert.

**Can I run it?** It's built for Blackwell GPUs via vLLM (the same dedicated image that
serves the official FP8 release — this architecture runs on RTX PRO 6000-class cards
today). Fair warning, stated plainly: **we haven't completed our own end-to-end serving
run of this file yet** — it's verified numerically and structurally, and it uses the exact
tensor layout the vLLM loader already consumes in production, but until we (or you!)
serve it, treat it as very-likely-working rather than proven. This note will be updated.

**VRAM tip:** set `VLLM_PLE_CPU_OFFLOAD=1` to keep the 51 GB n-gram memory in system RAM —
GPU weight footprint drops to ~90 GB (two 96 GB cards territory).

**New to quantization?** Think high-quality JPEG vs RAW: much smaller, nearly
indistinguishable — and this card publishes the exact measurements of what was lost.

---

Weight-only **NVFP4 (NVFP4A16)** quantization of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — the 125B-A6B
`qwen4_exp` hybrid MoE (Gated DeltaNet + Qwen Sparse Attention, PLE n-gram embedding, MTP,
vision) — in standard **compressed-tensors `nvfp4-pack-quantized`** format. ~174 GB.

Quantized from the **true BF16 release** (360 GB, pinned `f5d08274`), not the FP8 export.

## What makes this build different

- **4/6-optimal block scales**: per 16-element block, the scale is chosen by exact
  post-rounding squared error between the classic `amax/6` plan and ModelOpt's
  *four-over-six* plan — never worse than plain RTN by construction. On this model:
  **−16.11 % total squared error** vs plain RTN; 44.6 % of blocks chose M=4.
- **Split-convention cross-check**: the BF16 release stores experts *stacked*
  (`[512, out, in]`, fused gate_up). This build unstacks them into the per-expert layout of
  Qwen's own official FP8 export — and **cross-checks every sampled expert against that FP8
  export** (mean rel-err 0.0266 ≈ pure fp8 noise), ruling out gate/up-split errors by
  construction.

| Metric | Value |
|---|---|
| Mean relative error (quantized experts vs BF16 source) | **0.0869** |
| Max relative error | 0.0870 |
| Cross-check vs Qwen's official FP8 export | 0.0266 (48 samples) |
| Expert census | 73,728 / 73,728 (48 layers × 512 experts × 3 projections) |

## What is quantized, what is not

- **NVFP4**: routed-expert weights only (≈ 121B of the parameters).
- **BF16 (untouched)**: the 51B **PLE n-gram embedding**, all attention (Qwen Sparse
  Attention + DeltaNet), shared experts, router gates, embeddings, `lm_head`, the **full
  vision tower** (byte-identical, verified per-tensor: 333 tensors), and the entire **MTP
  tree** (speculative decoding stays full-precision).

## Serving notes

The `qwen4_exp` architecture requires the dedicated vLLM image
(`vllm/vllm-openai:qwen38-flash-next`) until upstream support lands. Recipe-critical args:
`--max-num-seqs 256` (Mamba cache blocks; lower values error at startup),
`--no-enable-flashinfer-autotune`, `VLLM_ENGINE_READY_TIMEOUT_S=3600`.

- The architecture serves on SM120 (RTX PRO 6000) today — we run the official FP8 release
  on 4×96 GB in production. Marlin is among the listed NVFP4 MoE backends for this arch.
- **Honesty note**: this NVFP4 checkpoint has been verified structurally and numerically
  (census, FP4-grid, FP8-export cross-check, vision byte-equality) but **we have not yet
  served it end-to-end**. The tensor layout mirrors the official FP8 export that the vLLM
  loader consumes in production, so loader-naming risk is low. This note will be updated
  when we complete a serving run.
- VRAM lever: `VLLM_PLE_CPU_OFFLOAD=1` moves the 51B PLE embedding to host RAM
  (~90 GB on-GPU weights; needs ~+51 GB pod/host RAM).

## Verification methodology

Quantization math bit-exact vs an llmcompressor `NVFP4A16` oneshot and byte-identical vs
NVIDIA ModelOpt's `NVFP4QTensor` export path at the pinned toolchain (`torch 2.13.0` /
`compressed-tensors 0.18.0` / `safetensors 0.8.0`). Streaming shard-by-shard conversion
(model never materialized). `MANIFEST.sha256` covers every file. Built 2026-08-26.

Licensed under the **Qwen Community License 1.0** (included), as is the base model.

*Quantized by dabsLabs.*
