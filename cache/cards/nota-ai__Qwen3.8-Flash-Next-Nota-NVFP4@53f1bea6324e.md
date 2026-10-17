---
language:
- en
- zh
- ko
base_model:
- Qwen/Qwen3.8-Flash-Next
tags:
- nota
- nvfp4
- moe
- quantization
license: other
license_name: qwen-community-1.0
license_link: LICENSE
library_name: transformers
pipeline_tag: image-text-to-text
---

# Nota Qwen3.8-Flash-Next · NVFP4 (W4A4)

A 4-bit (NVFP4) quantized release of [Qwen](https://qwen.ai/)'s
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) — a **180B-parameter**
natively multimodal Mixture-of-Experts model with ~7B active per token, built on the
architecture that will underpin Qwen4.

<div align="center">

**Half the GPUs, in every configuration**

**335.3 GiB → 173.6 GiB (51.8%)**

Full 262,144-token context · MTP speculative decoding preserved

</div>

| Deployment | BF16 base | This release |
|---|---:|---:|
| RTX PRO 6000, PLE in host memory | 4 cards | **2 cards** |
| RTX PRO 6000, PLE resident on GPU | 8 cards | **4 cards** |
| B200, PLE resident on GPU | 4 cards | **2 cards** |

Minimum cards that serve one full 262,144-token request, from measured per-rank footprints.

## Highlights

- **NVFP4 (4-bit float, W4A4)** — `group_size=16`, packed in the compressed-tensors
  `nvfp4-pack-quantized` format for direct serving in
  [vLLM](https://github.com/vllm-project/vllm). Both weights and activations are quantized to
  4-bit floating point.
  - > **Requires NVIDIA Blackwell.** NVFP4 relies on the FP4 tensor cores introduced in the
    > Blackwell architecture (e.g. B200 / B300 / GB200 / RTX PRO 6000). Earlier architectures
    > (Hopper, Ada, Ampere) do not support NVFP4 execution.
- **Only the routed experts are quantized.** They hold 67.1% of the parameters and 84.2% of the
  BF16 checkpoint's bytes, so the memory saving is captured almost in full while every
  precision-critical path stays in BF16 — the same split the reference NVFP4 release uses.
- **No architecture change.** Layer count, expert count and expert routing are identical to the
  base checkpoint, so stock vLLM serves it as-is — no patched modeling file.
- **MTP, PLE and multimodal preserved.** The multi-token-prediction block, the N-gram PLE
  embedding and the vision tower stay BF16, so speculative decoding, the PLE pathway and
  image/video inputs work as in the base model.

## Performance

| Benchmark | BF16 | NVFP4 |
|---|---:|---:|
| GPQA-Diamond (Graduate-Level Reasoning) | 89.90 | 90.40 |
| HumanEval (Code Generation) | 95.12 | 98.17 |
| AIME 2026 (Competition Mathematics) | 90.00 | 93.33 |
| HRM8K (Korean Mathematical Reasoning) | 78.85 | 78.64 |
| BFCL v4 (Tool Calling) | 77.64 | 78.83 |

* Total context budget 131,072 tokens; generation length 65,536 tokens.

* Greedy decoding (`temperature=0`, `top_p=1.0`, `seed=0`), single pass. These
  evaluations are intended to quantify quantization degradation rather than to produce
  leaderboard figures, and therefore employ no sampling, no avg@n and no
  self-consistency.

* **HRM8K** is restricted to the **KSM** subset (1,428 of 8,011 items). KSM is the only
  subset drawn from original Korean mathematics examinations; the remaining four subsets
  (MATH, GSM8K, OMNI_MATH, MMMLU) are translations of English benchmarks and would
  duplicate the reasoning already assessed by AIME and GPQA.

* **BFCL v4** is restricted to the **13 single-turn categories**: `simple_python`,
  `simple_java`, `simple_javascript`, `multiple`, `parallel`, `parallel_multiple`,
  `irrelevance`, `live_simple`, `live_multiple`, `live_parallel`,
  `live_parallel_multiple`, `live_irrelevance` and `live_relevance`. The multi-turn and
  memory categories are excluded, as are the two web-search categories, which require an
  external search API key.

* Where NVFP4 exceeds BF16, the result should be interpreted as **an absence of
  degradation rather than an improvement**. Each difference falls within the sampling
  error of its benchmark. Four-bit quantization does not improve a model; under greedy
  decoding, minute numerical differences resolve individual questions in either
  direction, and the aggregate may fall marginally above or below the baseline.

## Quantization scope

| | Modules | Parameters |
|---|---:|---:|
| NVFP4 — routed experts, all 48 MoE layers | 73,728 | 120.80 B (67.1%) |
| BF16 — everything else | 1,559 | 59.20 B (32.9%) |

The 73,728 quantized modules are 512 experts × 48 layers × 3 projections
(`gate_proj`, `up_proj`, `down_proj`). The base checkpoint stores them as 96 fused 3-D tensors
(`mlp.experts.gate_up_proj` `[512, 1280, 2560]`, `mlp.experts.down_proj` `[512, 2560, 640]`);
this release unfolds them into per-expert `mlp.experts.<e>.<proj>` modules, which is what
compressed-tensors and vLLM's `RoutedExperts` loader expect.

Kept in BF16:

| Group | Tensors |
|---|---|
| Gated DeltaNet linear attention (36 layers) | `linear_attn.{in_proj_qkv,in_proj_a,in_proj_b,in_proj_z,out_proj}`, `conv1d`, `norm` |
| QSA full attention (12 layers) | `self_attn.{q,k,v,o}_proj`, `self_attn.{q,k}_norm` |
| QSA indexer | `self_attn.indexer.{index_qk_proj,q_layernorm,k_layernorm}` |
| MoE routers | `mlp.gate`, `mlp.shared_expert_gate` |
| Shared experts | `mlp.shared_expert.{gate,up,down}_proj` |
| Hyper-connection (Gated Residual) | `{attn,mlp}_hyper_connection.*`, `hyper_connection_mixer.*` |
| PLE N-gram embedding | `ple.ple_embedding.ngram_embedding.shard_{0..127}`, `ple.{key_proj,value_proj,conv1d}`, `ple.norm_*` |
| MTP block | `mtp.*` |
| Embeddings / head / vision | `embed_tokens`, `lm_head`, `model.visual.*` |

### Where the bytes are

| Component | Parameters | BF16 | NVFP4 |
|---|---:|---:|---:|
| Routed experts | 120.80 B (67.1%) | 225.00 GiB | **63.28 GiB** |
| PLE N-gram embedding | 51.23 B (28.5%) | 95.43 GiB | 95.43 GiB |
| MTP block | 2.61 B | 4.86 GiB | 4.86 GiB |
| Attention / shared experts / norms / head | 4.91 B | 9.15 GiB | 9.15 GiB |
| Vision tower | 0.45 B | 0.84 GiB | 0.84 GiB |
| **Total** | **180.00 B** | **335.28 GiB** | **173.56 GiB** |

The PLE N-gram embedding is a 320,001,536 × 160 hashed-trigram lookup table — 28.5% of the
parameters at almost zero compute per parameter. It is left in BF16 because quantizing a table
that is read, not multiplied, buys nothing in speed; instead it is the natural candidate for
host memory (see below), which is what brings this model down to two GPUs.

## Calibration

512 conversations of exactly 4,096 tokens, rendered through the Qwen chat template and drawn
from the workloads this model is built for rather than generic web text:

| Category | Share |
|---|---:|
| Agentic tool use | 20.51% |
| SWE agent trajectories | 14.26% |
| Instruction following | 8.20% |
| Terminal agents | 7.81% |
| Code | 7.03% |
| STEM | 5.86% |
| Reasoning | 4.69% |
| Knowledge MCQ | 2.34% |
| **Korean** (code 13.67%, reasoning 6.84%, general 4.88%, agentic 3.91%) | **29.30%** |

27 sources in total. 65.6% of the samples carry reasoning traces inside `<think>` blocks and
38.9% contain `<tool_call>` / `<tool_response>` turns, so the calibration activations cover the
routing patterns of reasoning and tool use rather than plain prose.

## Requirements

```
vllm      built with qwen4_exp support   (vllm/vllm-openai:qwen38-flash-next or newer)
flashinfer >= 0.6.17
```

> **Pin the MoE kernel.** vLLM's automatic NVFP4 MoE backend selection picks
> `FLASHINFER_TRTLLM`, which **silently corrupts decode on this model**: requests return
> HTTP 200 and the output degenerates into a single repeated token. This is not specific to this
> release — the reference NVFP4 checkpoint fails identically. Always pass
> `--moe-backend flashinfer_cutlass` (Blackwell datacenter, sm100/sm103) or
> `--moe-backend marlin` (RTX PRO 6000 / GB202, sm120).

## Quick Start

**RTX PRO 6000 × 2** — PLE in host memory

```bash
VLLM_PLE_CPU_OFFLOAD=1 CUDA_VISIBLE_DEVICES=0,1 \
vllm serve nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4 \
  --served-model-name nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4 \
  --tensor-parallel-size 2 \
  --moe-backend marlin \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

**B200 × 2** — PLE resident on GPU

```bash
CUDA_VISIBLE_DEVICES=0,1 \
vllm serve nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4 \
  --served-model-name nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4 \
  --tensor-parallel-size 2 \
  --moe-backend flashinfer_cutlass \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --no-enable-flashinfer-autotune \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

**RTX PRO 6000 × 4** — add `--enable-expert-parallel`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
vllm serve nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4 \
  --served-model-name nota-ai/Qwen3.8-Flash-Next-Nota-NVFP4 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --moe-backend marlin \
  --gpu-memory-utilization 0.95 \
  --max-num-seqs 16 \
  --no-enable-flashinfer-autotune \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice
```

`--enable-expert-parallel` is required at TP=4, not optional. `moe_intermediate_size` is
640 = 2⁷ × 5, so tensor parallelism shards it to 160 per rank and the NVFP4 MoE kernels reject
the alignment (`flashinfer_cutlass`: "Intermediate size padding for w1 and w3 … not currently
supported"; `flashinfer_cutedsl`: "dim 1 size 320 must be divisible by 128"). Expert
parallelism splits the MoE by expert instead, leaving `moe_intermediate_size` at 640 on every
rank; 512 experts divide evenly by 1, 2, 4 and 8.

Do not add `--enable-eplb`. Qwen3.8-Flash-Next's `reasoning_effort` parameter
(`low` / `medium` / `xhigh`) works unchanged.

