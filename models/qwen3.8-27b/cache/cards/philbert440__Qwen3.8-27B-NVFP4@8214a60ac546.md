---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - qwen3_5
  - nvfp4
  - fp4
  - compressed-tensors
  - mtp
  - vision
  - thinking
  - v100
---

# Qwen3.8-27B — NVFP4

<p>
  <img src="https://img.shields.io/badge/weights-NVFP4%20(FP4)-2a78d6" alt="NVFP4">
  <img src="https://img.shields.io/badge/size-25.2%20GB-2a78d6" alt="25.2 GB">
  <img src="https://img.shields.io/badge/vision-intact-1baf7a" alt="vision intact">
  <img src="https://img.shields.io/badge/MTP-BF16%20head-1baf7a" alt="MTP BF16 head">
  <img src="https://img.shields.io/badge/V100%20%C3%972-serves-1baf7a" alt="2xV100">
  <img src="https://img.shields.io/badge/license-Apache--2.0-52514e" alt="Apache 2.0">
</p>

NVFP4 (4-bit floating point, weight-only) quantization of
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) — **the whole model, nothing
stripped**: vision tower intact, MTP speculative-decoding head intact in BF16, thinking mode
calibrated with real reasoning traces. 55 GB BF16 → **25.2 GB**.

**Optimized and validated for V100 inference.** This is v2 (2026-08-17) of the checkpoint: the
recipe was chosen by a bake-off of five candidate recipes against the other public Qwen3.8-27B
NVFP4 quants, all served on the same 2×V100 rig with the same battery. It matches the best public
NVFP4 quant on quality (hallucination probe, GSM8K) while running **~37% faster single-stream on
V100** — see [Bake-off](#bake-off-why-this-recipe) below for the full table.

Built as `compressed-tensors` **nvfp4-pack-quantized** deliberately: 1Cat-vLLM's SM70 NVFP4
path accepts this format at compute capability **7.0**, so it serves on
**2×V100-32GB under [1Cat-vLLM](https://github.com/rivetphilbot/1Cat-vLLM) 1.2.2** — whereas
ModelOpt-exported NVFP4 checkpoints require capability 7.5+ and reject Volta. On SM100+
(Blackwell) the FP4 tensor cores run this format natively.

## At a glance

| | |
|---|---|
| **Base model** | Qwen/Qwen3.8-27B — dense 27B vision-language model, 64 layers (16 full-attention + 48 GatedDeltaNet linear-attention), thinking on by default |
| **Format** | `compressed-tensors` **nvfp4-pack-quantized** — E2M1 4-bit weights, FP8-E4M3 per-group scales (group 16), FP32 per-tensor global scale, weight-only |
| **Method** | GPTQ (Hessian-based, sequential over `Qwen3_5DecoderLayer`), **act-order (weight)**, **MSE scale observer**; calibrated on 768 thinking-mode CoT samples + 256 Wikipedia passages @ 2048 tokens |
| **Files** | `model-0000{1,2}-of-00002.safetensors` 24.4 GB (packed body) + `model-mtp.safetensors` 0.81 GB (BF16 MTP head) |
| **Quantized** | 304 linears — all full-attention + MLP projections, GatedDeltaNet `out_proj`/`in_proj_ba` |
| **Kept in BF16** | vision tower (333 tensors) · **GatedDeltaNet `in_proj_qkv` and `in_proj_z`** (the quantization-sensitive projections — see bake-off) · `linear_attn.in_proj_a/b` · `embed_tokens` · `lm_head` · MTP head |
| **Context length** | 262,144 native · extensible to 1M with YaRN (see below) |
| **Reference hardware** | 2×V100-32GB (TP2) via 1Cat-vLLM 1.2.2 · native FP4 on SM100+ |

## Bake-off — why this recipe

Everything below was measured on one rig, one stack, one battery: 2×V100-32GB TP2, 1Cat-vLLM
main (`383bbdd`, incl. PRs [#228](https://github.com/1CatAI/1Cat-vLLM/pull/228)/[#230](https://github.com/1CatAI/1Cat-vLLM/pull/230)),
`FLASH_ATTN_V100`, `fp8_e5m2` KV, MTP K=2 greedy draft, piecewise CUDA graphs. Quality: a
150-item hallucination probe (75 factual questions scored for accuracy, 75 unanswerable/false-premise
questions scored for confabulation — lower is better — judged by grok-4.3) and GSM8K strict-match on
200 items. Speed: mean tokens/s for 512-token thinking generations, 2048-token thinking, 512-token
instruct, and 4-way concurrent aggregate.

| checkpoint | factual ↑ | confab ↓ (of 75) | GSM8K ↑ | tok/s 512 / 2048 / instr / conc4 |
|---|---|---|---|---|
| **this repo (v2 = recipe C)** | 0.973 | **27** | **0.965** | **47.1** / 42.3 / 44.7 / 126.6 |
| this repo v1 (2026-08-13 build) | 0.987 | 39 | 0.940 | 48.9 / 48.0 / 46.2 / 137.9 |
| Inferact/Qwen3.8-27B-NVFP4 (ModelOpt)¹ | 0.973 | 34 | 0.950 | 46.7 / 43.8 / 43.7 / 137.5 |
| unsloth/Qwen3.8-27B-NVFP4 (compressed-tensors, mixed)² | 0.973 | 29 | 0.955 | 34.4 / 35.1 / 33.8 / 107.1 |
| RadixArk/Qwen3.8-27B-NVFP4 (ModelOpt, FP8 attention) | — | — | — | does not load on SM70 (FP8 path has no Volta kernel) |
| recipe A — GDN `in_proj_qkv/z` fp16, 256 CoT calib | 0.987 | 30 | 0.950 | 46.1 / 44.6 / 45.3 / 136.5 |
| recipe B — A + full-attention layers fp16 | 0.973 | 31 | 0.955 | 41.9 / 38.9 / 40.8 / 124.9 |
| recipe D — AWQ smoothing → GPTQ, A ignore-set | 1.000 | 30 | 0.950 | 45.1 / 42.1 / 42.5 / 130.4 |
| recipe E — B ignore-set + C calibration | 0.973 | 24 | 0.950 | 42.0 / 39.1 / 39.4 / 127.4 |

¹ served via PR #228 (ModelOpt NVFP4 gate 75→70). ² served with `kv_cache_scheme` removed from
`config.json` (its FP8 KV scheme has no SM70 route); its checkpoint keeps more layers in 16-bit,
which is where the speed goes on V100.

Noise floor: ±4 confab probes, ±1.5 GSM8K points, ±5% tok/s run-to-run — so C, E and unsloth are
a quality tie, and C is the fastest of that tie by a wide margin. What the campaign taught us about
Qwen3.8's hybrid stack: the **GatedDeltaNet `in_proj_qkv`/`in_proj_z` projections are the
quantization-sensitive spot** (recipe A alone closes most of the gap for ~6% speed); keeping the
full-attention layers in 16-bit buys nothing measurable and costs ~10%; AWQ smoothing before GPTQ
gives a perfect factual score but does not move confabulation or GSM8K; the real quality lever is
calibration — more and more diverse thinking-mode data, act-order and an MSE observer.

## Measured performance

The bake-off table above is the v2 performance reference (same rig, greedy MTP draft, `max_num_seqs 4`,
`fp8_e5m2` KV, piecewise CUDA graphs): **47 tok/s** single-stream thinking @512, 42 @2048, 45 instruct,
**127 tok/s** aggregate at 4-way concurrency, mean MTP acceptance length 2.60. Reported as means over
varied prompts with fixed-length generations (`ignore_eos`) after warmup; the v1 card's 1.2.2 numbers
(53 tok/s / ~170 aggregate, probabilistic draft) were measured on the v1 weights and are not comparable
run-to-run — expect v2 to land within a few percent of v1 single-stream on the same stack, and ~8%
lower at 4-way concurrency, in exchange for the quality gain.

**Concurrency config:** use `{"cudagraph_mode":"piecewise"}` with auto capture sizes (a few
percent ahead of `full_and_piecewise` in our runs). One sizing rule matters: with MTP, each sequence schedules `K+1` tokens per step, so **never set
explicit `cudagraph_capture_sizes` below `max_num_seqs × (K+1)`** — a cap of `[1,2,4,8]` at
batch 4 pushes concurrent decode off CUDA graphs and collapses aggregate throughput ~3×
(measured 55–72 tok/s; reproduces identically on the W4A16 sibling, so it's a config trap, not
a format property). Engine-default auto sizing is correct.

**Pick the draft mode by workload:** verification rejection-samples against the target model,
so output quality is identical either way. At the official temp-1.0 thinking sampling,
**probabilistic** matches the verified distribution and wins (+2–6%); on low-temperature
workloads the two converge ; at `temperature 0` greedy is the natural choice (the bake-off used greedy).

Quality validation (passed on this rig, in addition to the bake-off battery): think-tag
discipline (zero `<think>` leakage with thinking disabled), vision through the VLM path (incl. a
2500×1406 image → 3434 tokens, the case that exposed the 2026-08-15 tokenizer bug), and long-form
generation with no repetition/degeneration.

## The base model

Qwen3.8-27B is the compact dense model of the Qwen3.8 generation: a native vision-language model
with flexible thinking control (`reasoning_effort`, `preserve_thinking`), built for long-horizon
agentic work. The hidden layout is 16 blocks of `3×(GatedDeltaNet→FFN) → 1×(GatedAttention→FFN)`.

Official base-model (BF16) benchmarks, from the
[Qwen3.8-27B model card](https://huggingface.co/Qwen/Qwen3.8-27B) — quantization applies on top
of these; they are not re-measured on this checkpoint:

| Benchmark | Qwen3.8-27B | Qwen3.6-27B |
|---|---|---|
| Terminal Bench 2.1 (Terminus) | **73.0** | 63.4 |
| SWE-bench Pro | 53.5 | 57.6 |
| NL2Repo-Bench | **42.3** | 36.2 |
| GPQA Diamond | **89.2** | 87.8 |
| HLE | **30.8** | 24.0 |
| LiveCodeBench v6 | 83.9 | 89.6 |
| IFBench | 69.1 | 79.1 |
| OmniDocBench 1.5 *(VL)* | **91.1** | 89.4 |
| RealWorldQA *(VL)* | **85.9** | 84.1 |
| ERQA *(VL)* | **65.5** | 62.5 |
| OSWorld-Verified *(VL)* | 63.9 | 73.3 |

## Why this build is careful

### Thinking-mode calibration

Qwen3.8 deploys with thinking **on** by default — so this quant was calibrated in thinking mode.
768 samples @ 2048 tokens of real `<think>…</think>` reasoning traces from
[Magpie-Reasoning-V2-250K-CoT](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)
(filtered to complete traces), applied with `enable_thinking=True`, plus 256 Wikipedia passages
for plain-prose coverage. Calibrating a thinking model
with `enable_thinking=False` shows the quantizer an *empty* think block on every sample — a
calibration/deployment mismatch documented to corrupt `<think>`-tag behavior after quantization
(llm-compressor [#2680](https://github.com/vllm-project/llm-compressor/issues/2680) /
[#2681](https://github.com/vllm-project/llm-compressor/pull/2681)).

### Scale packaging, verified

NVFP4 dequant is `w = e2m1 × weight_scale / weight_global_scale`. This build stores the fp8
block scales **with the global factor folded in** (block-scale max saturates fp8 at 448.0,
FP32 global per tensor) — the packaging convention verified against known-good checkpoints.
GPTQ Hessian calibration ran on the same thinking-mode dataset as the weight quantization.

### MTP head — BF16 by design

The base model's 15-tensor MTP head is grafted back **verbatim in BF16** as
`model-mtp.safetensors` and listed in `quantization_config.ignore`. High-precision MTP heads are
standard NVFP4 practice (NVIDIA's ModelOpt recipes do the same): the head drives speculative
decoding, where draft quality directly sets acceptance rate. One MTP layer; K=2 is the sweet
spot on V100.

## Serving

### 2×V100 / SM70 — 1Cat-vLLM 1.2.2

```bash
VLLM_SM70_QUANT_BACKEND=turbomind VLLM_SM70_FLASH_ATTN_V100=1 VLLM_SM70_NVFP4_TURBOMIND=1 \
VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0 NCCL_P2P_DISABLE=1 \
python -m vllm.entrypoints.openai.api_server \
  --model philbert440/Qwen3.8-27B-NVFP4 \
  --dtype half --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.78 \
  --max-model-len 32768 --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --compilation-config '{"cudagraph_mode":"piecewise"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"FLASH_ATTN_V100","draft_sample_method":"probabilistic"}'
```

SM70 notes, learned the hard way:

- `ninja` must be on `PATH` — the dequant kernels JIT-compile at startup.
- `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0` is required for `max_num_seqs > 1` with MTP.
- `NCCL_P2P_DISABLE=1` on V100 pairs without NVLink — leaving P2P on costs ~4× throughput.
- Boot takes ~3–5 min (weight load + `torch.compile` + CUDA graph capture); poll `/v1/models`.

### Modern GPUs

On SM100+ the compressed-tensors NVFP4 format runs on native FP4 tensor cores in recent vLLM;
on Ampere/Hopper it runs via dequant paths (Marlin). If you only need Ampere+, consider the
[W4A16-AWQ sibling](https://huggingface.co/philbert440/Qwen3.8-27B-W4A16-AWQ) as well.

## Recommended sampling (official)

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

Thinking is on by default; disable per request with
`"chat_template_kwargs": {"enable_thinking": false}`. For agentic work, give it generous output
budgets rather than tight `max_tokens` limits.

## Long context — 262K native, 1M with YaRN

Native 262,144-token context. For longer totals, Qwen's official YaRN override applies:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve philbert440/Qwen3.8-27B-NVFP4 \
  --max-model-len 1000000 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

Static YaRN can slightly hurt short-text quality — enable only when needed, and size `factor`
to your real context.

## Provenance

Quantized from the official BF16 release with
[llm-compressor](https://github.com/vllm-project/llm-compressor) 0.13 (`GPTQModifier`, act-order
`weight`, `mse` observer; recipe in `recipe.yaml`, variant parameters in `variant.json`). No
fine-tuning, no ablation, no chat-template edits — tokenizer, chat template, and preprocessor
configs are verbatim from the base model.

## Citation

Per the Qwen team, for the base model:

```bibtex
@misc{qwen38,
    title = {{Qwen3.8-Max}: A New Bar for Coding and Cowork},
    url = {https://qwen.ai/blog?id=qwen3.8},
    author = {{Qwen Team}},
    month = {August},
    year = {2026}
}
```

## Changelog

- **2026-08-17 — v2 weights (recipe C).** Whole checkpoint replaced with the bake-off winner
  described above: GatedDeltaNet `in_proj_qkv`/`in_proj_z` kept in BF16, 4× larger and more diverse
  thinking-mode calibration set, act-order + MSE observer. Confabulation 39→27/75, GSM8K
  0.940→0.965 on our battery, at 96% of v1's single-stream V100 speed. Size 19.8→25.2 GB (still
  fits 2×V100-32GB with 32K context at `gpu-memory-utilization 0.78`). Files are now sharded
  (`model-0000{1,2}-of-00002.safetensors`); the v1 single `model.safetensors` is gone — re-clone.
- **2026-08-15 — tokenizer fix.** `tokenizer.json` / `tokenizer_config.json` were re-serialized by the llm-compressor calibration run and shipped with an active `truncation` block (`max_length` 1024/2048) plus a drifted pre-tokenizer regex (and, on the Qwen3.6-based repos, 7 phantom audio/TTS special tokens the base model does not define). That broke image inputs larger than the limit under transformers 5 / vLLM (`Mismatch in image token count`, surfacing as an HTTP 400 `Failed to apply Qwen3VLProcessor`). Both files are now byte-identical to the upstream base model's (vocab/merges/added tokens were always identical — this is a metadata-only restore). If you downloaded before this date, re-fetch those two files. Thanks to @elBuffo for the report.
