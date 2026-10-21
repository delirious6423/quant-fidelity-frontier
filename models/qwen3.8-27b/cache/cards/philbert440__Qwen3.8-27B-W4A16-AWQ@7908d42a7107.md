---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - qwen3_5
  - w4a16
  - awq
  - int4
  - compressed-tensors
  - mtp
  - vision
  - thinking
  - v100
---

# Qwen3.8-27B — W4A16 (AWQ, g128)

<p>
  <img src="https://img.shields.io/badge/weights-INT4%20W4A16-2a78d6" alt="INT4 W4A16">
  <img src="https://img.shields.io/badge/size-19.5%20GB-2a78d6" alt="19.5 GB">
  <img src="https://img.shields.io/badge/vision-intact-1baf7a" alt="vision intact">
  <img src="https://img.shields.io/badge/MTP-validated-1baf7a" alt="MTP validated">
  <img src="https://img.shields.io/badge/V100%20%C3%972-serves-1baf7a" alt="2xV100">
  <img src="https://img.shields.io/badge/license-Apache--2.0-52514e" alt="Apache 2.0">
</p>

INT4 weight-only quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) —
**the whole model, nothing stripped**: vision tower intact, MTP speculative-decoding head intact,
thinking mode calibrated with real reasoning traces. 55 GB BF16 → **19.5 GB**, which puts a
current-generation 27B VLM (with headroom for 32K+ context) on a pair of 32 GB V100s — and it
serves on modern GPUs with stock vLLM, no ceremony.

Validated end-to-end on **2×V100-32GB under [1Cat-vLLM](https://github.com/rivetphilbot/1Cat-vLLM) 1.2.2**
(SM70 TurboMind dequant path): coherence, think-tag discipline, image understanding, math, long-form
generation, and MTP speculative decoding in both draft modes.

## At a glance

| | |
|---|---|
| **Base model** | Qwen/Qwen3.8-27B — dense 27B vision-language model, 64 layers (16 full-attention + 48 GatedDeltaNet linear-attention), thinking on by default |
| **Format** | `compressed-tensors` **pack-quantized** · INT4 · group 128 · asymmetric · weight-only |
| **Scale search** | MSE observer (reconstruction-error minimization, not min-max) |
| **Files** | `model.safetensors` 18.7 GB (packed body) + `model-mtp.safetensors` 0.85 GB (BF16 MTP head) |
| **Quantized** | 400 linears — all attention + MLP projections, including 144 GatedDeltaNet projections |
| **Kept in BF16** | vision tower (333 tensors) · `linear_attn.in_proj_a/b` · `embed_tokens` · `lm_head` · MTP head |
| **Context length** | 262,144 native · extensible to 1M with YaRN (see below) |
| **Reference hardware** | 2×V100-32GB (TP2) via 1Cat-vLLM 1.2.2 · any Ampere+ GPU via stock vLLM |

## Measured performance

![MTP draft-mode throughput](images/throughput-v100.png)

**Methodology:** warm serve (2 discarded warmup generations), fixed-length generations via
`ignore_eos` so every run produces exactly the stated token count, official Qwen3.8 sampling per
mode (thinking `1.0/0.95/20`, instruct `0.7/0.80/20 + presence 1.5`), varied prompts. Reported
as mean ± sd tokens/s. Rig: 2×V100-32GB, 1Cat-vLLM 1.2.2, TP2, `fp8_e5m2` KV,
`max_num_seqs 4`, MTP K=2.

| Regime | greedy draft | probabilistic draft |
|---|---|---|
| 512-tok, thinking (n=10) | 55.6 ± 1.7 | **59.3 ± 1.4** |
| 2048-tok, thinking (n=3) | 53.2 ± 1.8 | **56.3 ± 1.1** |
| 512-tok, instruct (n=6) | 54.7 ± 1.9 | 55.5 ± 1.5 |
| 4-way concurrent 512-tok, aggregate (n=3) | 161.9 ± 20.6 | **164.8 ± 23.4** |
| Mean acceptance length, whole workload | 2.34 | 2.52 |

**Pick the draft mode by workload.** Spec-decode verification rejection-samples against the
target model, so output quality is identical in both modes — only acceptance (speed) differs.
At the official temp-1.0 thinking sampling, **probabilistic** proposes from the same
distribution being verified and wins every regime (~+6%). On low-temperature workloads the two
converge (see the instruct row); at `temperature 0`, greedy is the natural choice.

Quality validation (passed on this rig): factual coherence, think-tag discipline (zero
`<think>` leakage with thinking disabled), vision (image understanding through the VLM path),
GSM8K sample 3/3, and long-form generation with no repetition/degeneration.

## The base model

Qwen3.8-27B is the compact dense model of the Qwen3.8 generation: a native vision-language model
with flexible thinking control (`reasoning_effort`, `preserve_thinking`), built for long-horizon
agentic work. The hidden layout is 16 blocks of `3×(GatedDeltaNet→FFN) → 1×(GatedAttention→FFN)`
— that 3:1 linear-attention hybrid is what makes 262K native context practical, and is also
exactly what makes it non-trivial to quantize (see below).

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
256 samples @ 1024 tokens of real `<think>…</think>` reasoning traces from
[Magpie-Reasoning-V2-250K-CoT](https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B)
(filtered to complete traces), applied with `enable_thinking=True`.

Calibrating a thinking model with `enable_thinking=False` shows the quantizer an *empty* think
block on every sample; that calibration/deployment mismatch is documented to corrupt
`<think>`-tag behavior after quantization (llm-compressor
[#2680](https://github.com/vllm-project/llm-compressor/issues/2680) /
[#2681](https://github.com/vllm-project/llm-compressor/pull/2681)). This build never has that
mismatch — and the think-tag battery checks confirm it.

### Full hybrid-attention AWQ smoothing

AWQ smoothing runs across the **entire hybrid topology** — all four projection groups
(`in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`) on the 48 GatedDeltaNet layers as well as
the 16 full-attention layers — not just the vanilla-attention subset. Two architecture-specific
details make that work:

- **llmcompressor pinned to `0.12.1.dev30`** — later 0.12.x has a regression that crashes
  replaying the linear-attention smoothing mapping on GatedDeltaNet models.
- `apply_mask_to_padding_states` neutralized during calibration — fx tracing can't handle its
  shape-dependent control flow, and calibration batches are size 1 (no padding exists to mask).

Sequential calibration over `Qwen3_5DecoderLayer`, `moe_calibrate_all_experts` irrelevant (dense).

### MTP head preserved — and it works

The base model's 15-tensor MTP head is grafted back **verbatim in BF16** as
`model-mtp.safetensors` and listed in `quantization_config.ignore` (transformers drops `mtp.*`
at load time, so the quantizer never sees it; high-precision MTP heads are standard practice —
the head sits at the end of the network and drives speculative decoding, where draft quality
directly sets acceptance rate).

Speculative decoding over the packed body is **validated on SM70**: 92.5% draft acceptance at
K=2. There is one MTP layer; higher `num_speculative_tokens` reuses it with decaying acceptance —
K=2 is the sweet spot on V100.

## Serving

### Stock vLLM (Ampere or newer)

`compressed-tensors` W4A16 is natively supported:

```bash
vllm serve philbert440/Qwen3.8-27B-W4A16-AWQ --max-model-len 32768 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

### 2×V100 / SM70 — 1Cat-vLLM 1.2.2

Volta lacks INT4 tensor-core paths in stock vLLM; the
[1Cat-vLLM](https://github.com/rivetphilbot/1Cat-vLLM) fork's SM70 TurboMind dequant path serves
this checkpoint at the numbers above:

```bash
VLLM_SM70_QUANT_BACKEND=turbomind VLLM_SM70_FLASH_ATTN_V100=1 \
VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0 NCCL_P2P_DISABLE=1 \
python -m vllm.entrypoints.openai.api_server \
  --model philbert440/Qwen3.8-27B-W4A16-AWQ \
  --dtype half --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.78 \
  --max-model-len 32768 --kv-cache-dtype fp8_e5m2 \
  --enable-prefix-caching --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"FLASH_ATTN_V100","draft_sample_method":"probabilistic"}'
```

SM70 notes, learned the hard way:

- `ninja` must be on `PATH` — the dequant kernels JIT-compile at startup.
- `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0` is required for `max_num_seqs > 1` with MTP.
- `NCCL_P2P_DISABLE=1` on V100 pairs without NVLink — leaving P2P on costs ~4× throughput.
- Boot takes ~3–5 min (weight load + `torch.compile` + CUDA graph capture); poll `/v1/models`.

## Recommended sampling (official)

Straight from the Qwen team's best practices — these apply unchanged to this quant:

| Mode | temperature | top_p | top_k | min_p | presence_penalty |
|---|---|---|---|---|---|
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 | 0.0 |
| Instruct (non-thinking) | 0.7 | 0.80 | 20 | 0.0 | 1.5 |

- Thinking is on by default; disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}`. Reasoning depth responds to
  `reasoning_effort`, and the model skips thinking on trivial prompts by design.
- `presence_penalty` can be raised toward 2.0 to curb repetition, at some risk of language
  mixing on the instruct side.
- For agentic work, give it room: Qwen recommends generous output budgets (up to 262K reasoning /
  128K final within the 1M window) rather than tight `max_tokens` limits.

## Long context — 262K native, 1M with YaRN

The checkpoint carries the base model's native 262,144-token context. For longer totals, the
official YaRN override works with this quant on vLLM:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve philbert440/Qwen3.8-27B-W4A16-AWQ \
  --max-model-len 1000000 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
```

Per Qwen's note: static YaRN applies the scaling factor at all lengths and can slightly hurt
short-text quality — enable it only when you need it, and size `factor` to your real context
(e.g. `2.0` for ~524K). For hour-scale video work, see the base card's
`video_preprocessor_config.json` guidance (`longest_edge: 469762048`).

## Provenance

Quantized from the official BF16 release with [llm-compressor](https://github.com/vllm-project/llm-compressor)
(`AWQModifier`, recipe in `recipe.yaml`). No fine-tuning, no ablation, no chat-template edits —
tokenizer, chat template, and preprocessor configs are verbatim from the base model.

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

- **2026-08-15 — tokenizer fix.** `tokenizer.json` / `tokenizer_config.json` were re-serialized by the llm-compressor calibration run and shipped with an active `truncation` block (`max_length` 1024/2048) plus a drifted pre-tokenizer regex (and, on the Qwen3.6-based repos, 7 phantom audio/TTS special tokens the base model does not define). That broke image inputs larger than the limit under transformers 5 / vLLM (`Mismatch in image token count`, surfacing as an HTTP 400 `Failed to apply Qwen3VLProcessor`). Both files are now byte-identical to the upstream base model's (vocab/merges/added tokens were always identical — this is a metadata-only restore). If you downloaded before this date, re-fetch those two files. Thanks to @elBuffo for the report.
