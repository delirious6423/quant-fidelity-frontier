---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - gptq
  - int4
  - w4a16
  - fp8-kv-cache
  - long-context
  - imatrix
  - llm-compressor
  - compressed-tensors
  - vllm
  - speculative-decoding
datasets:
  - nvidia/Nemotron-SFT-Instruction-Following-Chat-v3
  - nvidia/Nemotron-SFT-Math-v4
  - nvidia/Nemotron-SFT-Science-v2
  - nvidia/Nemotron-SFT-SWE-v3
  - nvidia/Nemotron-SFT-Multilingual-v2
---

# Qwen3.8-27B-GPTQ-INT4-FP8KV

A 4-bit weight-only **GPTQ** quantization of **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**
with an **fp8-e4m3 KV cache calibrated alongside the weights**. INT4 weights
shrink the footprint ≈2.8× and fp8 KV halves the cache, so the model serves its
native **262,144-token** context on a **single 48 GB GPU**. fp8 KV also makes
YaRN extension to **524,288 tokens** fit the same card, and up to 1M tokens on
96 GB (retrieval beyond 262k unvalidated).

### Fits a single 48 GB GPU

| Component | VRAM |
|---|---|
| Weights (INT4 + BF16 protected parts) | ≈20 GB |
| KV cache (fp8, 262k context) | ≈8.5 GB |
| Activations + engine overhead | ≈5 GB |
| **Total** | **≈33.5 GB** |

## TL;DR

| | value |
|---|---|
| Format | INT4, asymmetric (zero-point), `group_size=32`, weight-only (W4A16) |
| Method | **GPTQ** (Hessian error compensation) + `imatrix-mse` weight observer |
| KV cache | **fp8-e4m3, symmetric per-tensor, calibrated scales baked into the checkpoint** |
| Calibration | **8 packed sequences × 262,144 tokens** (≈2.1M tokens), Nemotron Post-Training v3 SFT blend, thinking-on |
| Disk size | ≈20 GB (vs ≈55 GB BF16) |
| VRAM (weights + 262k fp8 KV) | ≈28.5 GB |
| Accuracy vs BF16 base | No degradation beyond sampling noise on 10 benchmarks; 9/9 long-context retrieval at 262,144 tokens |

## Evaluation

Identical harness and hardware for base and quantized: lm-evaluation-harness
with the vLLM 0.27.1 backend on a single NVIDIA RTX PRO 6000 (Blackwell,
96 GB), greedy decoding, seed 1234. The base runs with bf16 KV and the
quantized model with fp8-e4m3 KV, each at its recommended serving
configuration, so every Δ measures the full int4 + fp8-KV stack rather than
weights alone. Generative tasks run with the model's native chat template,
thinking enabled, `max_gen_toks=8192`; loglikelihood tasks run untemplated.
`minerva_math500` reports `math_verify`, since strict `exact_match` measures
answer format, not content, on thinking-model outputs.

| Task (n) | protocol | Base BF16 | **This model** | Δ | Recovery % |
|---|---|---|---|---|---|
| GSM8K (1319) | thinking, generative | 95.60 | **95.53** | −0.1 | 99.9 |
| GPQA Diamond (198) | thinking, generative | 61.11 | **63.64** | +2.5 | 104.1 |
| Minerva Math500 (500) | thinking, generative, math_verify | 73.2 | **75.4** | +2.2 | 103.0 |
| RealWorldQA (765) | thinking, generative, vision | 84.31 | **85.10** | +0.8 | 100.9 |
| MathVision (3040) | thinking, generative, vision | 78.06 | **76.74** | −1.3 | 98.3 |
| ARC-Easy (2376) | loglikelihood | 82.53 | **82.32** | −0.2 | 99.7 |
| ARC-Challenge (1172) | loglikelihood | 56.91 | **56.40** | −0.5 | 99.1 |
| HellaSwag (2000) | loglikelihood | 56.50 | **56.00** | −0.5 | 99.1 |
| Winogrande (1267) | loglikelihood | 76.16 | **75.77** | −0.4 | 99.5 |
| TruthfulQA MC1 (817) | loglikelihood | 35.99 | **35.99** | 0.0 | 100.0 |
| **Average** | | **70.04** | **70.29** | **+0.3** | **100.4** |

All deltas are within sampling noise for their sample sizes.

### Long-context retrieval (needle-in-a-haystack, thinking, greedy)

| Context (tokens) | depth 10% | depth 50% | depth 90% |
|---|---|---|---|
| 32,768 | ✅ | ✅ | ✅ |
| 131,072 | ✅ | ✅ | ✅ |
| 262,144 (native) | ✅ | ✅ | ✅ |

The BF16 base scores identically (9/9), showing no retrieval degradation through the
int4 + fp8-KV stack at the context length it was calibrated at.

## Limitations

- Generation-task scores can vary by a few points across GPU vendors even for
  identical configs; base and quantized were measured on the same machine to
  keep every Δ above a within-config comparison.

## What was quantized (and what was protected)

| Component | Quantized (INT4 gs32 asymmetric) | Protected (kept BF16) |
|---|---|---|
| `language_model` (64 hybrid-attention text layers) | block linears: softmax-attn `self_attn.{q,k,v,o}_proj`, MLP `{gate,up,down}_proj`, DeltaNet `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` | layer norms incl. q/k RMS-norms, `embed_tokens`, `lm_head` |
| DeltaNet gating | none | `linear_attn.{in_proj_a,in_proj_b}` and conv states |
| `visual` (frozen ViT encoder) | none | entire `model.visual.*` |
| MTP predictor | none | kept intact for speculative decoding |

Unlike the common practice of keeping DeltaNet projections in BF16, they are
quantized here. The eval suite therefore deliberately includes vision and
long-context retrieval to cover that risk.

## Serving

The checkpoint carries its own calibrated k/v scales, and `--kv-cache-dtype
fp8_e4m3` picks them up automatically.

**What fits:**

- **48 GB + fp8 KV**: native 262k context, comfortable (≈33 GB used)
- **48 GB + bf16 KV**: 262k fits but tight (≈42 GB); drop `--kv-cache-dtype`
- **48 GB + fp8 KV + YaRN (`factor: 2.0`)**: 524k context
- **96 GB + fp8 KV + YaRN (`factor: 4.0`)**: up to 1M context

### Single 48 GB GPU: 262k, fp8 KV (the default)

```bash
vllm serve abhishekchohan/Qwen3.8-27B-GPTQ-INT4-FP8KV \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

fp8 KV is the default on 48 GB; bf16 KV is a 96 GB option. If you OOM at full
context on 48 GB, drop `--speculative-config` first.

### Single 96 GB GPU: 262k with concurrency headroom

```bash
vllm serve abhishekchohan/Qwen3.8-27B-GPTQ-INT4-FP8KV \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

On a 96 GB card (RTX PRO 6000 Blackwell) the fp8 KV pool holds ≈2M tokens
(≈8× the max context), so raise `--max-num-seqs` for concurrency. At bf16 KV
the full 262k context needs ≈17 GB of KV (≈65 KB/token, since only the 16
global-attention layers accumulate KV); drop `--kv-cache-dtype` if you prefer
bf16 KV.

**Host notes.** On CUDA-12.x-driver hosts install the `+cu129` vLLM wheel
(PyPI wheels are cu13-only from 0.26) and run with `--attention-backend
TRITON_ATTN` + `VLLM_USE_FLASHINFER_SAMPLER=0`; flashinfer JIT additionally
needs a CUDA ≥ 12.9 toolkit (the pip `nvidia-cuda-*-cu13` packages supply it).
On CUDA-13 hosts stock vLLM works as-is.

**Beyond 262k (YaRN).** The base architecture extends via YaRN, and fp8 KV is
what makes the long end practical on a single card. For 524,288 tokens on one
48 GB GPU use `factor: 2.0` (for 1M on 96 GB use `factor: 4.0`):

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve abhishekchohan/Qwen3.8-27B-GPTQ-INT4-FP8KV \
  --kv-cache-dtype fp8_e4m3 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 2.0, "original_max_position_embeddings": 262144}}}' \
  --max-model-len 524288
```

Static YaRN can mildly hurt short-text quality, so apply the override only when
you need the extra context. This checkpoint was calibrated and validated at the
native 262,144 tokens, so re-check retrieval at your target length before relying
on anything beyond that.

## Quantization recipe

GPTQ with weight-only INT4: asymmetric (zero-point, stored as int8),
`group_size=32`, `imatrix-mse` weight observer (importance statistics collected
by forward hooks over the calibration rows), `actorder=static`, block size 128,
`dampening_frac: 0.01`, applied to every `Linear` layer not protected in the
table above. The fully resolved recipe ships as `recipe.yaml` in this repo.

**Long-sequence calibration.** Conversations from five Nemotron Post-Training v3
SFT datasets (instruction-following chat, math, science, agentic code,
multilingual) were pooled (≈37M tokens), shuffled (seed 42), and packed
EOS-separated into 8 fixed 262,144-token rows, calibrating at the exact
context length the model is served at. Calibration texts are rendered with
thinking enabled (reasoning traces preserved), matching the deployment regime.

**KV cache.** The same oneshot pass observes k/v activations of the 16
softmax-attention layers (the 48 DeltaNet layers carry recurrent state, not a
KV cache) and bakes the resulting per-tensor scales into the checkpoint
(range across layers: 0.016–0.203).
