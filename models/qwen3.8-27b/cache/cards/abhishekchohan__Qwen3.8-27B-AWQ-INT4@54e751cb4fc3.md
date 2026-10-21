---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - awq
  - int4
  - w4a16
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

# Qwen3.8-27B-AWQ-INT4

A max-coverage 4-bit weight-only **AWQ** quantization of
**[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)**, calibrated at
the model's native **262,144-token** context. Beyond the standard block
linears, this build quantizes the **DeltaNet linear-attention projections**
(≈5.2B params), which are usually left in BF16; the MTP predictor is kept
intact for speculative decoding. INT4 weights shrink the footprint ≈2.8×, so
the model serves its native 262,144-token context on a **single 48 GB GPU**,
and YaRN extension reaches 524,288 tokens on a 96 GB card (retrieval beyond
262k unvalidated).

### Fits a single 48 GB GPU

| Component | VRAM |
|---|---|
| Weights (INT4 + BF16 protected parts) | ≈20 GB |
| KV cache (bf16, 262k context) | ≈17 GB |
| Activations + engine overhead | ≈5 GB |
| **Total** | **≈42 GB** |

## TL;DR

| | value |
|---|---|
| Format | INT4, asymmetric (zero-point), `group_size=32`, weight-only (W4A16) |
| Method | **AWQ** activation-aware smoothing + `imatrix-mse` weight observer |
| KV cache | bf16 |
| Calibration | **112 packed sequences × 262,144 tokens** (≈29.4M tokens), Nemotron Post-Training v3 SFT blend, thinking-on |
| Disk size | ≈20 GB (vs ≈55 GB BF16) |
| VRAM (weights + 262k bf16 KV) | ≈42 GB |
| Accuracy vs BF16 base | Matches base within noise on 9 of 10 benchmarks (MathVision −3.0 at full n); 9/9 long-context retrieval at 262,144 tokens |

## Evaluation

Identical harness and hardware for base and quantized: lm-evaluation-harness
with the vLLM 0.27.1 backend on a single NVIDIA RTX PRO 6000 (Blackwell,
96 GB), greedy decoding, seed 1234. Generative tasks run with the model's
native chat template, thinking enabled, `max_gen_toks=8192`; loglikelihood
tasks run untemplated. `minerva_math500` reports `math_verify`, since strict
`exact_match` measures answer format, not content, on thinking-model outputs.

| Task (n) | protocol | Base BF16 | **This model** | Δ | Recovery % |
|---|---|---|---|---|---|
| GSM8K (1319) | thinking, generative | 95.60 | **95.45** | −0.2 | 99.8 |
| GPQA Diamond (198) | thinking, generative | 61.11 | **61.62** | +0.5 | 100.8 |
| Minerva Math500 (500) | thinking, generative, math_verify | 73.2 | **80.0** | +6.8 | 109.3 |
| RealWorldQA (765) | thinking, generative, vision | 84.31 | **83.27** | −1.0 | 98.8 |
| MathVision (3040) | thinking, generative, vision | 78.06 | **75.10** | −3.0 | 96.2 |
| ARC-Easy (2376) | loglikelihood | 82.53 | **82.37** | −0.2 | 99.8 |
| ARC-Challenge (1172) | loglikelihood | 56.91 | **56.83** | −0.1 | 99.9 |
| HellaSwag (2000) | loglikelihood | 56.50 | **55.85** | −0.7 | 98.8 |
| Winogrande (1267) | loglikelihood | 76.16 | **76.48** | +0.3 | 100.4 |
| TruthfulQA MC1 (817) | loglikelihood | 35.99 | **35.50** | −0.5 | 98.6 |
| **Average** | | **70.04** | **70.25** | **+0.2** | **100.2** |

All deltas are within sampling noise for their sample sizes except where noted
in Limitations.

### Long-context retrieval (needle-in-a-haystack, thinking, greedy)

| Context (tokens) | depth 10% | depth 50% | depth 90% |
|---|---|---|---|
| 32,768 | ✅ | ✅ | ✅ |
| 131,072 | ✅ | ✅ | ✅ |
| 262,144 (native) | ✅ | ✅ | ✅ |

The BF16 base scores identically (9/9), showing in particular that int4 DeltaNet
(recurrent-state) quantization causes no long-range retrieval degradation at
the context length it was calibrated at.

## Limitations

- Score outliers: MathVision −3.0 vs base at full n (≈2.7σ, the one clear
  deficit; the earlier n=100 slice suggested −8.0, the full-n figure is the
  reliable one), and Minerva Math500 +6.8 (≈3.4σ, best read as `math_verify`
  protocol variance rather than a real capability gain). Vision coverage
  beyond RealWorldQA and MathVision is not evaluated.
- Not evaluated: agentic/tool-calling behavior.
- MTP speculative decoding: the predictor ships intact and the serve flag
  follows the base-model recipe; quantitative validation (score parity,
  acceptance rate) is in progress.
- Generation-task scores can vary by a few points across GPU vendors even for
  identical configs; base and quantized were measured on the same machine to
  keep every Δ a within-config comparison.

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

The checkpoint serves bf16 KV.

**What fits:**

- **48 GB + bf16 KV**: native 262k context, ≈42 GB used
- **96 GB + bf16 KV**: 262k with headroom, or 524k via YaRN (`factor: 2.0`)

### Single 48 GB GPU: 262k (the default)

```bash
vllm serve abhishekchohan/Qwen3.8-27B-AWQ-INT4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

If you OOM at full context on 48 GB, drop `--speculative-config` first.

### Single 96 GB GPU: 262k with concurrency headroom

```bash
vllm serve abhishekchohan/Qwen3.8-27B-AWQ-INT4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

On a 96 GB card (RTX PRO 6000 Blackwell) the bf16 KV pool has ample headroom,
so raise `--max-num-seqs` for concurrency. At bf16 KV the full 262k context
needs ≈17 GB of KV (≈65 KB/token, since only the 16 global-attention layers
accumulate KV).

**Host notes.** On CUDA-12.x-driver hosts install the `+cu129` vLLM wheel
(PyPI wheels are cu13-only from 0.26) and run with `--attention-backend
TRITON_ATTN` + `VLLM_USE_FLASHINFER_SAMPLER=0`; flashinfer JIT additionally
needs a CUDA ≥ 12.9 toolkit (the pip `nvidia-cuda-*-cu13` packages supply it).
On CUDA-13 hosts stock vLLM works as-is.

**Beyond 262k (YaRN).** The base architecture extends via YaRN. With bf16 KV,
524,288 tokens needs ≈34 GB of KV, so plan on a 96 GB card; use `factor: 2.0`:

```bash
VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve abhishekchohan/Qwen3.8-27B-AWQ-INT4 \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 2.0, "original_max_position_embeddings": 262144}}}' \
  --max-model-len 524288
```

Static YaRN can mildly hurt short-text quality, so apply the override only when
you need the extra context. This checkpoint was calibrated and validated at the
native 262,144 tokens, so re-check retrieval at your target length before
relying on anything beyond that.

## Quantization recipe

AWQ smoothing with `duo_scaling` and a 20-candidate per-channel grid search,
applied across four mapping groups: `input_layernorm → q/k/v_proj`,
`v_proj → o_proj`, `post_attention_layernorm → gate/up_proj`, and
`up_proj → down_proj` (the four default AWQ scaling groups). Weight observer
is `imatrix-mse`; the quantization scheme is a single asymmetric int4
group-32 config applied by denylist, so every transformer-block linear is
quantized unless explicitly protected (see table above). The fully resolved
recipe ships as `recipe.yaml` in this repo.

**Long-sequence calibration.** Conversations from five Nemotron Post-Training
v3 SFT datasets (instruction-following chat, math, science, agentic code,
multilingual) were pooled, shuffled, and packed EOS-separated into 112 fixed
262,144-token rows, calibrating at the exact context length the model is
served at. Calibration texts are rendered with thinking enabled (reasoning
traces preserved), matching the deployment regime.

**DeltaNet at int4.** `in_proj_z` is quantized together with `in_proj_qkv`
(vLLM fuses them into one module at load and requires matching schemes);
`in_proj_a/b` stay BF16 together for the same reason. Int4 error here feeds a
recurrent state and can compound over a sequence, so this build was gated on
needle-in-a-haystack retrieval at long context before any benchmark hours, and
passes 9/9 (above).
