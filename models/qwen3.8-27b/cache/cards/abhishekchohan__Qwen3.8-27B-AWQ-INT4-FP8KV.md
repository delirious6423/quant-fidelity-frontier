---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- awq
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

# Qwen3.8-27B-AWQ-INT4-FP8KV

[abhishekchohan/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/abhishekchohan/Qwen3.8-27B-AWQ-INT4)
with a calibrated **fp8-e4m3 KV cache**. The weights are byte-identical to the
source AWQ model; this checkpoint adds measured K/V quantization scales so the
model serves its full 262,144-token context with half the KV memory of bf16.

## KV scale derivation

The scales were measured on the quantized model itself, not on the BF16 base,
because activation-aware smoothing and int4 rounding change the K/V activations
the served model actually writes to cache.

Calibration replayed the exact same sequences that were used to calibrate the
AWQ model: **112 packed sequences × 262,144 tokens (~29.4M tokens)** from the
same Nemotron Post-Training v3 SFT blend (instruction-following chat, math,
science, agentic code, multilingual), same shuffle seed, rendered with thinking
enabled, so the measured KV distribution matches the deployment regime the
weights were tuned for, at the exact context length the model serves at.

While the model processed those sequences, we tracked the running maximum
absolute value of the K states (after QK-norm) and V states each of the 16
global attention layers writes to the cache. Each scale is `amax / 448` (the
fp8-e4m3 representable maximum), symmetric, per tensor, so no cached value can
clip. The result is 32 per-layer scalars (`k_scale` / `v_scale`). Only the 16
global attention layers accumulate KV; the 48 DeltaNet linear-attention layers
carry recurrent state and have no KV scales.

## Evaluation

Same harness and hardware as the
[abhishekchohan/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/abhishekchohan/Qwen3.8-27B-AWQ-INT4)
model card.

The comparison base is
[abhishekchohan/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/abhishekchohan/Qwen3.8-27B-AWQ-INT4):
this checkpoint's weights are byte-identical to it. The only configuration
difference is the KV cache: fp8-e4m3 with calibrated per-layer scales here,
bf16 on the base, so every Δ isolates the KV-cache dtype.

| Task (n) | protocol | AWQ-INT4 | **This model** | Δ | Recovery % |
|---|---|---|---|---|---|
| GSM8K (1319) | thinking, generative | 95.45 | **95.91** | +0.5 | 100.5 |
| GPQA Diamond (198) | thinking, generative | 61.62 | **60.61** | −1.0 | 98.4 |
| Minerva Math500 (500) | thinking, generative, math_verify | 80.0 | **79.00** | −1.0 | 98.8 |
| ARC-Easy (2376) | loglikelihood | 82.37 | **82.45** | +0.1 | 100.1 |
| ARC-Challenge (1172) | loglikelihood | 56.83 | **56.66** | −0.2 | 99.7 |
| HellaSwag (2000) | loglikelihood | 55.85 | **56.05** | +0.2 | 100.4 |
| Winogrande (1267) | loglikelihood | 76.48 | **76.56** | +0.1 | 100.1 |
| TruthfulQA MC1 (817) | loglikelihood | 35.50 | **35.74** | +0.2 | 100.7 |

All measured deltas are within sampling noise for their sample sizes.

### Long-context retrieval (needle-in-a-haystack, thinking, greedy, fp8-e4m3 KV)

| Context (tokens) | depth 10% | depth 50% | depth 90% |
|---|---|---|---|
| 32,768 | ✅ | ✅ | ✅ |
| 131,072 | ✅ | ✅ | ✅ |
| 262,144 (native) | ✅ | ✅ | ✅ |

9/9 with fp8-e4m3 KV, identical to the [abhishekchohan/Qwen3.8-27B-AWQ-INT4](https://huggingface.co/abhishekchohan/Qwen3.8-27B-AWQ-INT4) model's 9/9 with bf16 KV:
fp8 KV causes no long-range retrieval degradation at the context length the
model was calibrated at.

## Serving

This checkpoint serves fp8-e4m3 KV with the calibrated per-layer scales
shipped in the checkpoint (vLLM picks them up automatically with
`--kv-cache-dtype fp8_e4m3`). fp8 KV halves the cache footprint versus bf16:
≈33 KB/token instead of ≈65 KB/token (only the 16 global attention layers
accumulate KV).

**What fits (262,144-token context):**

| Component | VRAM |
|---|---|
| Weights (INT4 + BF16 protected parts) | ≈20 GB |
| KV cache (fp8-e4m3, 262k context) | ≈8.5 GB |
| Activations + engine overhead | ≈5 GB |
| **Total** | **≈33.5 GB** |

- **48 GB**: native 262k with ≈14 GB headroom for concurrency. fp8 KV also makes
  a YaRN extension to **524,288 tokens** fit (≈42 GB total).
- **96 GB**: 262k with large headroom; 524k via YaRN with ≈50 GB free.

Retrieval beyond 262,144 tokens is unvalidated; re-check at your target length
before relying on it.

### Single 48 GB GPU: 262k

```bash
vllm serve abhishekchohan/Qwen3.8-27B-AWQ-INT4-FP8KV \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3
```

### Single 96 GB GPU: 262k with concurrency headroom

```bash
vllm serve abhishekchohan/Qwen3.8-27B-AWQ-INT4-FP8KV \
  --max-model-len 262144 \
  --kv-cache-dtype fp8_e4m3 \
  --reasoning-parser qwen3
```

On a 96 GB card the fp8 KV pool leaves ample headroom, so raise
`--max-num-seqs` for concurrency.

**MTP speculative decoding.** The MTP predictor ships intact, but vLLM's
flashinfer JIT path currently rejects MTP together with fp8 KV on Blackwell
hosts, so the commands above leave it off; the evaluation above measures
spec-free serving.

**Host notes.** Same as the AWQ-INT4 card: on CUDA-12.x-driver hosts install
the `+cu129` vLLM wheel (PyPI wheels are cu13-only from 0.26) and run with
`--attention-backend TRITON_ATTN` + `VLLM_USE_FLASHINFER_SAMPLER=0`; on
CUDA-13 hosts stock vLLM works as-is.