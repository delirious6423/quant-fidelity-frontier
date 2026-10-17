---
library_name: vllm
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: image-text-to-text
tags:
  - qwen
  - int4
  - auto-round
  - quantization
  - xpu
  - intel-arc
  - vllm
---

# Qwen3.8-27B INT4 (AutoRound)

INT4 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), produced with Intel AutoRound.

- **Base model:** Qwen/Qwen3.8-27B (Apache-2.0)
- **Quantization:** AutoRound, INT4, group size 128 (w4g128), symmetric, with 16-bit exceptions for `linear_attn.in_proj_a/b` and `mtp.fc`
- **Size:** ~18 GB (vs ~52 GB for BF16)
- **Context:** 256K
- **Architecture:** qwen3_5 (Qwen3_5ForConditionalGeneration), 64 layers, 3:1 linear:full attention

## Why

Runs a 27B dense model on 4x Intel Arc Pro B70 (32 GB each) with vLLM tensor parallelism. INT4 fits comfortably and runs faster than BF16 on this hardware.

## Load with vLLM

```bash
vllm serve devan-carlin/Qwen3.8-27B-int4-AutoRound \
  --quantization auto-round \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code
```

Note: loading symmetric INT4 checkpoints with empty-qzeros layers on the XPU/ARK path requires the qzeros guard fix ([vllm PR #52428](https://github.com/vllm-project/vllm/pull/52428)). Without it, symmetric layers crash with a `copy_() shape mismatch` during weight loading.

## Benchmark (4x Intel Arc Pro B70, TP=4, max_tokens=16384)

| Model | Gen speed | Notes |
|-------|-----------|-------|
| **INT4 (this repo)** | **47.8 tok/s** | 18 GB |
| Qwen3.8-27B BF16 | 30.2 tok/s | ~52 GB |

- **58% faster** than BF16
- **Quality parity** on math, logic, algorithm, and code prompts (identical final answers)
- INT4 reasons more verbosely (more completion tokens), so on very long outputs it can hit a fixed `max_tokens` cap earlier — raise the cap if needed

## Quantization details

- Method: AutoRound (Intel)
- Bits: 4, group size: 128, symmetric
- Only `model.language_model.layers` is quantized to INT4; embeddings, `lm_head`, the MTP head, and the vision tower stay in their original precision (model stays MTP-capable)
- Within the quantized layers, `linear_attn.in_proj_a/b` are kept 16-bit

## License

Apache-2.0 (same as the base model).

## Notes

- **Multimodal.** The repo ships the Qwen3-VL vision tower (`model.visual.*`) plus processor configs, and the vLLM `Qwen3_5ForConditionalGeneration` port loads it. Send images as `image_url` content parts (base64 or URL); text-only prompts work unchanged.

---

*Quantized and benchmarked by devan-carlin on 4x Intel Arc Pro B70 (XPU).*
