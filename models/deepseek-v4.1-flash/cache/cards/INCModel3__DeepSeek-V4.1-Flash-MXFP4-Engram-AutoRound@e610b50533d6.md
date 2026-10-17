---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash
library_name: transformers
tags: [auto-round, mxfp8, mxfp4, ocp-mx, engram, model-free, deepseek]
---

# DeepSeek-V4.1-Flash — D1-ar-v2 (dense OCP-MXFP8 rowwise + engram MXFP4)

A **format-homogenized** variant of DeepSeek-V4.1-Flash: every quantized weight now
follows OCP MX + llm-compressor conventions, produced by a single
[auto-round](https://github.com/intel/auto-round) `--model_free` command (with the
DeepSeek-V4.1 source patch set).

| Component | Official layout | This checkpoint |
|---|---|---|
| dense fp8 family (attn/ shared_experts / engram.wkv / indexer.wq_b / main_proj) | FP8 E4M3 + E8M0, **blockwise 32×32**, `.weight`+`.scale` | **OCP MXFP8 rowwise 1×32**, `.weight`(bytes untouched)+`.weight_scale`(uint8 e8m0, [N,K/32]) — lossless scale expansion (max diff = 0.0) |
| routed experts | packed MXFP4 (E2M1+E8M0/32), `.weight`+`.scale` | unchanged (official bytes, official names) |
| engram embed tables (layers.1/14) | FP8 E4M3+E8M0/32 | **MXFP4** packed (int8 [R,128]+e8m0 [R,8]), rel_err≈0.12 vs fp8 source |
| bf16 families (vision/embed/head/compressor/indexer 部分/norms) | BF16 | unchanged |

Checkpoint size: **412.2 GB** (vs 510.3 GB official, -19.2%). Cross-validated:
byte-identical to an independently script-built twin (96,085/96,085 tensors).

## Inference (vLLM)

vLLM `main` (≥ PR #56201):

```bash
DSV41_ENGRAM_DTYPE=fp4 NCCL_NVLS_ENABLE=0 \
vllm serve <this-model-dir> \
  --tensor-parallel-size 4 --max-model-len 8192 --language-model-only \
  --gpu-memory-utilization 0.90 --port 8100
```

Reference accuracy (same compute recipe, sibling D1 checkpoint): gsm8k 92.19/92.27 vs
92.87 baseline (raw 5-shot, n=1319).

