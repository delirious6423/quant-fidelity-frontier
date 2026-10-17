---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-Flash-Next
tags: [compressed-tensors, quantized, fp8, moe, vllm]
pipeline_tag: image-text-to-text
library_name: vllm
---

# Qwen3.8-Flash-Next-fp8

`FP8` (per-output-channel weights, dynamic per-token activations) quantization of the **routed
experts** of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
Verified end to end on 4x NVIDIA H100 80 GB — text and image prompts.

| | |
|---|---|
| Quantized | the 48x512 routed experts (`mlp.experts.{gate,up,down}_proj`), 241 of 335 GiB of the model |
| Kept in bf16 | PLE n-gram table, embeddings, attention / Gated-DeltaNet, shared experts, gates, hyper-connections, vision tower, MTP, `lm_head` |
| Size | **220.8 GiB** (source bf16: 335.3 GiB) |
| Tensors | 152,088 |
| Calibration | none — weights are per-channel, activations are dynamic |

## Measured on 4x H100 80 GB

| variant | size | GSM8K (200 q, chat) | mean answer chars |
|---|---|---|---|
| bf16 source | 335.3 GiB | 0.950 | 1183 |
| FP8 | 220.8 GiB | 0.965 | 1146 |
| NVFP4 | 170.2 GiB | 0.960 | 1087 |

At n=200 the standard error is ~1.5 pp, so all three are statistically indistinguishable —
neither quantization costs measurable accuracy on this benchmark. GSM8K was run through the chat
endpoint with the final number parsed off the response, because this model emits a reasoning trace
and lm-eval's strict-match filter expects a completion-style ending.

## How this was built, and why not with llm-compressor

**No transformers version implements `Qwen4ExpForConditionalGeneration`** (checked 5.14.1 and
5.15.1). The architecture lives in vLLM's out-of-tree `vllm.models.qwen3_8_flash_next` plugin, so
the model cannot be instantiated outside vLLM and llm-compressor's `oneshot` pipeline is unusable.

This checkpoint was therefore produced by quantizing the expert tensors directly, shard by shard,
without ever building the model. That is only sound because FP8_DYNAMIC is weight-only: activations
are quantized at runtime, so no calibration pass is required.

The base checkpoint stores routed experts fused as `[512, 1280, 2560]` (gate and up concatenated)
and `[512, 2560, 640]`. vLLM's expert loader expects them per-expert per-projection, so they are
split out. The split order was verified against the reference conversion
(`Inferact/Qwen3.8-Flash-Next-NVFP4`) by row-magnitude correlation: 0.991 for `[:640] -> gate_proj`
and 0.987 for `[640:] -> up_proj`.

The `quantization_config` targets `Linear` and lists 1319 non-expert modules under `ignore`, so
only the routed experts are quantized.

## Serving

```bash
VLLM_PLE_CPU_OFFLOAD=1 vllm serve mbehr90/Qwen3.8-Flash-Next-fp8 \
    --tensor-parallel-size 4 --max-model-len 8192 --max-num-seqs 16
```

`VLLM_PLE_CPU_OFFLOAD=1` keeps the ~95 GiB PLE n-gram table in host memory; ask the scheduler for
plenty of RAM. Requires a vLLM build that registers `Qwen4ExpForConditionalGeneration` — vLLM
0.26.0 does **not**.
