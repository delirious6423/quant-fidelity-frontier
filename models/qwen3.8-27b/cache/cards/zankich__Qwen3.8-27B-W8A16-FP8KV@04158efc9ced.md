---
base_model: Qwen/Qwen3.8-27B
quantized_from: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: vllm
pipeline_tag: image-text-to-text
tags:
  - qwen3_5
  - compressed-tensors
  - w8a16
  - int8
  - vllm
  - quantized
  - mtp
  - speculative-decoding
  - fp8-kv
---

# Qwen3.8-27B-W8A16-FP8KV

Quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) built for self-hosted serving on Ampere GPUs: INT8 weight-only weights, bfloat16 activations, and a calibrated FP8 KV cache. Produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.13.0, shipped in the `compressed-tensors` `pack-quantized` format. For model capabilities, benchmarks and usage documentation see the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B).

## Summary

| Property | Value |
|---|---|
| Base model | `Qwen/Qwen3.8-27B` |
| Weights | INT8, group size 128, symmetric, weight-only GPTQ |
| Activations | bfloat16 |
| KV cache | FP8 E4M3, static per-tensor symmetric scales |
| Format | `compressed-tensors`, auto-detected by vLLM from `config.json` |
| Size | 31.6 GB total, 16 shards plus `model_mtp.safetensors`, down from 52 GB bf16 |
| MTP head | preserved at bf16, byte-identical to base |

## What is quantized

Every `Linear` except the ignore list below: all attention and FFN projections in the 16 full-attention layers, and `in_proj_qkv`, `in_proj_z` and `out_proj` in the 48 Gated DeltaNet layers.

## What is kept at bf16

- `lm_head`
- The entire vision tower, all 333 tensors. The checkpoint remains a vision-language model.
- The MTP head, 15 tensors copied byte-identical from the base checkpoint into `model_mtp.safetensors`.
- `linear_attn.in_proj_a` and `linear_attn.in_proj_b`, the two GDN recurrent-gate projections, which are precision-sensitive.

The ignore-list shape, matching the upstream llm-compressor Qwen3.5 guidance of quantizing the rest of the linear-attention path:

```python
KV_CACHE_SCHEME = {
    "num_bits": 8,
    "type": "float",
    "strategy": "tensor",
    "dynamic": False,
    "symmetric": True,
}

recipe = GPTQModifier(
    targets="Linear",
    scheme="W8A16",
    ignore=[
        "lm_head",
        "re:.*visual.*",
        "re:.*mtp.*",
        "re:.*linear_attn[.]in_proj_a$",
        "re:.*linear_attn[.]in_proj_b$",
    ],
    kv_cache_scheme=KV_CACHE_SCHEME,
)
```

## Serving

Upstream vLLM serves this checkpoint. A forked vLLM with additional fixes is also available: [`zankich/vllm`](https://github.com/zankich/vllm), with a prebuilt image at [`zankich/vllm-openai:latest`](https://hub.docker.com/r/zankich/vllm-openai/tags) on Docker Hub. It resolves KV-offload correctness issues on this model's hybrid GDN + MTP layout, surfaces per-request spec-decode metrics through the `/v1/messages` converter, and adds a thinking-budget wrap-up for `thinking.budget_tokens` enforcement under MTP. See the fork README for the full patch list.

## KV cache calibration

The KV cache scales are calibrated during the same forward passes as the GPTQ weights, via `kv_cache_scheme` on the modifier, so no separate calibration step runs. The GDN layers carry no KV cache, so the 16 full-attention layers hold the `k_scale` and `v_scale` tensors. Serve with `--kv-cache-dtype fp8` so vLLM loads them.

## Calibration data

768 samples, 512 sampled per run at max sequence length 2048, stratified for an agentic coding workload:

| Share | Source |
|---|---|
| ~45% | interactive coding-agent turns and retrieval session pairs |
| ~30% | function-level code in Python, TypeScript, Go and shell |
| ~21% | structured markdown knowledge-base prose |

The calibration set is built from private session data and is not published. No accuracy benchmark delta against the bf16 base is published for this checkpoint.
