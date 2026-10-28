---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: transformers
pipeline_tag: text-generation
tags:
- glm
- nvfp4
- modelopt
- vllm
- quantized
- blackwell
---

# GLM-5.3-Flash NVFP4 W4A16 for 2× RTX PRO 6000 Blackwell

GLM-5.3-Flash, quantized and tuned to run on **two NVIDIA RTX PRO 6000 Blackwell Workstation Edition GPUs** (96 GB each) with tensor parallelism 2 and no CPU offload. It was converted from the official [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) weights (revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`), and every untouched tensor was verified byte-for-byte against them.

The weights are sized so each card keeps at least 5.5 GiB free after loading. The serving recipe below targets the RTX PRO 6000's SM120 Blackwell architecture, which many off-the-shelf NVFP4 kernels don't support. Other W4A16 builds of this model exist; what this one adds is a full-provenance conversion plus a serving setup proven end to end on this hardware.

Only the routed experts are quantized. Everything else ships exactly as Z.AI released it.

| Part | Format |
|---|---|
| Routed experts (layers 3–45, 288 per layer, gate/up/down; plus the MTP experts in layer 45) | NVFP4 weights, 16-bit activations (W4A16) |
| Dense and shared MLP | Original FP8 block weights, byte-identical |
| Attention (KDA, MLA, indexer), router, norms, embeddings, `lm_head`, vision tower | Original BF16 / native FP8, byte-identical |

Checkpoint size is 178.7 GiB in 62 shards (113,260 tensors), down from 305.8 GiB. The config is ModelOpt `MIXED_PRECISION`: experts use `W4A16_NVFP4`, dense/shared MLP use `FP8_PB_WO`. No activation calibration was needed.

All 1,804 non-expert tensors were checked byte-for-byte against the source. The one format change is lossless: 138 FP32 MLP scale tensors were renamed from `.weight_scale_inv` `[out_blocks, in_blocks]` to `.weight_scale` `[out_blocks, 1, in_blocks, 1]` so stock ModelOpt loads them. Their bytes are unchanged.

## Serving with vLLM

Requires a vLLM nightly. Stock vLLM 0.30.0 fails on the NoPE cache path (`pe_dim must be 64 for fp8_ds_mla`). Validated image: `vllm/vllm-openai:nightly@sha256:8e6e3752946ff2dde451b313cbaec2ea8e7b52221b21bfbcf53fb68dd4f4c42e` (vLLM 0.30.1rc1.dev558, torch 2.13.0+cu130).

```bash
vllm serve <this-repo> \
  --tensor-parallel-size 2 --dtype bfloat16 --quantization modelopt_mixed \
  --moe-backend humming --attention-backend FLASHINFER_MLA_SPARSE_SM120 \
  --kv-cache-dtype fp8_ds_mla --kv-cache-memory-bytes 2415919104 \
  --gpu-memory-utilization 0.98 --max-model-len 262144 --max-num-seqs 8 \
  --disable-custom-all-reduce --no-enforce-eager \
  --additional-config '{"kda_prefill_backend":"triton"}' \
  --kernel-config '{"enable_flashinfer_autotune":false}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}' \
  --reasoning-parser glm47 --tool-call-parser glm47 --enable-auto-tool-choice
```

Text-only serving skips the vision tower; the weights for it are still included. Use the `glm47` parsers, per the [official vLLM recipe](https://recipes.vllm.ai/zai-org/GLM-5.3-Flash).

## Measured results (2× RTX PRO 6000, TP=2)

| Concurrency | Output tok/s (aggregate, incl. prefill) | Mean TTFT | Mean TPOT |
|---|---:|---:|---:|
| 1 | 83.2 | 0.45 s | 11.2 ms |
| 4 | 213.7 | 1.35 s | 15.9 ms |
| 8 | 304.0 | 2.11 s | 21.8 ms |

- Single-stream decode after the first token: about 89.6 tok/s. 16K-token prefill: about 4,700 tok/s (3.5 s TTFT).
- Long context: needle retrieval passed at 32K, 64K, 128K and 464,640 input tokens with a 3.5 GiB/GPU KV profile. The serving profile above uses 262,144.
- Quality spot checks: 29/29 functional acceptance tests; 20/20 factual/arithmetic answers correct (19/20 exact-string); a 30-step math problem fully correct; generated code passed 1,006 test cases; no loops or degenerate output observed.

These are spot checks on one machine, not a benchmark-suite comparison against the full-precision model. Upstream benchmark scores on the base model card apply to the original weights, not this quantization.

## Limitations

- **Tight memory.** Peak VRAM reached 97,242 of 97,246 MiB per GPU at the serving profile, and recoverable allocator retries occurred near maximum context. A 4 GiB KV / 539,648-token profile OOMs during warmup.
- **Thermals.** One GPU reached 94 °C under sustained benchmark load. Plan cooling accordingly.
- **MTP speculative decoding** works and helps single-stream speed, but it costs context headroom and slows batched decode and prefill, so it is off in the profile above.
- Results are specific to the image digest and flags above; other vLLM builds or backends may not fit in memory.

## License and credit

Released under the MIT License, inherited from the base model. Original model © 2026 Z.AI Co., Ltd. This is an unofficial quantization and is not affiliated with or endorsed by Z.AI.
