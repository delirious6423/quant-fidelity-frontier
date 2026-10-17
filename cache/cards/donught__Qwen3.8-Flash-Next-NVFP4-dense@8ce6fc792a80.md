---
base_model: Qwen/Qwen3.8-Flash-Next
license: other
license_name: qwen-community-1.0
license_link: LICENSE
tags: [nvfp4, modelopt, qwen3.8, flash-next, moe, vllm]
---

# Qwen3.8-Flash-Next — NVFP4 including the dense path

An in-house NVFP4 (W4A4) quantization of `Qwen/Qwen3.8-Flash-Next` (176B total, ~6B active,
512 routed experts, 51.2B-param PLE n-gram table, 4B MTP head). In production on a single
RTX PRO 6000 Blackwell (96 GB, sm_120) since 2026-09-03.

## How it differs from the public quants

Public NVFP4/FP8 releases of this model (Inferact, RadixArk, Qwen FP8) quantize **only the routed
experts** (`mlp.experts`). `primitive-ai/…-mixed-NVFP4-FP8` additionally puts **FP8** on the
attention and Gated-DeltaNet projections. This checkpoint takes the dense per-token path to
**NVFP4** as well:

| component | this checkpoint | experts-only quants | primitive-ai mixed |
| --- | --- | --- | --- |
| routed experts (48 layers × 512) | NVFP4 | NVFP4 | NVFP4 |
| shared expert gate/up/down (144) | **NVFP4** | BF16 | BF16 |
| GDN `in_proj_qkv` / `in_proj_z` / `out_proj` (108) | **NVFP4** | BF16 | FP8 |
| QSA `o_proj` (12) | **NVFP4** | BF16 | FP8 |
| QSA q/k/v | BF16 (see below) | BF16 | FP8 |
| lm_head, embeddings, routers, gates, conv, norms, mHC, QSA indexer | BF16 | BF16 | BF16 |
| PLE n-gram table (128 shards, ~102 GB) | BF16, byte-identical to source | BF16 | BF16 |
| MTP head (31 tensors) | BF16, byte-identical to source | varies | BF16 |
| vision tower | BF16 | BF16 | BF16 |

**q/k/v were calibrated but ship BF16.** The Flash-Next vLLM image builds the fused QSA `qkv_proj`
with `without_modelopt_fp4(quant_config)` and cannot load NVFP4 there. `lm_head` stays BF16 because the
image builds it without `quant_config`.

Size: 171 GB on disk (39 shards). ~76 GiB resides on the GPU; the PLE table is served from host RAM.

## Measured on RTX PRO 6000 (vLLM, TP=1, PLE host offload, MTP 3 drafts)

Compared with `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` on the same box:

| | primitive-ai | this checkpoint |
| --- | --- | --- |
| resident weights | 77.02 GiB | **76.09 GiB** |
| KV pool at gpu-mem-util 0.95 | 332,993 tokens | **367,001 tokens** (+10 %) |
| 4-stream aggregate decode | 476.5 tok/s | **511.1 tok/s** |
| single-stream decode | ~165–181 tok/s | ~165–188 tok/s (**a tie** within run-to-run noise) |
| GSM8K n=200, thinking on | 97.5 % | **98.0 %** |
| mean completion tokens per GSM8K answer | 378.9 | 450.9 (**+19 %**) |

Single-stream decode does not improve because batch-1 sparse-MoE decode on sm_120 is limited by SM
occupancy, not memory bandwidth (memory interface ~51 % busy during decode). The gains are capacity and
concurrency. Coherence, tool calling (`tool_choice: auto`), 36K needle and image input all pass.

## How it was made

Quantized 2026-09-02 on a rented **Lambda 1× GH200 480GB** with NVIDIA ModelOpt 0.46.0
(torch 2.14.0+cu130, transformers 5.16.0.dev0).

- **Format:** ModelOpt `NVFP4_DEFAULT_CFG`: E2M1 weights and activations, group size 16, FP8 E4M3 block
  scales, FP32 global scale, static calibrated input scales. **Max calibration.**
- **Calibration set:** 512 samples × 1024 tokens (335,106 tokens) rendered through the model's chat
  template: UltraChat-200k 230, CodeFeedback 179, GSM8K-train 103. Batch 16, seed 1234.
- **Fitting a 360 GB BF16 model on one 96 GB GPU:** the PLE n-gram table was read straight from the
  shards into host RAM and looked up on the CPU at full fidelity (`PLE_MODE=cpu`, the same shape
  production serves). Layers that did not fit on the GPU were streamed per forward pass by accelerate
  over NVLink-C2C (345 GB CPU offload). Python stages ran under `numactl --membind` to the CPU node,
  because GH200 exposes HBM as a CPU-less NUMA node and host allocations otherwise spill into it.
- **Recipe checked at runtime:** the enabled ModelOpt quantizers were checked against the planned module
  list before and after calibration.
- **Assembly:** BF16 tensors are copied byte-for-byte from the source; NVFP4 tensors are used for exactly
  the quantized modules. The MTP head and PLE shards come from the source, because transformers drops
  `mtp.*` on load. vLLM reads the exclude list from `config.json` (`ignore`), so it is written there and
  in `hf_quant_config.json`.
- **X-ray:** safetensors headers were checked. Quantized categories are 100 % U8 plus scales, and
  pass-through categories contain no FP4/FP8.
- **Gate on the GH200** (vLLM arm64 image, Marlin W4A16 fallback on Hopper): coherence 4/4, tool call
  PASS, 32K needle PASS, GSM8K n=200 98.5 %. The W4A4 numbers above come from the Blackwell box.

`modelopt_calibration/` holds the calibrated ModelOpt state (`modelopt_state.pth`, `quantizer_amax.pth`),
the recipe, calibration manifest, logs and the assembler. With these, the checkpoint can be re-exported
with a different module set (for example with NVFP4 q/k/v for an engine that accepts it) without
recalibrating.

## Serving

```bash
docker run --rm --gpus all --ipc host --network host --shm-size 32g \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800 \
  -v /path/to/models:/models:ro --entrypoint vllm \
  vllm/vllm-openai:qwen38-flash-next-x86_64-cu130 \
  serve /models/Qwen3.8-Flash-Next-NVFP4-dense --served-model-name flashnext \
  --trust-remote-code --tensor-parallel-size 1 \
  --distributed-executor-backend mp \
  --speculative-config '{"method":"qwen3_8_flash_next_mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --gpu-memory-utilization 0.92 --max-model-len 262144 --enable-prefix-caching
```

- `--distributed-executor-backend mp` is **mandatory** at TP=1. Without it the PLE offload worker is never
  spawned and the engine hangs silently.
- The KV cache must be BF16. The QSA layers reject every quantized `--kv-cache-dtype`.
