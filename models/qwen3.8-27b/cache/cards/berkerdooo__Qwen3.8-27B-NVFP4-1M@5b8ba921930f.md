---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - nvfp4
  - fp8
  - w4a4
  - compressed-tensors
  - llm-compressor
  - blackwell
  - rtx-5090
  - 1m-context
  - multimodal
  - conversational
---

> [!Note]
> **vLLM only.** Like all quants of this model with an FP8 `lm_head`, SGLang loads this
> checkpoint without error but silently drops `lm_head.weight_scale` and serves corrupted
> logits (verified empirically: needle retrieval succeeds but tokens come out mangled).
> Use vLLM.

# Qwen3.8-27B-NVFP4-1M

A mixed **NVFP4 + FP8** quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
built with [llm-compressor](https://github.com/vllm-project/llm-compressor) 0.13 and
**validated end-to-end at 1,000,000-token context on 2× RTX 5090 (consumer Blackwell, 64 GB total VRAM)** —
including a 4/4 needle-in-haystack retrieval at 989,349 prompt tokens.

The layout follows the mixed-precision profile that makes 1M context physically fit on
32 GB cards (a uniform-W4A4 build leaves ~7 GiB less KV pool), with quantization
calibrated on an English technical mix and **calibrated FP8 KV-cache scales** included.

## What's inside

| component | precision |
|---|---|
| MLP gate/up/down, layers 0–55 | **NVFP4** (W4A4, group 16, GPTQ with static activation ordering) |
| Attention q/k/v/o projections | FP8 W8A8 (channel-wise weights, dynamic per-token activations) |
| GDN (linear attention) in_proj_qkv / in_proj_z / out_proj | FP8 W8A8 |
| MLP layers 56–63, `lm_head` | FP8 W8A8 |
| Vision tower, embeddings, GDN conv1d/in_proj_a/in_proj_b/norms | BF16 |
| MTP draft head (`model_mtp.safetensors`) | BF16, preserved for speculative decoding |
| KV cache scales (`k_scale`/`v_scale`, 16 attention layers) | FP8 per-tensor, static-minmax calibrated |

Total weights: **21.8 GiB** (23,417,591,200 bytes) including the MTP shard.
Calibration: 512 samples × 4,096 tokens — 384 UltraChat chat-templated conversations,
64 long-form English prose documents, 64 code/math/technical prompts. The FP8 `lm_head`
is deliberate: a BF16 head costs ~0.6 GiB/GPU at TP2, which is exactly the margin that
1M context needs on 32 GB cards.

## Serving at 1M context on 2× RTX 5090

```bash
VLLM_USE_DEEP_GEMM=0 \
vllm serve berkerdooo/Qwen3.8-27B-NVFP4-1M \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.96 \
  --max-model-len 1010000 \
  --hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}' \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 4 \
  --attention-backend flashinfer \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

Measured on vLLM 0.27.1, 2× RTX 5090, CUDA 13.2:

- **KV pool: 1,011,551 tokens** (`Maximum concurrency for 1,010,000 tokens per request: 1.00x`)
- **Needle-in-haystack @ 989,349 prompt tokens: 4/4 planted codes retrieved byte-exactly**
- Prefill ~1,330 tok/s averaged over the full 990k prompt (TTFT ≈ 12.4 min cold; prefix caching makes repeat queries fast)
- Decode ~63 tok/s at full 1M context
- FP8 weights cannot reach 1M on these cards (~32 GiB/GPU needed); NVFP4's ~3 GiB/GPU saving is what makes it fit

Boot notes: 0.96 utilization has slim pool margin and can fail intermittently from CUDA-graph
profiling jitter — retrying the boot (or 0.97 with `--max-model-len 1000000`) resolves it.
First boot after changing `--max-model-len` recompiles graphs (10–20 min). If you use
`--kv-offloading-backend native`, clean stale `/dev/shm/vllm_offload_*.mmap` files after any
crashed boot, and expect an occasional warmup race on non-P2P GPU pairs — plain retry works.
At native context (`--max-model-len 262144`, default overrides) none of this applies.

## Quality: token-level divergence vs BF16, FP8, and unsloth's NVFP4

Teacher-forced top-24 logprobs over one held-out 990k-token wikitext stream, identical
token positions for all models. KL is truncated to the reference's top-24 support
(≥99.3% of reference mass observed exactly). ΔNLL is exact. BF16 reference caps at 44k
and FP8 at 460k because nothing larger fits alongside them on 2×32 GB.

**vs BF16 (gold), depth ≤ 44k:**

| quant | KL mean | ΔNLL (≈PPL tax) | top-1 agreement |
|---|---|---|---|
| official FP8 (W8A8) | 0.003–0.004 | +0.003–0.006 (~0.4%) | 96.6–97.0% |
| **this repo** | **0.017–0.020** | **+0.010 (~1.0%)** | **93.4–93.9%** |
| unsloth NVFP4 | 0.019–0.022 | +0.013–0.028 (1.3–2.9%) | 92.4–93.0% |

**vs official FP8 reference, by depth (this repo):**

| depth | KL mean | ΔNLL | top-1 |
|---|---|---|---|
| 0–4k | 0.022 | +0.004 | 92.9% |
| 4–16k | 0.023 | +0.006 | 92.9% |
| 16–48k | 0.020 | +0.008 | 93.0% |
| 48–128k | 0.021 | +0.008 | 93.4% |
| 128–256k | 0.024 | +0.010 | 92.9% |
| 256–460k | 0.023 | +0.010 | 93.1% |

The divergence-vs-depth curve is **flat**: quantization error does not compound as the
context fills. Beyond 460k no same-hardware reference exists; stability is shown by this
model's own NLL by depth on the same stream, which stays in line with shallower buckets
all the way out (460–700k: 1.759, 700–990k: 1.819, vs 128–256k: 1.833) — consistent with
the 4/4 needle result at 989k.

Head-to-head against unsloth's NVFP4 on the same stream, this build has **lower KL to
BF16, lower KL to FP8, higher top-1 agreement, and lower NLL at every one of the eight
depth buckets out to 990k**.

Honest limitations: the divergence eval is a single natural-text (wikitext) stream and the
calibration set is UltraChat-heavy, so part of the edge over unsloth's build is
domain-matched calibration; independent code/math benchmark results are not (yet)
included. Vision quality is untouched by quantization (tower kept BF16) but was not
separately benchmarked.

## MTP speculative decoding

The in-checkpoint MTP draft head is preserved (BF16, `model_mtp.safetensors`). Enable with:

```bash
--speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

At 1M context, skip MTP — the extra weights eat the KV margin. At native context it works
as in the base model.

## Reproduction & provenance

- Base: `Qwen/Qwen3.8-27B` (BF16)
- Toolchain: llm-compressor 0.13.0 / compressed-tensors, GPTQ one-shot, 512×4096 calibration
- The exact quantization recipe ships in this repo as `recipe.yaml`
- Quantization ran layer-sequential on a single RTX 5090 (~11 GB VRAM) with the model in
  host RAM, ~55 minutes end-to-end
- Serving/eval stack: vLLM 0.27.1, FlashInfer attention, `--kv-cache-dtype fp8`

## License

Apache-2.0, same as the base model. You are responsible for complying with the base
model's license and applicable laws.
