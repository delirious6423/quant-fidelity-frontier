---
library_name: transformers
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- nvfp4
- modelopt
- fp4
- dense
- multimodal
- gated-deltanet
- hybrid
- mtp
- speculative-decoding
---

# Qwen3.8-27B NVFP4 (ModelOpt, uniform W4A4)

NVFP4 (W4A4) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with the NVIDIA TensorRT Model Optimizer (ModelOpt). This is a derivative
checkpoint; see the [base model card](https://huggingface.co/Qwen/Qwen3.8-27B) for the
model's capabilities and general usage.

| | |
|---|---|
| **Base model** | [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Qwen3.8 dense, multimodal, apache-2.0) |
| **Architecture** | `Qwen3_5ForConditionalGeneration`, Gated-DeltaNet hybrid (64 layers: 48 linear-attention, 16 full-attention), dense FFN, 27B |
| **Precision** | uniform W4A4 NVFP4 across dense FFN, full attention and the Gated-DeltaNet path; FP8 KV cache with baked scales. Vision tower, embeddings, `lm_head` and the NEXTN/MTP head kept BF16. |
| **Size** | 55.6 GB BF16 source, 20.6 GB quantized (2.7x): 12.2 GB packed NVFP4 weights, 6.9 GB BF16 (embeddings, `lm_head`, vision, MTP), 1.5 GB FP8 block scales |
| **Context** | 262144 native (1M needs YaRN RoPE scaling, not configured here) |
| **GSM8K** | 97.42% (1285/1319), zero errors, zero empty responses. See the benchmark table. |
| **Producer** | NVIDIA ModelOpt via the kikube single-node quant driver |

> **Serving note:** this is a *uniform* W4A4 build, i.e. attention is quantized too.
> NVIDIA's own Qwen3.5/3.6 NVFP4 checkpoints are MoE/FFN-only and leave attention in
> BF16, so SGLang's stock `qwen3_5` path has never had to handle this. See the
> requirements below before serving.

### Speculative decoding

Use **DSPARK** with the external drafter
[RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)
(1.36B BF16, 2.6 GB). **NEXTN on this checkpoint's built-in MTP head crashes the SGLang
scheduler** on the first request (FlashInfer `fast_prefill_plan`); DSPARK does not.

```
--speculative-algo DSPARK \
--speculative-draft-model-path <drafter> \
--speculative-draft-model-quantization unquant \
--speculative-dspark-block-size 7 \
--speculative-num-draft-tokens 8 \
--speculative-num-steps 1 \
--speculative-eagle-topk 1 \
--mamba-scheduler-strategy extra_buffer \
--attention-backend flashinfer
```

Three of these are not optional:

- `--speculative-draft-model-quantization unquant` — SGLang otherwise inherits this
  target's `modelopt_fp4` for the BF16 drafter and the load dies on a state-dict shape
  mismatch.
- `--speculative-num-draft-tokens 8` — must equal `dspark_block_size + 1`; SGLang asserts
  it. The drafter's `config.json` declares `block_size: 7`.
- `--mamba-scheduler-strategy extra_buffer` — required for correct Gated-DeltaNet state
  rollback when draft tokens are rejected.

`--attention-backend triton` is not a substitute for `flashinfer` here: it hits a
forward-time crash in `RadixLinearAttention.forward` for this architecture
([sgl-project/sglang#29577](https://github.com/sgl-project/sglang/issues/29577)).
DSPARK pins `num_steps` and `eagle_topk` to 1 itself. `SGLANG_ENABLE_SPEC_V2` is a no-op on
SGLang 0.5.17 (the V2 worker always runs); the corresponding knob is
`--disable-overlap-schedule`.

**Measured throughput** on one DGX Spark (GB10, sm121), SGLang 0.5.17, over 415 decode
batches during the GSM8K run: **~186 tok/s aggregate** at 11-13 concurrent requests
(roughly 15 tok/s per stream), mean accept len **3.77** of a ceiling of 8, accept rate
0.38-0.42. No same-settings run without speculation was measured, so this is not a speedup
factor. Note the Mamba state cache, not the GPU, is what caps concurrency here: state usage
sat at 0.68-0.80 with requests queued, and speculation consumes more state slots than plain
decoding.

<!-- BEGIN quant:preview -->

## Quantization details (auto-generated)

- **source model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
- **qformat:** `nvfp4`  **kv_cache:** `fp8`
- **calibration:** `512` samples from each of `pg19`, `cnn_dailymail`
- **producer:** NVIDIA ModelOpt `0.45.0`
- **generated:** 2026-08-15T11:44:42Z

_Before/after sample generation was skipped for this run (`SKIP_GENERATE=1`)._

<!-- END quant:preview -->

<!-- BEGIN quant:remarks -->

## Notes

### Validation status -- structurally verified and smoke-tested coherent

Quantized and served on a single DGX Spark (GB10/sm121) on 2026-08-15. The export passed every structural check: the vision tower is BF16 (333 tensors, zero scale tensors, dtypes and shapes identical to the source), `quant_algo` is `NVFP4` rather than `MIXED_PRECISION`, the FP8 KV scales sit on exactly the 16 full-attention layers (indices 3, 7, ..., 63), `in_proj_qkv` is NVFP4-packed on all 48 Gated-DeltaNet layers while `conv1d`/`in_proj_a`/`in_proj_b` stayed BF16, and all 15 `mtp.*` tensors are present and unquantized. Serving on SGLang 0.5.17 then produced coherent output on all four probe types (German two-sentence explanation, a multi-step train word problem, a five-sentence historical paragraph, a translation), each finishing with `finish_reason=stop`. The word problem was solved correctly (17:00, with the right derivation), which is the more informative signal: a broken Gated-DeltaNet path degrades into word salad rather than into arithmetic mistakes. A full GSM8K run followed on the same day and settled the quality question: 97.42% (1285/1319) with zero errors and zero empty responses, run over the chat endpoint with DSPARK speculative decoding active. See the benchmark table. Uniform W4A4 including the Gated-DeltaNet path is therefore not merely coherent on this architecture, it holds reasoning accuracy. What remains unmeasured is the IMAGE path: calibration was text-only, and no multimodal benchmark has been run against this build.

### Uniform W4A4 on a Gated-DeltaNet hybrid

Unlike NVIDIA's Qwen3.5/3.6 NVFP4 releases, which quantize the FFN/expert path and leave attention in BF16, this build quantizes attention as well, including the Gated-DeltaNet linear-attention path that carries 48 of the model's 64 layers. That is the deliberate point of the profile and it is also where the risk sits: an inadequately loaded scale on the fused `linear_attn.in_proj_qkv` degrades this architecture into complete word salad rather than into a measurable accuracy drop. Judge the build on generated output, not on the fact that it loads.

### Serving requires the qwen3_5 attention-quant and KV-scale loader changes

Because attention is quantized and FP8 KV scales are baked into the checkpoint, SGLang needs the `qwen3_5` attention-quant override and baked-KV-scale loader changes (sgl-project/sglang PR #31220) plus the NVFP4 scalar-scale fix for merged and fused linears (PR #29151, merged upstream 2026-07-13) that the fused Gated-DeltaNet `in_proj_qkv` depends on. Use the `flashinfer` attention backend: the `triton` backend hits a forward-time crash in `RadixLinearAttention.forward` for this exact configuration (sgl-project/sglang#29577, still open). A reliable check that the KV scales actually loaded is that the server logs "Using FP8 KV cache but no scaling factors provided" zero times.

### Multimodal, vision tower kept BF16

Calibration is text-only, so the 27-layer vision tower, its merger and the embeddings are excluded from quantization and stay BF16, avoiding the amax=0 degenerate-quant failure mode. The language-side FFNs that consume projected image tokens ARE quantized, and they were calibrated on text alone, so the image path is the least-validated surface of this build and should be checked against the BF16 source before being relied on.

### Speculative decoding -- use DSPARK, NEXTN crashes here

Measured on this checkpoint, SGLang 0.5.17 on a DGX Spark (GB10/sm121). DSPARK with the external drafter `RadixArk/Qwen3.8-27B-DSpark` works: the drafter loads in 18.5 s (2.95 GB, gamma=7, verify_num_draft_tokens=8), draft CUDA-graph capture completes, and a full 1319-item GSM8K run finished at 97.42% with zero errors and zero empty responses, mean accept len 3.77 of a ceiling of 8. NEXTN on the built-in MTP head, by contrast, killed the scheduler on the first real request with a FlashInfer exception in `fast_prefill_plan` -> `call_begin_forward` -> `update_single_wrapper` -> `init_forward_metadata_out_graph`. Without speculation the same checkpoint serves fine, so the failure is specific to the NEXTN path rather than to speculation in general. Caveat on that comparison: the two runs also differed in context length (32768 vs 65536), and the crashed container's log was lost before the final exception line could be captured, so the exact cause is narrowed but not proven. The MTP head itself ships unquantized regardless (transformers drops `mtp.*` at load for every Qwen3.5 architecture, so ModelOpt never sees it). `serving.speculative.enabled` stays `false` in the shipped profile because the working configuration is DSPARK with an external drafter, which needs the `draft_model_*` keys documented in the speculative block above, not the NEXTN defaults that sit there.

### Quantize on a single Spark with offload, not force-on-GPU

Quantize with `SEQ_DEVICE_MAP=0` (offload / `device_map=auto`). GPU and CPU share the same ~121 GB unified memory pool on a DGX Spark, so offloading the 55.6 GB source costs nothing here. Do NOT use `SEQ_DEVICE_MAP=1` with a high `GPU_MAX_MEM_PCT` on a single Spark: that reserves most of the shared pool as GPU and OOM-kills weight loading regardless of the percentage. Force-on-GPU is a multi-GPU (4x H200) setting only.

<!-- END quant:remarks -->

<!-- BEGIN quant:benchmarks -->

## Benchmarks

| Task | Metric | Value | Setup | Hardware | Date | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K (1319 items, 5-shot) | accuracy | 97.42% (1285/1319) | chat endpoint, max_tokens 16384, temperature 0.6, concurrency 16, DSPARK speculative decoding | 1x DGX Spark (GB10, sm121), SGLang 0.5.17 | 2026-08-15 | strict and flexible extraction agree exactly (1285 both), which is the signal that the chat endpoint was the right choice: on the completions endpoint a thinking model over-generates and strict falls below flexible, an artifact that reads like quantization damage. Zero errors and zero empty responses across all 1319 items, so the number is not propped up by dropped requests. |
| DSPARK speculative decoding | mean accept len | 3.77 of 8 (accept rate 0.38-0.42) | drafter RadixArk/Qwen3.8-27B-DSpark, block_size 7, num_draft_tokens 8, num_steps 1, eagle_topk 1, draft quant unquant, flashinfer, mamba extra_buffer | 1x DGX Spark (GB10, sm121), SGLang 0.5.17 | 2026-08-15 | Measured over 415 decode batches. The value scattering around half the ceiling is the healthy case: an accept len pinned AT the ceiling of 8 would mean the verify stage has stopped rejecting, i.e. silent corruption rather than speedup. NEXTN on the built-in MTP head crashes the scheduler on this checkpoint and is not a usable alternative. |
| Decode throughput (with DSPARK) | tokens/s | ~186 aggregate (~15 per stream) | 11-13 concurrent requests, context 32768, max_tokens 16384, temperature 0.6, FP8 KV cache | 1x DGX Spark (GB10, sm121), SGLang 0.5.17 | 2026-08-15 | Aggregate across concurrent requests, not single-stream. No same-settings run without speculation was measured, so no speedup factor is claimed. Concurrency here is capped by the Mamba state cache rather than the GPU: state usage sat at 0.68-0.80 with 3-5 requests queued behind 11-13 running, and speculative decoding consumes more state slots than plain decoding. |

<!-- END quant:benchmarks -->
