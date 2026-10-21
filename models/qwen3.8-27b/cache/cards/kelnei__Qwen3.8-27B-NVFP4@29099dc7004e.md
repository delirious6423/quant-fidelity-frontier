---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- nvfp4
- fp8
- quantized
- compressed-tensors
- vllm
---

# Qwen3.8-27B-NVFP4

NVFP4 (W4A4) quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)
for NVIDIA Blackwell GPUs (SM 12.0+). 21.8 GB on disk and in VRAM,
down from ~56 GB BF16, with a mixed-precision recipe chosen from KL
measurements against the original model.

> **Updated:** the output head is now FP8 rather than BF16. That is 1.2 GB
> off the checkpoint and **+14% single-stream decode throughput** under k=2
> MTP speculative decoding, for +0.0005 mean weight-only KL. See
> [Decode throughput](#decode-throughput). If you want the previous
> fidelity-first build — **or if you serve with SGLang**, which cannot apply
> a quantization scheme to the output head (see [Serving](#sglang)) — pin
> revision
> [`83ceaf9`](https://huggingface.co/kelnei/Qwen3.8-27B-NVFP4/tree/83ceaf9297ab72a9c11044977cbedfb8937c2ee8).

## Recipe

| Component | Scheme |
|---|---|
| Most linear layers | NVFP4 (group-16, W4A4) |
| Attention q/k/v/o + KV cache | FP8 (channel-wise weights, static KV scale) |
| Gated DeltaNet in_proj_qkv / in_proj_z / out_proj | FP8 channel-wise |
| DeltaNet in_proj_a / in_proj_b | BF16 (unquantized) |
| Dense MLP gate/up/down (most layers) | NVFP4 via GPTQ (imatrix observer, static actorder) |
| MLP layers 56–63 | FP8 — selected automatically by ranking every layer on GPTQ's own Hessian-weighted proxy loss (`--fp8-mlp gptq-loss:8`). This is the late activation-outlier region (down_proj input amax runs 300–717 vs ~30 mid-stack), and promoting it measures better end-to-end than any isolated per-layer sensitivity scan predicts |
| Output head (lm_head) | FP8 (channel-wise weights, dynamic per-token activations) |
| Vision tower + merger | BF16 (unquantized) |
| Embeddings, norms | BF16 (unquantized) |
| MTP (multi-token prediction) head | preserved |

Calibrated on a 512-sample multi-source mixture (1024 tokens each).
Quantized with [llm-compressor](https://github.com/vllm-project/llm-compressor)
via [kelnei/nvfp4-vllm](https://github.com/kelnei/nvfp4-vllm), and scanned
for silent weight corruption (bit flips) against the source checkpoint
with the same repo's `verify.py` before upload.

This checkpoint is byte-for-byte what the pipeline produces from the BF16
original with `--fp8-deltanet --fp8-mlp gptq-loss:8 --fp8-lm-head` — no
hand-picked layer list involved.

## Decode throughput

Single-stream decode is memory-bandwidth bound, and the output head is the
largest tensor left once the decoder stack is quantized — 2.37 GiB at BF16
on this vocab. Every decode step reads all of it, and under MTP speculative
decoding it is read once per draft position too, because the draft head
shares `lm_head` with the target. At k=2 that is three passes per step.

Measured against a build differing in this one tensor and nothing else
(RTX PRO 6000 Blackwell Workstation, vLLM 0.27.1, `--kv-cache-dtype fp8`,
identical flags on both arms):

| tok/s | BF16 head | FP8 head |
|---|---|---|
| k=2 MTP, single stream | 81.1 | **92.6** (+14.2%) |
| k=2 MTP, 8 concurrent | 586.9 | **679.5** (+15.8%) |
| no speculation, single stream | 53.0 | **55.4** (+4.5%) |
| no speculation, 8 concurrent | 397.5 | **413.9** (+4.1%) |

A bytes-moved model accounts for both: one head pass per step predicts
+5.5%, three passes predicts +14.8%. Draft acceptance costs 1.7 points
(56.4% → 54.7%) since the draft logits come out of the same quantized head,
and the KV cache gains 2.5% from the freed VRAM.

## Fidelity

Full-vocab KL divergence against the BF16 original, teacher-forced over
30 chat prompts × 384 tokens. "Weight-only" measures the quantized
weights with BF16 activations; "emulated" also fake-quantizes
activations to approximate W4A4 serving.

| Metric | Weight-only | W4A4 emulated |
|---|---|---|
| Mean KL | 0.0134 | 0.0259 |
| Median KL | 0.0013 | 0.0028 |
| Top-1 agreement | 96.1% | 94.1% |

The FP8 head costs +0.0005 mean weight-only KL against the BF16-head build
(paired bootstrap 95% CI `[+0.0003, +0.0006]`), and leaves top-1 agreement
unmoved. For reference, the same recipe on Qwen3.6-27B (same architecture)
measured 0.00924 mean weight-only KL with 97.8% top-1 agreement on the
equivalent harness.

## Serving

```bash
pip install vllm
vllm serve kelnei/Qwen3.8-27B-NVFP4
```

Confirm the optimized kernels loaded: the log should say
`Using CutlassNvFp4LinearKernel for NVFP4 GEMM`.

With multi-token-prediction speculative decoding (the MTP head ships in
this checkpoint):

```bash
vllm serve kelnei/Qwen3.8-27B-NVFP4 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

If you don't have the CUDA toolkit (nvcc) installed, vLLM 0.27's default
FlashInfer backends fail to JIT at startup — the
[serve.py](https://github.com/kelnei/nvfp4-vllm/blob/main/serve.py) wrapper
in the quantization repo steers vLLM to its built-in kernels automatically
(`--linear-backend cutlass`, `--attention-backend TRITON_ATTN`, and the
same for the draft model when using MTP).

### SGLang

SGLang serves this checkpoint, but you must pin the BF16-head revision
[`83ceaf9`](https://huggingface.co/kelnei/Qwen3.8-27B-NVFP4/tree/83ceaf9297ab72a9c11044977cbedfb8937c2ee8):

```bash
pip install sglang
python -m sglang.launch_server \
  --model-path kelnei/Qwen3.8-27B-NVFP4 \
  --revision 83ceaf9297ab72a9c11044977cbedfb8937c2ee8 \
  --reasoning-parser qwen3
```

SGLang applies quantization schemes only to `LinearBase` modules, and
`ParallelLMHead` is not one — it falls back to `UnquantizedEmbeddingMethod`,
which loads `lm_head.weight` and silently discards `lm_head.weight_scale`.
The per-channel scale is gone, so `main` **starts up cleanly and then emits
degenerate output** instead of failing. The only sign is one line in the log:
`Parameter lm_head.weight_scale not found in params_dict`. Verified on
SGLang 0.5.17; vLLM is unaffected.

Pinning costs nothing but the head's throughput gain — `83ceaf9` is
byte-identical to `main` in all 1966 other tensors (same FP8 attention, KV
cache, DeltaNet and GPTQ MLP with layers 56–63 promoted). It differs only in
`lm_head.weight` being BF16 rather than FP8.

`--reasoning-parser qwen3` is needed for the same reason as on vLLM: the
chat template ends in an open `<think>`, so without it the reasoning trace
and a stray `</think>` land in `content` instead of `reasoning_content`.

Two further notes on SGLang: it ignores the checkpoint's static KV scales
(`model.layers.N.k_scale not found in params_dict`) and allocates a BF16 KV
cache instead, and its FlashInfer FP4 GEMM path JIT-compiles at startup, so
nvcc must be on `PATH` — there is no built-in-kernel fallback as there is
with vLLM.
