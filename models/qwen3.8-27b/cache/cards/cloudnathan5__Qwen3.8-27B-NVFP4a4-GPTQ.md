---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - compressed-tensors
  - fp4
  - gptq
  - nvfp4
  - quantized
  - vllm
  - w4a4
---

# Qwen3.8-27B-NVFP4a4-GPTQ

NVFP4 (W4A4), GPTQ-quantized quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) and
saved in the `compressed-tensors` format for direct use with vLLM.

Weights **and** activations in NVFP4, NVIDIA's 4-bit floating-point format with a 16-element block scale. Unlike a round-to-nearest NVFP4 checkpoint, the weights here are chosen by **GPTQ**, which uses second-order (Hessian) information from calibration data to compensate for quantization error layer by layer. That costs nothing at inference time — the result is still a plain NVFP4 checkpoint. See the measured perplexity below rather than assuming a recovery benefit.

This runs on Blackwell's native FP4 tensor cores. In practice the gain shows up most in **prefill** (compute-bound, so 4-bit activations help — TTFT is roughly half that of the W4A16 variant) and in **batched throughput**. Single-stream decode is memory-bound and therefore close to W4A16, since both store the same FP4 weights.

**Hardware:** NVIDIA Blackwell (SM100 / SM120) for the accelerated path. On older GPUs vLLM falls back to a dequantize-and-emulate path that is functional but slower than BF16.

**Checkpoint size:** 28.6 GB (BF16 base is 55.6 GB, so 1.95x smaller).


## Usage

```bash
vllm serve cloudnathan5/Qwen3.8-27B-NVFP4a4-GPTQ --max-model-len 32768 --max-num-seqs 512
```

> **`--max-num-seqs` matters on this architecture.** 48 of the 64 layers use
> linear attention, and vLLM allocates one Mamba-style cache block per decode
> sequence. vLLM's default `max_num_seqs=1024` can exceed the number of blocks
> that fit, and startup then fails during CUDA graph capture with
> `max_num_seqs (1024) exceeds available Mamba cache blocks`. Lower
> `--max-num-seqs` (512 is a safe starting point) or raise
> `--gpu-memory-utilization`. This is a property of the base model, not of
> quantization.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="cloudnathan5/Qwen3.8-27B-NVFP4a4-GPTQ")
out = llm.generate(
    ["Explain 4-bit quantization in two sentences."],
    SamplingParams(temperature=0.7, max_tokens=256),
)
print(out[0].outputs[0].text)
```

## Single-user latency

Concurrency **1** (one request in flight — no batching), on a single NVIDIA RTX PRO 6000 Blackwell 96GB, vLLM 0.27.1. Prefix caching disabled and `--ignore-eos` set so prefill is never skipped and every generation is exactly 256 tokens. Median over 24 requests after 4 warmups, via `vllm bench serve --max-concurrency 1`.

| input tokens | TTFT | inter-token latency | decode tok/s | BF16 base tok/s |
|---|---|---|---|---|
| 1024 | 105 ms | 20.4 ms | **49.1** (**1.87x**) | 26.2 |
| 4096 | 343 ms | 20.5 ms | **48.8** (**1.87x**) | 26.1 |

This is single-stream interactive performance, not batched throughput; under concurrency the ranking between variants differs.

## Quantization details

Calibrated on 256 samples of
[HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k)
at 4096 tokens, chat template applied.

The following modules are left in their original precision:

| pattern | reason |
|---|---|
| `lm_head`, `embed_tokens` | quantizing these costs accuracy for no speed benefit |
| `visual.*` | the vision tower is small and quantization-sensitive |
| `linear_attn.*` | the gated-delta / linear-attention state paths are numerically fragile at 4 bits and are not GEMM-bound |
| `mlp.gate`, `shared_expert_gate` | MoE routing weights |
| `mtp.*` | multi-token-prediction head — **included in this checkpoint at original BF16 precision**, so MTP speculative decoding works |

Reproduce with [`quantize.py`](./quantize.py):

```bash
python quantize.py --model-id Qwen/Qwen3.8-27B --method nvfp4-gptq
```

## Evaluation — perplexity

Token-level perplexity on `wikitext-2-raw-v1` (test), 48 non-overlapping 4096-token windows (196,560 tokens scored), measured through vLLM with identical settings for both rows.

| model | perplexity | vs BF16 |
|---|---|---|
| this checkpoint | **6.7733** | **+3.29%** |
| Qwen/Qwen3.8-27B (BF16) | 6.5574 | — |

This is token-level perplexity over a fixed window, which is not the same statistic as lm-eval's `word_perplexity` — compare it only against numbers produced the same way.


## Caveats

- Quantization is lossy. Validate on your own workload before production use.
- The exclusion list above was derived from the architecture at release; if you
  fine-tune or otherwise alter module naming, re-derive it.
