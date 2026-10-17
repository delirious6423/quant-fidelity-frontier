---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - compressed-tensors
  - fp8
  - gptq
  - mixed-precision
  - nvfp4
  - quantized
  - vllm
  - w4a4
---

# Qwen3.8-27B-NVFP4a4-FP8-GPTQ

NVFP4 + FP8 mixed precision, GPTQ-quantized quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) and
saved in the `compressed-tensors` format for direct use with vLLM.

A two-precision checkpoint chosen for this architecture specifically. The GEMM-bound paths — MLP and the full-attention projections — are in **NVFP4 (W4A4)** and run on Blackwell's FP4 tensor cores. The **linear-attention** paths are in **FP8 (W8A8)** instead: 48 of this model's 64 layers are linear-attention, so leaving them at BF16 would cap compression near 2x, while pushing their gated-delta state to 4 bits is numerically risky. FP8 keeps them safe and still halves them.

Weights are chosen by **GPTQ** (Hessian-based error compensation), which costs nothing at inference time.

**Hardware:** NVIDIA Blackwell (SM100 / SM120) for the FP4 path; FP8 requires SM89+. On older GPUs vLLM falls back to emulation.

**Checkpoint size:** 23.0 GB (BF16 base is 55.6 GB, so 2.42x smaller).


## Usage

```bash
vllm serve cloudnathan5/Qwen3.8-27B-NVFP4a4-FP8-GPTQ --max-model-len 32768 --max-num-seqs 512
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

llm = LLM(model="cloudnathan5/Qwen3.8-27B-NVFP4a4-FP8-GPTQ")
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
| 1024 | 94 ms | 17.3 ms | **57.9** (**2.21x**) | 26.2 |
| 4096 | 302 ms | 17.3 ms | **57.7** (**2.21x**) | 26.1 |

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
python quantize.py --model-id Qwen/Qwen3.8-27B --method nvfp4-fp8-gptq
```

## Evaluation — perplexity

Token-level perplexity on `wikitext-2-raw-v1` (test), 48 non-overlapping 4096-token windows (196,560 tokens scored), measured through vLLM with identical settings for both rows.

| model | perplexity | vs BF16 |
|---|---|---|
| this checkpoint | **6.8614** | **+4.64%** |
| Qwen/Qwen3.8-27B (BF16) | 6.5574 | — |

This is token-level perplexity over a fixed window, which is not the same statistic as lm-eval's `word_perplexity` — compare it only against numbers produced the same way.


## Caveats

- Quantization is lossy. Validate on your own workload before production use.
- The exclusion list above was derived from the architecture at release; if you
  fine-tune or otherwise alter module naming, re-derive it.
