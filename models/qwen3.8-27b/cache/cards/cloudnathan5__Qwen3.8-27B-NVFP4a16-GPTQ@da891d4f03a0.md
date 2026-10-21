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
  - w4a16
---

# Qwen3.8-27B-NVFP4a16-GPTQ

NVFP4 (W4A16), GPTQ-quantized quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B),
produced with [llm-compressor](https://github.com/vllm-project/llm-compressor) and
saved in the `compressed-tensors` format for direct use with vLLM.

NVFP4 weights chosen by **GPTQ** (Hessian-based error compensation), with activations left at BF16.

Note this checkpoint is the **same size** as the W4A4 variant — the stored weights are identical FP4, and the difference is purely how activations are handled at runtime. So the reason to pick this one is not extra compression: it is lower accuracy risk (activations stay in BF16), and portability, since it does not require the Blackwell FP4 tensor-core GEMM path.

Measured behaviour versus the W4A4 variant: single-stream decode is essentially identical (both are memory-bound at batch 1, and the weights are the same FP4 either way), but **prefill is roughly 2x slower** — W4A16 must dequantize to BF16 for the large compute-bound prefill GEMMs. Batched throughput is also lower. So prefer this variant for non-Blackwell hardware or accuracy caution, not for speed.

**Hardware:** Any GPU vLLM supports; no Blackwell requirement.

**Checkpoint size:** 28.6 GB (BF16 base is 55.6 GB, so 1.95x smaller).


## Usage

```bash
vllm serve cloudnathan5/Qwen3.8-27B-NVFP4a16-GPTQ --max-model-len 32768 --max-num-seqs 512
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

llm = LLM(model="cloudnathan5/Qwen3.8-27B-NVFP4a16-GPTQ")
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
| 1024 | 191 ms | 20.0 ms | **50.1** (**1.91x**) | 26.2 |
| 4096 | 691 ms | 20.1 ms | **49.8** (**1.91x**) | 26.1 |

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
python quantize.py --model-id Qwen/Qwen3.8-27B --method nvfp4a16-gptq
```

## Evaluation — perplexity

Token-level perplexity on `wikitext-2-raw-v1` (test), 48 non-overlapping 4096-token windows (196,560 tokens scored), measured through vLLM with identical settings for both rows.

| model | perplexity | vs BF16 |
|---|---|---|
| this checkpoint | **6.7129** | **+2.37%** |
| Qwen/Qwen3.8-27B (BF16) | 6.5574 | — |

This is token-level perplexity over a fixed window, which is not the same statistic as lm-eval's `word_perplexity` — compare it only against numbers produced the same way.


## Caveats

- Quantization is lossy. Validate on your own workload before production use.
- The exclusion list above was derived from the architecture at release; if you
  fine-tune or otherwise alter module naming, re-derive it.
