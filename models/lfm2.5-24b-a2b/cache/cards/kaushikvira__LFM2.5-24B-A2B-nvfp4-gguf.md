---
license: other
base_model: LiquidAI/LFM2.5-24B-A2B
tags:
- gguf
- nvfp4
- moe
- reasoning
- llama.cpp
- blackwell
- quantization
- 24b
pipeline_tag: text-generation
library_name: gguf
---
# LFM2.5-24B-A2B NVFP4 (all-4-bit) GGUF for llama.cpp

An **all-NVFP4** (4.55 BPW) GGUF of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) — Liquid AI's 24B-total / 2.45B-active reasoning MoE (30 gated short-conv blocks + 10 GQA layers, 64 experts top-4, 131K context, 128K vocab) — quantized for **llama.cpp on Blackwell** (RTX 5090 / RTX PRO 5000). Experts **and** dense weights are NVFP4 (Blackwell's native FP4 tensor-core format); only the small quality-sensitive tensors (norms, router biases, token embeddings) stay Q8_0.

**Why NVFP4:** on Blackwell, prefill runs through native FP4 MMA instructions — this build measures **1.6× faster prefill** than Liquid's official Q4_K_M at a *smaller* file size, with decode within 4% and no measurable quality loss on our gate.

## Credits & provenance

All credit for the underlying model belongs to **Liquid AI**. This artifact is a format conversion + quantization only — **no fine-tuning, no weight modification**.

| Role | Source | Notes |
|---|---|---|
| Source weights (BF16) | [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) | Liquid AI, LFM Open License v1.0 (`lfm1.0`); 10-shard BF16 safetensors, sha-verified on download |
| Quantization toolchain | [llama.cpp](https://github.com/ggml-org/llama.cpp) PR [#26869](https://github.com/ggml-org/llama.cpp/pull/26869) head `c100a98` (qikpl4/llama.cpp @ xxfp4) | "quantization: complete MXFP4 and NVFP4" — adds `NVFP4`/`NVFP4_MOE` types + quant routines; CUDA 13.1 build |
| Comparison baseline | [LiquidAI/LFM2.5-24B-A2B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) `Q4_K_M` | Liquid's official 4-bit GGUF (13.52 GiB) |

## What we did

1. **BF16 → f16 GGUF** via `convert_hf_to_gguf.py` (48.0 GiB intermediate, 428 tensors).
2. **Quantized to NVFP4** with the PR-branch `llama-quantize`: base type `NVFP4` for all weight families (experts, shortconv in/out-proj, attention, dense FFN, router), with `Q8_0` overrides for `*_norm.weight`, `exp_probs_b.bias`, and `token_embd.weight` (128K-vocab embedding kept high-precision). Result: **12.70 GiB, 4.55 BPW**.

## Measured results (RTX PRO 5000 72 GB @ 270 W cap, driver 615.71.09, CUDA 13.1, llama.cpp `c100a98`)

### llama-bench (single stream, `-p 1024 -n 256`, mean of 2)

| quant | size | pp1024 (t/s) | tg256 (t/s) |
|---|---|---|---|
| **NVFP4 (this repo)** | 12.70 GiB | **15,789 ± 2,035** | 395.8 ± 1.1 |
| Q4_K_M (Liquid official) | 13.52 GiB | 9,815 ± 725 | **413.3 ± 1.6** |

NVFP4 wins prefill **1.61×** (native FP4 tensor cores); Q4_K_M wins pure decode by ~4% (mature dequant path — the NVFP4 GEMV/MoE-scheduling optimizations are still open upstream).

### Serving profile (llama-server, 131K ctx/slot, 4 slots, q4_0 KV cache)

| test | result |
|---|---|
| Prefill 8k / 32k / 100k | 26.3k / 31.7k / 28.2k t/s (TTFT 0.31 / 1.03 / 3.55 s) |
| Decode (1k ctx, 256 out, incl. thinking) | 327 t/s |
| Concurrency-4 (4 × 8k-in/128-out) | 195 aggregate out-t/s, ~49 t/s per stream, TTFT 0.95–1.76 s |

### Quality gate (identical probes on both quants, temp 0.1 / top_k 50 / rep 1.05)

| probe | NVFP4 | Q4_K_M |
|---|---|---|
| Needle @ 8k (ZEBRA-42) | ✅ | ✅ |
| Needle @ 32k (ZEBRA-42) | ✅ exact | ✅ exact |
| Instruction-following (exactly-3-words) | ✅ | ✅ |
| Math (17×24) | ✅ 408 | ✅ 408 |
| Tool call (Pythonic `<\|tool_call_start\|>` format) | ✅ | ✅ |

No quality difference detectable on this probe set. (Small probe set — a full needle/IF ladder is recommended before high-stakes use.)

## Artifact

| Field | Value |
|---|---|
| Filename | `lfm25-nvfp4-full.gguf` |
| Size | 13,641,678,944 bytes (12.70 GiB, 4.55 BPW) |
| **SHA-256** | `ec37554c76f34784699a41754dd4538d780f560c9d64a136441399d4be7df6ec` |
| Tensor layout | experts + dense = NVFP4 (group-16, FP8 scales); norms/router-bias/embedding = Q8_0 |
| Context | 131,072 (model max) |

## Serving (Blackwell, single GPU)

```bash
# requires the PR-branch build (NVFP4 GGUF support); see "Build note" below
llama-server -m lfm25-nvfp4-full.gguf \
  -ngl 99 -c 524288 --parallel 4 -a lfm25 \
  --temp 0.1 --top-k 50 --repeat-penalty 1.05 \
  --jinja --metrics -ctk q4_0 -ctv q4_0 --port 8085
```

> **Build note:** NVFP4 GGUF support ships in llama.cpp PR [#26869](https://github.com/ggml-org/llama.cpp/pull/26869) (open at time of publishing). This artifact was quantized with and verified on head `c100a98`; **serve it with that build (or any later release containing the merged PR)** — release llama.cpp without the PR cannot load NVFP4 tensors. If the PR's type layout changes on merge, re-quantize from the f16 intermediate.
>
> `-c` is the **total** context across `--parallel` slots (524288 = 4 × 131072). q4_0 KV halves the KV pool (~6 GiB for 4×131K) with no measurable recall loss at 32k in our gate.
>
> **Sampling** (per the model card): `temperature 0.1, top_k 50, repetition_penalty 1.05`; leave `max_tokens` unset — capping it truncates the chain-of-thought.

## Comparison with Q4_K_M (when to pick which)

| | **NVFP4 (this repo)** | [Q4_K_M (Liquid)](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF) |
|---|---|---|
| Size | 12.70 GiB | 13.52 GiB |
| Prefill | **1.6× faster** (FP4 tensor cores) | baseline |
| Decode | within 4% | ~4% faster |
| Engine | PR-branch llama.cpp only | any release llama.cpp |
| Pick it when | long-context / TTFT-sensitive serving on Blackwell | maximum decode throughput, zero build risk |

## License terms carried by this artifact

* **LFM Open License v1.0** (Liquid AI) — full copy in `LICENSE`. The weights in this repository are **modified** relative to the upstream work (NVFP4/Q8_0 requantization only; no architectural or post-training changes). All use is subject to the LFM Open License v1.0 terms, including its commercial-use provisions.
* Attribution: [Liquid AI — LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) (blog: [lfm2-5-24b-a2b](https://www.liquid.ai/blog/lfm2-5-24b-a2b)).

### Reporting

If you become aware of misuse of this model, report it to the platform where the misuse occurs and to the relevant authorities. The publisher of this artifact (`kaushikvira`) does not monitor, endorse, or take part in downstream use.
