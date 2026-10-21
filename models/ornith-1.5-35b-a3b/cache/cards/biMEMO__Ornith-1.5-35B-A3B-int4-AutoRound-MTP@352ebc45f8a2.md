---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
tags:
  - quantization
  - int4
  - w4a16
  - auto-round
  - vllm
  - mtp
  - speculative-decoding
  - moe
  - expert-parallel
  - qwen3_5_moe
  - long-context
---

# Ornith-1.5-35B-A3B quantized to INT4 (AutoRound W4A16) with a working native MTP head — runs the full MoE on consumer 24 GB cards

💬 **[Join the biMEMO Discord](https://discord.gg/z5kJJeaTeS)** — questions, ideas, and discussion about this quant and what we're building.

**A production-ready INT4 quantization of the `ornith-ai/Ornith-1.5-35B-A3B` mixture-of-experts model, built and verified for vLLM — with multi-token prediction (MTP) that actually works and the full 262K context intact.**

This isn't a generic weight-shrink. The 35B/≈3B-active MoE was quantized with **AutoRound** on a code/reasoning corpus, and — critically — the MTP fusion layer was kept in BF16 so speculative decoding **loads AND contributes** instead of silently doing nothing. We verified output quality at the raw token level, not just throughput.

> ⚡ **Who this is for — speed over depth.** With only ~3B parameters active per token, this is a **latency/throughput-first** model: it answers fast and holds long context cheaply, but it does **not** match a dense model of comparable total size on breadth-of-knowledge and hard reasoning. Pick it when responsiveness and concurrency matter more than maximum world knowledge — coding assistants, agentic tool-loops, high-QPS serving.

## Why this quant

- 🚀 **~3.5× smaller** — about **20 GB** vs. ~70 GB in BF16 — the whole 35B MoE fits on **2–4× RTX 3090 (24 GB)** instead of datacenter cards.
- ⚡ **MTP speculative decoding works** — on code workloads we measure **~85–92% draft acceptance** and **~2.6 accepted tokens per step** (1-layer MTP head, `num_speculative_tokens` 1–3). Many INT4 MoE quants load MTP but verify nothing; this one doesn't.
- 🧩 **Runs the MoE across 4 GPUs via expert parallelism.** Plain tensor-parallel ≥ 4 on W4A16 MoE currently mis-shards in vLLM ([#41511](https://github.com/vllm-project/vllm/issues/41511)); use `--enable-expert-parallel` (details below). `--tensor-parallel-size 2` also works out of the box.
- 📏 **Full 262,144-token context** — fits comfortably (KV budget ≈ 1M tokens on 2× 24 GB with fp8 KV, ~4 full-context requests in parallel).
- 🔌 **Standard `compressed-tensors` / auto_round format for vLLM ≥ 0.26** — no forked runtime, no patches.
- 🧠 **Reasoning, tool-calling and vision preserved** — thinking mode, `qwen3_xml` tool calls, and the vision tower are intact.

## Benchmarks

**Methodology:** our streaming prefill+generation harness — real chat completions, a fixed **~8,200-token prompt**, 512-token generations, `temperature=0`, thinking off, on **4× RTX 3090** (PCIe Gen4 x4, no NVLink). We report per-request decode speed, server-aggregate throughput, and time-to-first-token (TTFT) at increasing concurrency. Numbers are from our hardware; different harnesses aren't directly comparable.

**TP=2 (2 GPUs) + MTP (spec 3), 8K prompt — single-stream is fastest here (less inter-GPU traffic):**

| Concurrent | Decode / req (tok/s) | Aggregate (tok/s) | TTFT (s) |
|:--:|:--:|:--:|:--:|
| 1 | 195.0 | 96.2 | 2.69 |
| 8 | 74.4 | 366.8 | 3.84 |
| 16 | 33.7 | 300.4 | 5.85 |
| 32 | 19.1 | 343.0 | 10.2 |

On 2× RTX 3090 the useful concurrency ceiling is ~32 (aggregate ~360 tok/s); single-user decode peaks near ~195 tok/s.

**TP=4 + expert-parallel + MTP (spec 3), 8K prompt — higher aggregate under heavy parallel load:**

| Concurrent | Decode / req (tok/s) | Aggregate (tok/s) | TTFT (s) |
|:--:|:--:|:--:|:--:|
| 1 | 125.9 | 62.2 | 4.16 |
| 2 | 114.9 | 143.8 | 2.47 |
| 4 | 112.1 | 299.3 | 2.17 |
| 8 | 76.9 | 416.5 | 2.81 |

## MTP quantization approach

Getting MTP to work in an INT4 MoE checkpoint takes one required exception: the fusion layer (`mtp.fc`) is kept in **BF16**. Quantize it and vLLM's MTP loader chokes (or silently skips the head). The rest of the MTP draft block (its 256 MoE experts) is quantized along with the model. `lm_head`, embeddings, norms, router gates and the vision tower are kept in higher precision (standard AutoRound behavior).

## Quick start (vLLM ≥ 0.26)

**This is a MoE — for tensor-parallel ≥ 4 you MUST add `--enable-expert-parallel`,** otherwise the W4A16 MoE mis-shards and produces garbage (vLLM [#41511](https://github.com/vllm-project/vllm/issues/41511)).

### 2 GPUs (recommended — fastest single-stream, full 262K context, frees 2 cards)

```bash
vllm serve biMEMO/Ornith-1.5-35B-A3B-int4-AutoRound-MTP \
  --served-model-name Ornith-1.5-35B-A3B-int4-AutoRound-MTP \
  --tensor-parallel-size 2 --disable-custom-all-reduce \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8_e4m3 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --trust-remote-code
```

### 4 GPUs (all cards, expert-parallel — higher aggregate throughput)

```bash
vllm serve biMEMO/Ornith-1.5-35B-A3B-int4-AutoRound-MTP \
  --served-model-name Ornith-1.5-35B-A3B-int4-AutoRound-MTP \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len 200704 \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --trust-remote-code
```

Notes:
- On GPUs without working P2P (e.g. PCIe-only pairs) add `--disable-custom-all-reduce` to avoid a custom all-reduce CUDA error at init.
- This model uses hybrid linear (Mamba-style) + full attention; if you hit `Mamba cache align: block_size must be <= max_num_batched_tokens`, raise `--max-num-batched-tokens` (e.g. 4096).
- To turn thinking off per request, pass `chat_template_kwargs={"enable_thinking": false}`.

## Compatibility notes

- **vLLM:** verified on v0.26.0 (TP2 and TP4+expert-parallel).

## License

Inherits the base model's **MIT** license (`ornith-ai/Ornith-1.5-35B-A3B`). Quantized by MinerNinja.
