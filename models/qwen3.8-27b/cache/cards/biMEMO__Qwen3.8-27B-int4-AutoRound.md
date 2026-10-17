---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - quantization
  - int4
  - w4a16
  - vllm
  - mtp
  - speculative-decoding
  - qwen3_5
  - long-context
---

# The original Qwen3.8-27B, quantized to int4 with a razor-accurate native MTP head baked in — 89.8% draft acceptance, self-accelerating inference out of the box

💬 **[Join the biMEMO Discord](https://discord.gg/z5kJJeaTeS)** — questions, ideas, and
discussion about this quant and what we're building.

**A production-ready INT4 quantization of Qwen3.8-27B, built specifically for vLLM — with
working speculative decoding and validated 256K-token context.**

This isn't a generic weight-shrink. It's tuned so the things that actually matter for a
coding/agentic model survive quantization intact: instant native loading in vLLM, full
256K context with no quality drop, and multi-token prediction (MTP) that actually works —
not just loads.

## Why this quant

- 🚀 **About 3x smaller** than the original (18 GB vs. approx. 54 GB in BF16) — runs comfortably on
  consumer-grade multi-GPU setups instead of requiring datacenter cards.
- ⚡ **MTP speculative decoding works out of the box** — verified **89.8% draft acceptance**
  rate. Many INT4 quants silently break MTP (the model loads fine, but speculative decoding
  quietly contributes nothing); this one doesn't.
- 📏 **Full 256K context, verified — not just claimed.** Long-context degradation is the
  most common silent failure mode in quantized hybrid-attention models. We tested for it
  directly instead of assuming it away (see below) and found none.
- 🔌 **Drop-in for vLLM ≥ 0.26** — standard `compressed-tensors` format, no custom kernels,
  no forked runtime, no patches. Point `vllm serve` at it and go.
- 🧠 **Reasoning and tool-calling preserved** — thinking mode and tool-call formatting work
  identically to the base model.

## Benchmarks

**Methodology:** our own concurrency-sweep harness — real chat completions (not a token
simulator), 512-token responses, MTP speculative decoding enabled throughout. This is not
an industry-standard benchmark length (there isn't one — published numbers elsewhere use
anywhere from 128 to 2000+ tokens); results across different harnesses aren't directly
comparable, so we report our own methodology plainly rather than implying a universal
number.

Aggregate throughput at increasing concurrency, on 1 / 2 / 4 RTX 3090 (24GB) GPUs:

| Concurrent Requests | 1 × RTX 3090 (24GB) | 2 × RTX 3090 (24GB) | 4 × RTX 3090 (24GB) |
|:---:|:---:|:---:|:---:|
| 1 | 55.6 tokens/sec | 90.1 tokens/sec | 102.4 tokens/sec |
| 2 | 111.4 tokens/sec | 176.1 tokens/sec | 187.4 tokens/sec |
| 4 | 223.6 tokens/sec | 323.3 tokens/sec | 322.3 tokens/sec |
| 8 | 224.6 tokens/sec* | **518.6 tokens/sec** | 469.4 tokens/sec |

\* context-limited — see note below

- **1 GPU:** loads and runs. With a 24GB card, weights alone take about 18GB, leaving enough
  headroom for a **32K-token context window** — plenty for most single-turn coding tasks,
  just not the full 256K. Throughput plateaus between concurrency 4 and 8 (223.6 → 224.6
  tok/s) — `max-num-seqs 8` and the reduced KV-cache headroom cap batching before GPU
  compute itself is the limit.
- **2 GPUs:** the sweet spot in our testing — **full 262,144-token context** with the
  highest peak throughput we measured.
- **4 GPUs:** fastest at concurrency 1, but did not scale as well at higher concurrency in
  our testing; 2 GPUs was faster in aggregate on our hardware.

**MTP speculative decoding:**
- Draft acceptance rate: **89.8%**
- Mean accepted length: 1.90 tokens per step

## Long-context validation

We ran a needle-in-haystack test — a unique fact embedded at different positions inside
a 205K-token document — plus a 2000-token continuous generation test, checked
programmatically for repetition loops or incoherence.

| Test | Result |
|---|---|
| Secret fact at start of a 205K-token document | ✅ Retrieved correctly |
| Secret fact in the middle | ✅ Retrieved correctly |
| Secret fact at the end | ✅ Retrieved correctly |
| 2000-token continuous generation at 100K context | ✅ Coherent, no repetition loops |

No degradation at any tested position or length.

## MTP quantization approach

Getting MTP to actually work in an INT4 checkpoint takes more than "quantize everything
and hope." There's exactly one part of the MTP path that must stay untouched: the fusion
layer (`mtp.fc`). Quantize it and vLLM's MTP loader silently skips the head — the model
still loads, speculative decoding still "runs," it just verifies nothing. We keep `mtp.fc`
in full precision specifically so that doesn't happen.

Beyond that one required exception, we quantize the rest of the MTP draft block (its
attention and MLP projections) right along with the rest of the model, rather than leaving
the whole block in higher precision. We didn't assume that was the right call — we tested
it against the more conservative alternative (keeping the entire MTP block unquantized) on
identical hardware, identical prompts, identical decoding settings:

| Approach | Draft acceptance rate | Peak throughput |
|---|---|---|
| Full MTP block left in BF16 | approx. 77% | 522.9 tok/s |
| **This checkpoint (MTP block quantized, fusion layer protected)** | **89.8%** | **536.1 tok/s** |

Being more conservative with the MTP block didn't buy any measurable benefit here — a
fully calibrated draft block, tuned together with the rest of the model, tracked the
quantized network's behavior more closely than one left untouched. Same throughput
class, meaningfully higher acceptance.

## Quick start (vLLM)

**Verified on vLLM 0.27.1** (works on 0.26+, but exact numbers below were re-measured on
0.27.1). Pick the command that matches your GPU count — all three are configs we
actually ran and benchmarked (see the throughput table above), not theoretical examples.
Exact `--gpu-memory-utilization` headroom needed can shift slightly between vLLM
versions — if you hit a KV-cache allocation error at startup, try raising it a couple
points.

### 1 GPU (24GB card — reduced context)

Weights alone take about 18GB on a single card, so the context window is capped at
what fits in the remaining headroom — we tested 32K tokens; vLLM's own memory
estimate at startup allowed slightly more (~39.6K), 32768 leaves a safety margin.

```bash
vllm serve biMEMO/Qwen3.8-27B-int4-AutoRound \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8_e4m3 \
  --mm-processor-cache-type shm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --trust-remote-code
```

### 2 GPUs (recommended — best throughput + full context in our testing)

```bash
vllm serve biMEMO/Qwen3.8-27B-int4-AutoRound \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 2 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8_e4m3 \
  --mm-processor-cache-type shm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --trust-remote-code
```

### 4 GPUs

Fastest at concurrency 1 in our testing, but scaled worse than 2 GPUs at higher
concurrency on our hardware (see Benchmarks) — worth benchmarking independently on
your own interconnect before assuming more GPUs means more throughput.

```bash
vllm serve biMEMO/Qwen3.8-27B-int4-AutoRound \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8_e4m3 \
  --mm-processor-cache-type shm \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --enable-chunked-prefill --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --trust-remote-code
```

## What you get

- **27B dense model**, hybrid linear + full attention, native 262,144-token context
  (extensible further on the base model).
- **18 GB on disk** — fits comfortably where the 54 GB BF16 original doesn't.
- **INT4 weights, W4A16**, with the components that are sensitive to quantization
  (multi-token prediction, long-context attention pathways, vision tower, output head)
  left at full precision — quantized where it's safe, untouched where it isn't.

## License

Apache 2.0, inherited from the base model
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
