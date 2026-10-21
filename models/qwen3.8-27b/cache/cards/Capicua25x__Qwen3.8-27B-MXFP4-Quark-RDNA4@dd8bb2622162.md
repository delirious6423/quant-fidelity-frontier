---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - 4-bit
  - mxfp4
  - quark
  - amd
  - rocm
  - rdna4
  - gfx1201
  - vllm
  - quantized
---


# Qwen3.8-27B — MXFP4 (AMD Quark) for RDNA4

MXFP4 weight quantisation of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), built with
[AMD Quark](https://quark.docs.amd.com) 0.12.post1 for **RDNA4** (gfx1200/gfx1201: Radeon AI PRO R9700,
RX 9070 XT). > **⚡ 2026-08-20 — major performance upgrade (rc10):** the serving port gained tuned per-shape GEMM
> configs for the R9700, regenerated with vLLM's own tuner on the exact serving stack. This model's
> throughput rose **+12–28% per cell** (short c32 579→713 aggregate); the FP8 arm rose up to **+34%**
> (c32 754→1,014). All tables below are current rc10 measurements — pull `latest` and you get them.
> Credit where due: this port was always geared to **concurrent serving** (32 sequences, speculative
> decoding, accuracy-gated) — the single-stream tuning insight came from the community.
> [andysalerno](https://github.com/andysalerno/R9700-serving)'s and
> [prcoe1](https://github.com/prcoe1/r9700-serving)'s benchmarks surfaced the untuned-GEMM gap; folding
> that lever into the concurrency stack is what closed the loop.

> **Measurement update (2026-08-24, bench v4):** the essay workload behind the throughput numbers above
> used one fixed prompt at temperature 0, and on a speculative-decoding server the stateful drafter
> partially replays previously-generated text — inflating acceptance and tok/s with the server's own
> content history. The **relative** rc-ladder gains quoted above (579→713, 754→1,014) were measured
> like-for-like and stand as relative claims; the **absolute** essay tok/s figures are replay-inflated
> (shape-dependent on re-measurement: ≈10-20% on most greedy-raw cells, up to ≈2x on the worst (a GB10-pair short cell), while one sampled-path 6k cell even measured slightly higher under v4 (different sampling provenance + build drift)). Bench v4 (rotating distinct topics, per-invocation nonce, temp 0.7 —
> nothing is ever regenerated) gives the honest absolutes for the shapes this card quotes, measured on
> the FP8 serving arm (B): short-prompt c32 **789 aggregate / 27.0 per-user** (was 1,014), 6k-prefix
> c32 **≈390 aggregate**, comfort ceiling (≥20 tok/s per user) **≈16 concurrent users at 6k**. Full v4
> table below. Harness: [Capicua25x/modelbench](https://github.com/Capicua25x/modelbench) (v4).

Serve it with our vLLM 0.26.1 RDNA4 port: **[GitHub](https://github.com/Capicua25x/vllm-rocm-rdna4)** ·
**[Docker image](https://hub.docker.com/r/capicua25x/vllm-rocm-rdna4)** (`capicua25x/vllm-rocm-rdna4:latest`, rc10) ·
benchmark harness: **[Capicua25x/modelbench](https://github.com/Capicua25x/modelbench)**.

**What's 4-bit:** MLP/expert projections only (12.05B packed U8 = 22.7B params). Attention (q/k/v/o + norms),
embeddings, `lm_head`, routers and the entire vision path stay bf16 — ≈27.8B logical params, architecture
unchanged. (HF's sidebar "8-bit"/"16B" auto-tags read the U8 container, not the contents; every MXFP4 repo
on the Hub gets them.) Calibrated fp8 KV-cache scales ship as a side-file; they only activate under
`--kv-cache-dtype fp8`.

## Quick start (2× R9700, TP2, full native 262k window, MTP-3)

```bash
docker run --rm --name vllm-qwen --network=host \
  --device=/dev/kfd --device=/dev/dri/renderD128 --device=/dev/dri/renderD129 \
  --group-add=video --group-add=render --ipc=host \
  -e NCCL_PROTO=Simple \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint /usr/local/bin/vllm capicua25x/vllm-rocm-rdna4:0.26.1-rdna4-rc10 \
  serve Capicua25x/Qwen3.8-27B-MXFP4-Quark-RDNA4 --served-model-name qwen --port 8011 --trust-remote-code \
  --tensor-parallel-size 2 --gpu-memory-utilization 0.95 --max-model-len 262144 \
  --attention-backend TRITON_ATTN --enable-prefix-caching \
  --max-num-seqs 32 --max-num-batched-tokens 8000 --max-cudagraph-capture-size 128 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN"}'
```

We A/B this configuration (**C · MXFP4 @ bf16 KV**) in production against **B · FP8 @ fp8 KV**
([`Qwen/Qwen3.8-27B-FP8`](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) + `--kv-cache-dtype fp8
--mamba-ssm-cache-dtype bfloat16`). Both pass the same gates; so far Qwen behaves correctly on both.

## Thinking mode — read this before benchmarking

Qwen3.8's reasoning ("thinking") is controlled by a **chat-template kwarg**, not by the OpenAI
`reasoning_effort` parameter. vLLM **silently ignores** `reasoning_effort` — requests that pass it
run in no-think mode with no error, which measurably degrades agentic/multi-step performance
(we measured τ²-Bench telecom dropping from ≈0.90 to ≈0.63 on both this model and the FP8 arm
before catching it). To enable thinking, send:

```json
{"chat_template_kwargs": {"enable_thinking": true}}
```

at the top level of the request body (or inside `extra_body` when using an OpenAI client / LiteLLM).
With `--reasoning-parser qwen3` (as in the quick start), the think block is stripped into
`reasoning_content` and the visible `content` stays clean — tool calls are unaffected. A cheap
preflight to verify thinking is active: send a short question with `max_tokens: 60`; a thinking
serve spends most of the budget invisibly (tiny visible `content` vs `completion_tokens`), a
no-think serve returns ≈60 tokens of visible prose. All accuracy numbers above were measured
with thinking ON except the "nothink" rows.

## Measured performance (per-user / aggregate tok/s)

Paired back-to-back on the same box (2× R9700, TP2, thinking ON, 256-token completions).
**C** = this model @ bf16 KV · **B** = FP8 checkpoint @ fp8 KV.

> **Note (2026-08-24):** the two C-vs-B tables below are **pre-v4, replay-era** measurements — valid
> as **comparatives** (both arms ran the identical replay-prone workload back-to-back), but the
> absolute essay tok/s values are inflated by the fixed-prompt/temp-0 replay confound (shape-dependent on re-measurement: ≈10-20% on most greedy-raw cells, up to ≈2x on the worst (a GB10-pair short cell), while one sampled-path 6k cell even measured slightly higher under v4 (different sampling provenance + build drift))
> described in the measurement update above. Honest v4 absolutes follow in the third table.

**Short prompts (≈30 tokens — interactive chat) — pre-v4, replay-era: comparative only (see note)**

| users | C (this model) | B |
|---|---|---|
| 1 | 62 / 62 | 66 / 66 |
| 4 | 53 / 208 | 60 / 232 |
| 8 | 42 / 335 | 54 / 421 |
| 16 | 32 / 505 | 45 / 687 |
| 32 | 24 / 713 | 33 / **1,014** |
| 64 | 17 / 714 | 24 / **1,016** |

*B column: in-tree tuned R9700 GEMM configs; C column: measured on rc10 (both 2026-08-20).*

**6,000-token prompts (RAG / long-system-prompt workloads) — pre-v4, replay-era: comparative only (see note)**

| users | C (this model) | B |
|---|---|---|
| 1 | 54 / 54 | 61 / 61 |
| 4 | 43 / 163 | 43 / 170 |
| 8 | 30 / 231 | 31 / 241 |
| 16 | 20 / 312 | 20 / 311 |
| 32 | 12 / 368 | 11 / 356 |

At long context the two are a statistical tie; on short prompts B leads from 8 users up. KV pool:
C ≈ 415k tokens · B ≈ 539k (2.06× the window). `NCCL_PROTO=Simple` matters on this PCIe pair
(RCCL's LL protocol is 2.8× slower for the ≈640 KB decode all-reduces).

**Honest v4 absolutes (measured 2026-08-24, idle-verified) — FP8 arm B, rc10, MTP-3 (k=3)**

Bench v4: rotating distinct topics + per-invocation nonce, temp 0.7 / top_p 0.95 (closer to
production sampling than the old greedy fixed prompt — nothing is ever regenerated). Per-user /
aggregate tok/s; acceptance is accepted-per-draft out of k=3, scraped from `/metrics`.

| workload | c1 | c4 | c16 | c32 | c64 | acceptance (of 3) |
|---|---|---|---|---|---|---|
| short essay | 59.1 / 59 | 49.5 / 183 | 37.1 / 541 | 27.0 / 789 | 20.1 / 825 | 1.52 @ c1 |
| 6k-prefix essay | 55.4 / 55 | 44.1 / 168 | 21.8 / 327 | 12.8 / 390 | 8.4 / 397 | 1.60 @ c1 · 1.72–1.84 multi-user |
| trivial (count-to-300) | 98.0 / 98 | — | 64.5 / 1,030 | 47.7 / 1,523 | 35.7 / 1,521 | 2.99 flat |

6k aggregate saturates at ≈390–400 from c=32 (c48: 10.0 / 373). Comfort ceiling at ≥20 tok/s per
user: **≈16 in-flight at 6k**, ≈64 on short prompts. Novel prose lands ≈2.4–2.8 tokens/step
regardless of draft length. Pre-v4 C-vs-B tables above remain relative-shape-only.

**Honest v4 absolutes — MXFP4 arm C, THIS MODEL (measured 2026-08-24 late, idle-verified; rc10
image, reference C config: TP2, bf16 KV, MTP-3, --max-num-seqs 32)**

| workload | c1 | c4 | c16 | c32 | c64 | acceptance (of 3) |
|---|---|---|---|---|---|---|
| short essay | 43.4 / 43 | 43.7 / 166 | 30.2 / 448 | 20.5 / 602 | 13.8 / 576 | 1.14–1.40 |
| 6k-prefix essay | 54.1 / 54 | 40.9 / 158 | 21.5 / 325 | 12.6 / 387 | 8.7 / 395 | 1.64–1.90 |
| trivial (count-to-300) | 64.5 / 65 | 74.8 / 299 | 52.6 / 741 | 36.1 / 1,042 | 27.1 / 1,039 | 2.99–3.00 flat |

**The headline: at realistic ≈6k context, C matches the FP8 arm within noise at every level**
(c16 325 vs 327, c32 387 vs 390, c64 395 vs 397 aggregate — same ≈16-user comfort ceiling).
The FP8 arm's advantage lives in compute-bound shapes: trivial peak 1,523 vs 1,042 (≈1.46×)
and short-prompt (≈1.3–1.4×), with equal speculative acceptance — i.e., MXFP4 dequant cost,
not drafting. What C buys instead: the full 262k native window (FP8+MTP tops out earlier).
Pick FP8 for short-query burst capacity; pick this model for max context on the same silicon —
at real context sizes you give up nothing measurable.

## Accuracy (AA class-A, paired items, seed 1234, on-spec sampling)

**ref** = the same checkpoint served in bf16 by a cloud provider. Same judge for all judged rows.
Cells show the most recent run at the stated n on the shipping config; ±2 items is the noise band.

| benchmark (n) | ref (bf16) | C (this model) | B |
|---|---|---|---|
| GSM8K think, flex·strict (50) | 0.96·0.82 | 0.98·0.96 | 0.98·0.86 |
| GSM8K nothink (50) | 0.98·0.98 | 0.98·0.98 | 0.98·0.98 |
| IFEval inst·prompt (80) | 0.97·0.95 | 0.95·0.91 | 0.98·0.98 |
| AA-LCR judged (100) | 0.78 | 0.78 | 0.77 · 0.81 (s1234·s99) |
| GPQA-Diamond (60) | 0.78 | 0.92 | 0.85 |
| AIME'25 (30) | 0.93 | 0.93 | 0.97 |
| τ²-telecom agentic (114) | 0.939 | 0.904 | 0.94 |
| τ²-airline agentic (50) | 0.760 | 0.840 | 0.86 |
| τ²-retail agentic (60) | 0.850 | 0.767 | 0.82 |
| MMLU-Pro (1120) | 0.804 | — | 0.817 |
| Terminal-Bench hard (44, 3600s/task) | 0.273 | — | 0.341 |
| HLE, judged (120) | 0.30 | 0.25 | 0.275 |


## Reproducing the quantisation

Data-free, CPU-only, file-to-file — ≈3 minutes; no calibration set. (The KV-cache scalars are the one
calibrated artefact and came from a separate capture pass on the served model.)

```python
from quark.torch.export.api import direct_quantize_checkpoint

EXCLUDE = [
    "lm_head", "*embed_tokens*",
    "*.self_attn.q_proj", "*.self_attn.k_proj", "*.self_attn.v_proj", "*.self_attn.o_proj",
    "*.self_attn.q_norm", "*.self_attn.k_norm", "*norm*",
    "*.linear_attn.conv1d", "*.linear_attn.norm",
    "*.mlp.gate", "*.mlp.shared_expert_gate",
    "mtp*", "*visual*", "*vision*",
]
```

Full recipe, engine patches and methodology: [`RDNA4-PORT.md`](https://github.com/Capicua25x/vllm-rocm-rdna4/blob/rdna4-port-0.26.1/RDNA4-PORT.md) · throughput numbers are reproducible with [Capicua25x/modelbench](https://github.com/Capicua25x/modelbench) (bench v4; the pre-v4 replay-era essay absolutes are not reproducible by design — see the 2026-08-24 measurement update).

## Licence and attribution

Apache-2.0, following the base model. Quantised and served by [Capicua25x](https://github.com/Capicua25x);
base model by the Qwen team; quantisation toolkit by AMD (Quark).
