---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
- quantized
- compressed-tensors
- moe
- int8
- bf16
- vllm
- qwen
- ampere
---

# Qwen3.8-Flash-Next — INT8x25 experts + BF16 attention

**A hybrid-precision checkpoint of Qwen3.8-Flash-Next 125B MoE, built and
tuned specifically for a 2× NVIDIA CMP 170HX 64 GB (GA100 / Ampere) box** —
and it runs anywhere else Ampere+ with ≥2×48 GB can.

The idea in one sentence: *attention is never quantized (full BF16), experts
are INT8 exactly where INT4 hurts, INT4 where it doesn't.* The result is the
highest-quality W4-class release we could measure, at ~6.1 average bits per
expert weight instead of a misleading "W4A16" label.

## Variants

| | This checkpoint | [INT8x17-AttnINT4](https://huggingface.co/SuperPUPS/Qwen3.8-Flash-Next-INT8x17-AttnINT4) |
|---|---|---|
| INT8 expert layers | 25 / 48 | 17 / 48 |
| Attention | **BF16** | INT4 g128 |
| KV pool @262K | 483 850 tok (1.85× concurrency) | **810 136 tok (3.09×)** |
| Prefill | 906–1156 tok/s | **~1363 tok/s** |
| Checkpoint | 212 GB | 202 GB |

Pick **this one** for maximum per-token quality; pick the
[high-concurrency variant](https://huggingface.co/SuperPUPS/Qwen3.8-Flash-Next-INT8x17-AttnINT4)
when you need more simultaneous long dialogues or faster prefill.

| Component | Precision | Notes |
|---|---|---|
| Attention q/k/v/o (12 full-attention layers) | **BF16** | byte-identical to the official FP8 checkpoint |
| Expert matrices — **25 of 48 layers** | **INT8** g128 sym | re-quantized from the FP8 checkpoint, msegrid-refined scales |
| Expert matrices — remaining 23 layers | INT4 g128 sym | unchanged from the W4A16 base |
| Routers, norms, `hc_*`, GDN inputs, QSA indexer, `lm_head`, embeddings, MTP, PLE | BF16 | unchanged |
| KV cache | BF16 at runtime | not baked into the checkpoint |

## Why this shape

Attention projections steer all 512 experts per layer and are the largest
systematic error source in uniform W4A16 — so they are **fully unquantized**
(here that is free of risk: the official FP8 checkpoint stores attention in
BF16, so the upgrade is a pure byte-copy, zero requantization).

Expert INT4 error, measured against the FP8 checkpoint, concentrates in a
minority of layers. We promoted exactly those — and stopped where the measured
quality return drops sharply while the VRAM cost does not.

## Where the 25 INT8 layers come from

Starting point: the community **W4A16** checkpoint (uniform INT4 g128 experts).
Reference proxy: the **official FP8 checkpoint**. Selection method: measure the
actual INT4 error per layer — rel-Frobenius + cosine over **1536 expert-matrix
pairs per layer** (512 experts × 3 projections), CPU-only.

**Wave 1 — 11 layers** (`17, 21, 25, 27, 28, 29, 35, 40, 44, 45, 47`), guided
by the community RCO sensitivity map as navigation. (Per-layer metrics from
that first pass were not retained.)

**Waves 2–3 — 14 layers**, chosen by the measured ranking below (INT4 error vs
FP8 reference; higher error = higher priority):

| Rank | Layer | rel-F | cos | Wave |
|---:|---:|---:|---:|---|
| 1 | 0 | 0.13253 | 0.99135 | 2 |
| 2 | 38 | 0.12674 | 0.99208 | 2 |
| 3 | 46 | 0.12636 | 0.99213 | 2 |
| 4 | 22 | 0.12475 | 0.99232 | 2 |
| 5 | 42 | 0.12468 | 0.99233 | 2 |
| 6 | 37 | 0.12446 | 0.99236 | 2 |
| 7 | 18 | 0.12444 | 0.99236 | 3 |
| 8 | 14 | 0.12400 | 0.99242 | 3 |
| 9 | 7 | 0.12394 | 0.99242 | 3 |
| 10 | 41 | 0.12359 | 0.99246 | 3 |
| 11 | 43 | 0.12311 | 0.99252 | 3 |
| 12 | 31 | 0.12308 | 0.99253 | 3 |
| 13 | 16 | 0.12308 | 0.99253 | 3 |
| 14 | 33 | 0.12301 | 0.99253 | 3 |

**Recipe** (reproducible with [`quantization/`](./quantization)): FP8 e4m3 +
128×128 block scales → bf16 → per-group g128 amax/127 → **msegrid** scale
refinement (percentile clipping + MSE grid search) → INT8 → pack 4/int32
(offset 128), compressed-tensors `pack-quantized`. Roundtrip INT8 cos vs
reference: **0.99995** (worst sampled matrix 0.99990).

## Verification and quality gates

Structural (all passed): weight_map surgery exact; 48 attention tensors
**byte-identical** to the FP8 checkpoint; 64 sampled INT8 matrices cos > 0.9999
(worst 0.999902); 300 random untouched tensors identical to base; untouched
shard files SHA256-identical; `total_size` matches. `SHA256SUMS.txt` covers
every file in this repo.

Live gates on the serving stack (all passed):

| Gate | Result |
|---|---|
| Golden probes (temp=0) | 3/3 |
| 10-task battery (math / logic / code / translate / RU writing, temp=0) | 10/10 |
| Determinism (8× same prompt) | 1/8 unique |
| Needle-in-a-haystack @128K | **HIT** (112 917 prompt tokens, verbatim) |
| Soak, 30 min / 4 concurrent workers | 853 requests, **0 errors** |

## Measured on the target hardware (2× CMP 170HX 64 GB, TP2+EP, MTP k=4)

| Metric | [INT8x17-AttnINT4 variant](https://huggingface.co/SuperPUPS/Qwen3.8-Flash-Next-INT8x17-AttnINT4) | This checkpoint |
|---|---|---|
| Checkpoint size | 202 GB | 212 GB |
| Prefill | **~1363 tok/s** | **906–1156 tok/s** |
| Decode, single stream | ~106 tok/s | **81–105 tok/s** (long-decode ≈ parity) |
| MTP acceptance | 57–65% | 57–65% (3.3–3.6 tok/step) |
| KV pool @262K context | **810 136 tokens** (3.09× concurrency) | **483 850 tokens** (1.85× concurrency) |
| Startup (warm) | ~5.5 min | ~5.5 min |

Honest cost note: expert bytes read per token grow +12.3% vs the
[INT8x17-AttnINT4 variant](https://huggingface.co/SuperPUPS/Qwen3.8-Flash-Next-INT8x17-AttnINT4)
(23 INT4 + 25 INT8 layers vs 31 + 17), which is why prefill is ~15% slower —
a deliberate quality-for-bandwidth trade. Decode is nearly unaffected because
MTP amortizes the step.

## Running the model

### Serving engine

These checkpoints were validated with a patched vLLM source snapshot, not
plain upstream vLLM.

- Engine: `vllm-qfn`
- Authoritative snapshot: branch `feature/qsa-int8-kv`, commit `e12683d245`
- Includes: PLE mmap offload, CUTLASS FP8 guard, TurboQuant K8V4 wiring, and
  `int8_per_token_head` KV-cache support
- Reported `vllm --version`:
  `0.28.1rc1.dev78+gb009560d6.d20260923.cu132`
- Source snapshot:
  [SuperPUPS/vllm-qfn-source](https://huggingface.co/datasets/SuperPUPS/vllm-qfn-source)
- Required PLE variables: `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_PLE_MMAP_TABLE=1`
- Optional INT8 KV cache (`--kv-cache-dtype int8_per_token_head`): validated on
  this checkpoint — KV pool 483 850 → 776 632–796 231 tok (1.85× → ~3×) at a
  ~4–5% prefill cost. Recipe + evidence:
  [SuperPUPS/Qwen3.8-Flash-Next-INT8KV](https://huggingface.co/datasets/SuperPUPS/Qwen3.8-Flash-Next-INT8KV)

Plain upstream vLLM 0.31 is not a drop-in replacement for 128 GB RAM hosts:
its Engram/PLE path pins the full ~102.4 GiB table and can exceed ~117 GiB
RSS.

Reference host note: zram-based swap was enabled on the test host. This is
host tuning only and is not required for the mmap path.

### Requirements

- **2× GPUs with ≥48 GB VRAM each**, Ampere or newer (this build was made for
  and tested on **2× CMP 170HX 64 GB**, GA100). TP=2 + expert parallel.
- **Host RAM depends on the vLLM build.** The model carries a 102.4 GB PLE
  embedding table. With the `vllm-qfn` snapshot above and
  `VLLM_PLE_MMAP_TABLE=1`, the tested path keeps the PLE table in mapped pages
  and ran with ~13 GB RSS, so **≥64 GB host RAM** is enough. With upstream
  vLLM 0.31 / Engram without mmap, the table becomes a fully pinned host-RAM
  allocation and startup can reach **~117 GB**, so use **≥135 GB host RAM**.
- Use the `vllm-qfn` source snapshot linked above, or another vLLM build that
  supports `VLLM_PLE_CPU_OFFLOAD=1` and `VLLM_PLE_MMAP_TABLE=1`. Do not bypass
  RAM checks on a 128 GB machine — it will OOM.
- ~212 GB free disk.

### Launch

```bash
export VLLM_PLE_CPU_OFFLOAD=1          # PLE table (102.4 GB) in host RAM
export VLLM_PLE_MMAP_TABLE=1           # required on <135 GiB hosts: zero-copy mmap
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1

vllm serve SuperPUPS/Qwen3.8-Flash-Next-INT8x25-AttnBF16 \
  --served-model-name Qwen3.8-Flash-Next \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.97 \
  --compilation-config '{"mode":"VLLM_COMPILE","cudagraph_mode":"FULL"}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "xhigh", "thinking_budget": 16384}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0}' \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

What matters, briefly:

- `--compilation-config … cudagraph_mode:"FULL"` — **required** for MTP: the
  PIECEWISE graph mode refuses to capture draft decodes.
- `--speculative-config … num_speculative_tokens:4` — MTP k=4 is the measured
  acceptance peak (2.6–3.6 accepted tokens per step).
- `--max-model-len 262144` — 262K context fits in 64 GB cards with this
  checkpoint (KV pool ≈ 484K tokens at this size).
- `--no-enable-flashinfer-autotune` — autotune breaks deterministic startup.
- `--override-generation-config … temperature:0.6` — the checkpoint's default
  sampling is unstable at temp 1.0 (truncated generations, tool spam); 0.6 is
  the Qwen3 thinking-mode recommendation and also raises MTP acceptance.
- Startup takes ~5.5 minutes on the CMP 170HX box (weights + PLE + CUDA
  graphs); first requests may JIT-spike for a few seconds — normal.

### Thinking control (chat-template kwargs)

This repo ships the original Qwen template plus one **backward-compatible**
extension:

- `reasoning_effort`: `xhigh` (default — "think carefully, validate
  assumptions"), `medium` (no instruction), `low` (brief thinking). Any other
  value raises, by design.
- `thinking_budget: N` — *new*: appends a soft "keep your thinking within
  approximately N tokens" instruction (e.g. `16384`). Soft, not a hard engine
  cap; without the kwarg behavior is identical to the original template.

Per-request override example:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Сколько будет 17*23?"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}
}'
```

## Reproducing the build

`quantization/` contains the full pipeline: `make_donor_index.py` (index the
FP8 reference), `rank_sensitive_layers.py` (measured per-layer INT4 error
ranking), `build_quantized_checkpoint.py` (INT8 re-quant + BF16 attention
byte-copy + config surgery), `verify_checkpoint.py` (structural verification).
Edit the path constants at the top of each script.

## Files and integrity

30 safetensors shards (~212 GB total), full tokenizer set, configs, patched
`chat_template.jinja`, `LICENSE`, and `SHA256SUMS.txt` covering every file.

## License & attribution

Derivative of **Qwen3.8-Flash-Next**, distributed under the **Qwen Community
License 1.0** (see `LICENSE`). Built on top of the community W4A16 quantized
checkpoint, re-quantized with reference to the official FP8 checkpoint
(attention tensors byte-copied from it). Upstream terms apply: attribution
required above 100M MAU / $20M monthly revenue; MaaS and commercial AI
work-assistant offerings require a separate license from Qwen.