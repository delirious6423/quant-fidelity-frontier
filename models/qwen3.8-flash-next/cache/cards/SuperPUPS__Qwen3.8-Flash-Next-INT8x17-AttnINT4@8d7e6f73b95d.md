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
- int4
- vllm
- qwen
- ampere
---

# Qwen3.8-Flash-Next — INT8x17 experts + INT4 attention (high-concurrency)

**The high-concurrency sibling of
[Qwen3.8-Flash-Next-INT8x25-AttnBF16](https://huggingface.co/SuperPUPS/Qwen3.8-Flash-Next-INT8x25-AttnBF16).**
Same hybrid-quantization method and the same target hardware —
**2× NVIDIA CMP 170HX 64 GB (GA100 / Ampere)** — but a lighter weight budget
that leaves **68% more KV cache**: **3.09× concurrent 262K-context dialogues
vs 1.85×**, and ~20% faster prefill, in exchange for slightly lower precision.

Pick by workload:

| | **This checkpoint** (INT8x17-AttnINT4) | [INT8x25-AttnBF16](https://huggingface.co/SuperPUPS/Qwen3.8-Flash-Next-INT8x25-AttnBF16) |
|---|---|---|
| INT8 expert layers | 17 / 48 | 25 / 48 |
| Attention (12 full-attn layers) | INT4 g128 (as the W4A16 base) | BF16 (byte-copied from FP8) |
| Avg bits per expert weight | ~5.4 | ~6.1 |
| Checkpoint size | 188.5 GiB (202 GB) | 199.4 GiB (212 GB) |
| KV pool @262K ctx | **810 136 tokens** | 483 850 tokens |
| Max concurrency @262K | **3.09×** | 1.85× |
| Prefill (2× CMP 170HX) | **~1363 tok/s** | 906–1156 tok/s |
| Decode, single stream | ~106 tok/s | 81–105 tok/s |
| Quality gates | all passed (see below) | all passed |

## Composition

| Component | Precision | Notes |
|---|---|---|
| Expert matrices — **17 of 48 layers** | **INT8** g128 sym | re-quantized from the official FP8 checkpoint, msegrid-refined scales |
| Expert matrices — remaining 31 layers | INT4 g128 sym | unchanged from the W4A16 base |
| Attention q/k/v/o | INT4 g128 sym | as the W4A16 base |
| Routers, norms, `hc_*`, GDN inputs, QSA indexer, `lm_head`, embeddings, MTP, PLE | BF16 | unchanged |
| KV cache | BF16 at runtime | not baked into the checkpoint |

## Where the 17 INT8 layers come from

Starting point: the community **W4A16** checkpoint (uniform INT4 g128 experts).
Reference proxy: the **official FP8 checkpoint**. Expert INT4 error, measured
against that reference (rel-Frobenius + cosine over **1536 expert-matrix pairs
per layer** = 512 experts × 3 projections), concentrates in a minority of
layers — so INT8 was spent only there.

**Wave 1 — 11 layers** (`17, 21, 25, 27, 28, 29, 35, 40, 44, 45, 47`), guided
by the community RCO sensitivity map (ISTA-DASLab GSQ-RCO-GGUF allocation) as
navigation. (Per-layer metrics from that first pass were not retained.)

**Wave 2 — 6 layers**, chosen by the measured ranking (INT4 error vs FP8
reference; higher error = higher priority):

| Rank | Layer | rel-F | cos |
|---:|---:|---:|---:|
| 1 | 0 | 0.13253 | 0.99135 |
| 2 | 38 | 0.12674 | 0.99208 |
| 3 | 46 | 0.12636 | 0.99213 |
| 4 | 22 | 0.12475 | 0.99232 |
| 5 | 42 | 0.12468 | 0.99233 |
| 6 | 37 | 0.12446 | 0.99236 |

The remaining 31 INT4 layers survive even at aggressive budgets — measured
returns beyond this point drop sharply while VRAM cost keeps growing.

**Recipe** (reproducible with [`quantization/`](./quantization)): FP8 e4m3 +
128×128 block scales → bf16 → per-group g128 amax/127 → **msegrid** scale
refinement (percentile clipping + MSE grid search) → INT8 → pack 4/int32
(offset 128), compressed-tensors `pack-quantized`. Roundtrip INT8 cos vs
reference: **0.999962** (parent INT4 was 0.9925).

## Verification and quality gates

Structural (ALL PASS, 25/25): weight_map identity; target tensor counts;
config regex catches the 17 layers and not the control layer or routers;
**pack roundtrip bit-exact**; INT8 cos vs reference 0.999962 > parent INT4
0.9925; scale shapes; 300 random untouched tensors identical; sizes.
`SHA256SUMS.txt` covers every file in this repo.

Live gates on the serving stack (all passed):

| Gate | Result |
|---|---|
| Golden probes (temp=0) | 3/3 |
| 10-task battery (math / logic / code / translate / RU writing, temp=0) | 10/10 |
| In-box determinism (8× same prompt) | 7/8 (single one-word flip ≈12%, a stack property) |
| Needle-in-a-haystack @128K | **HIT** |
| Soak, 30 min / 4 concurrent workers | 1164 requests, **0 errors** |

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
- The INT8 KV cache mode (`--kv-cache-dtype int8_per_token_head`) is an
  engine-level option and applies to this checkpoint too; it was validated on
  the INT8x25 sibling, not separately re-measured here. Recipe + evidence:
  [SuperPUPS/Qwen3.8-Flash-Next-INT8KV](https://huggingface.co/datasets/SuperPUPS/Qwen3.8-Flash-Next-INT8KV)

Plain upstream vLLM 0.31 is not a drop-in replacement for 128 GB RAM hosts:
its Engram/PLE path pins the full ~102.4 GiB table and can exceed ~117 GiB
RSS.

Reference host note: zram-based swap was enabled on the test host. This is
host tuning only and is not required for the mmap path.

### Requirements

- **2× GPUs with ≥48 GB VRAM each**, Ampere or newer (built for and tested on
  **2× CMP 170HX 64 GB**, GA100). TP=2 + expert parallel.
- **Host RAM depends on the vLLM build.** The model carries a 102.4 GB PLE
  embedding table. With the `vllm-qfn` snapshot above and
  `VLLM_PLE_MMAP_TABLE=1`, the tested path keeps the PLE table in mapped pages
  and ran with ~13 GB RSS, so **≥64 GB host RAM** is enough. With upstream
  vLLM 0.31 / Engram without mmap, the table becomes a fully pinned host-RAM
  allocation and startup can reach **~117 GB**, so use **≥135 GB host RAM**.
- Use the `vllm-qfn` source snapshot linked above, or another vLLM build that
  supports `VLLM_PLE_CPU_OFFLOAD=1` and `VLLM_PLE_MMAP_TABLE=1`. Do not bypass
  RAM checks on a 128 GB machine — it will OOM.
- ~202 GB free disk.

### Launch

```bash
export VLLM_PLE_CPU_OFFLOAD=1          # PLE table (102.4 GB) in host RAM
export VLLM_PLE_MMAP_TABLE=1           # required on <135 GiB hosts: zero-copy mmap
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ENGINE_READY_TIMEOUT_S=1800
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=1

vllm serve SuperPUPS/Qwen3.8-Flash-Next-INT8x17-AttnINT4 \
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
  --default-chat-template-kwargs '{"preserve_thinking": true, "reasoning_effort": "xhigh"}' \
  --override-generation-config '{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0.0}' \
  --trust-remote-code \
  --host 0.0.0.0 --port 8000
```

What matters, briefly:

- `--compilation-config … cudagraph_mode:"FULL"` — **required** for MTP: the
  PIECEWISE graph mode refuses to capture draft decodes.
- `--speculative-config … num_speculative_tokens:4` — MTP k=4 is the measured
  acceptance peak.
- `--max-model-len 262144` — 262K context fits in 64 GB cards with this
  checkpoint (KV pool ≈ 810K tokens, i.e. **3.09× concurrent full-length
  dialogues**); this is the point of this variant.
- `--no-enable-flashinfer-autotune` — autotune breaks deterministic startup.
- `--override-generation-config … temperature:0.6` — the checkpoint's default
  sampling is unstable at temp 1.0 (truncated generations, tool spam); 0.6 is
  the Qwen3 thinking-mode recommendation and also raises MTP acceptance.
- Startup takes ~5.5 minutes on the CMP 170HX box (weights + PLE + CUDA
  graphs); first requests may JIT-spike for a few seconds — normal.

### Thinking control (chat-template kwargs)

This repo ships the **original Qwen chat template** (unmodified). It supports
`reasoning_effort`: `xhigh` (default — "think carefully, validate
assumptions"), `medium` (no instruction), `low` (brief thinking). Per-request
override:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "Сколько будет 17*23?"}],
  "chat_template_kwargs": {"reasoning_effort": "low"}
}'
```

(If you need the soft `thinking_budget` kwarg, it is documented in the
[INT8x25-AttnBF16 card](https://huggingface.co/SuperPUPS/Qwen3.8-Flash-Next-INT8x25-AttnBF16)
— the template patch is drop-in compatible with this checkpoint too.)

## Reproducing the build

`quantization/` contains the full pipeline: `make_donor_index.py` (index the
FP8 reference), `rank_sensitive_layers.py` (measured per-layer INT4 error
ranking), `build_quantized_checkpoint.py` (INT8 re-quant wave + config
surgery), `verify_checkpoint.py` (structural verification). Edit the path
constants at the top of each script.

## Files and integrity

30 safetensors shards (188.5 GiB total), full tokenizer set, configs, the
original `chat_template.jinja`, `LICENSE`, and `SHA256SUMS.txt` covering every
file.

## License & attribution

Derivative of **Qwen3.8-Flash-Next**, distributed under the **Qwen Community
License 1.0** (see `LICENSE`). Built on top of the community W4A16 quantized
checkpoint, re-quantized with reference to the official FP8 checkpoint.
Upstream terms apply: attribution required above 100M MAU / $20M monthly
revenue; MaaS and commercial AI work-assistant offerings require a separate
license from Qwen.