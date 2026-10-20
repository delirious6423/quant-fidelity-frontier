---
license: apache-2.0
base_model: google/gemma-4-26B-A4B-it
tags:
  - gemma4
  - moe
  - nvfp4
  - compressed-tensors
  - vllm
  - blackwell
  - hybrid-quantization
  - instruction-following
  - structured-output
  - 32gb-gpu
library_name: transformers
pipeline_tag: text-generation
---

# Gemma 4 26B A4B IT — NVFP4 hybrid **IF** variant (shared MLP BF16)

Instruction-following–first hybrid NVFP4 of [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it).

**Canonical** undeadindustries Gemma 4 26B A4B hybrid (~**16.4 GiB**, 32 GB GPU target). The earlier “shared MLP NVFP4” hybrid and its drafter were removed after this recipe proved better on structured IF workloads.

Built and validated on **DGX Spark (GB10, 128 GB UMA)**. Intended deployment: **single 32 GB GPU**, TP=1.

## Quantization recipe

| Component | Precision |
|-----------|-----------|
| Attention (`q/k/v/o`) | **BF16** |
| MoE router | **BF16** |
| **Shared dense MLP** (`gate/up/down`) | **BF16** |
| MoE expert linears (128×8) | **NVFP4 W4A4** |

Keeping the shared MLP in BF16 (every token hits it) improves outline hierarchy, ordered steps, heading discipline, and multi-source disambiguation vs quantizing that path to NVFP4, while experts stay NVFP4 so disk/VRAM stay ~16 GB.

AutoRound flag: `AR_PROTECT_SHARED_MLP=1` with bare `layer_config` keys `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` (not pre-escaped regex — see project Bug X).

## Architecture (MoE + hybrid attention)

| Item | Value |
|------|-------|
| Total params | ~25.2B |
| Active params / token | ~3.8B (A4B) |
| Layers | 30 (hybrid 5-sliding : 1-global) |
| MoE | 128 experts, top-8 routed + shared dense MLP |
| Text-only | Vision stripped; `Gemma4ForCausalLM` |

## Verified quality (local)

Greedy **`/v1/chat/completions`** on vLLM 0.20.2:

- Baseline chat: Paris, arithmetic — **PASS**
- **Structured probe battery** (markdown outline order, no raw JSON headings, numbered steps, N64 vs DS disambiguation): **0 failures**
- **GameFAQs-style A/B probes** (EarthBound/Chrono spines, Advance Wars `###`/`#####` hierarchy, SM64 no-JSON/no-DS, elaborative density floor): **0 failures** on the in-repo battery

Use the **chat API** for `-it` models — not raw `/v1/completions`.

## Serve on a 32 GB GPU

```bash
vllm serve undeadindustries/gemma-4-26b-a4b-it-nvfp4-hybrid-smlp \
  --port 8000 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enforce-eager \
  --trust-remote-code \
  --generation-config vllm
```

**Requires vLLM ≥ 0.20.2** with NVFP4 + compressed-tensors support.

### DFlash speed (pair with the smlp drafter)

```bash
vllm serve undeadindustries/gemma-4-26b-a4b-it-nvfp4-hybrid-smlp \
  --port 8000 --max-model-len 8192 \
  --gpu-memory-utilization 0.90 --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enforce-eager --trust-remote-code --generation-config vllm \
  --speculative-config '{"model":"undeadindustries/gemma-4-26b-a4b-it-dflash-drafter-nvfp4-smlp","method":"dflash","num_speculative_tokens":15}'
```

Use the **paired smlp drafter** above — a BF16-base or mismatched-verifier drafter gets ~0% acceptance.

## Known limits

- **TP=2 MoE Gemma 4** broken on vLLM ([#39595](https://github.com/vllm-project/vllm/issues/39595)) — **TP=1 only**
- **BF16 KV cache** recommended (avoid default FP8 KV on Gemma 4 hybrid SWA)
- MoE NVFP4 on GB10 observed with **MARLIN** backend for experts

## Provenance

- **Quantization:** AutoRound 0.14.1, `scheme=NVFP4`, `iters=200`, `nsamples=256`, `llm-compressor` export
- **Hardware:** NVIDIA DGX Spark (GB10)
- **Published:** 2026-07-11 (supersedes earlier shared-MLP-NVFP4 hybrid, removed 2026-07-12)

## License

Derived from [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) (Apache 2.0).  
Subject to [Gemma Terms of Use](https://ai.google.dev/gemma/docs/gemma_4_license).
