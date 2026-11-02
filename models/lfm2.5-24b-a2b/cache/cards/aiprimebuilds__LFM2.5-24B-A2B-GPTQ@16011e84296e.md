---
library_name: transformers
pipeline_tag: text-generation
license: other
license_name: lfm1.0
license_link: LICENSE
base_model: LiquidAI/LFM2.5-24B-A2B
base_model:quantized: LiquidAI/LFM2.5-24B-A2B
author: aiprimebuilds
language:
- en
tags:
- gptq
- 4-bit
- int4
- moe
- lfm2_moe
- lfm2.5
- liquid
- text-generation
- conversational
- vllm
- quantized
---

# LFM2.5-24B-A2B-GPTQ

4-bit **GPTQ** (`group_size=32`) quantization of [`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) — a 24B-parameter hybrid short-convolution + Grouped Query Attention (GQA) Mixture-of-Experts (MoE) reasoning model with only **~2B active parameters per token**.

Produced with **GPTQModel 7.6.0** (`LFM2MoeQModel`), which writes all 64 experts in every MoE layer as discrete per-expert `w1/w3/w2` quantized Linears, the layout vLLM's native `lfm2_moe` loader expects (no `--trust-remote-code`).

> **What this is:** A compressed serving artifact of `LiquidAI/LFM2.5-24B-A2B`, released under the same license (**LFM Open License v1.0 / `lfm1.0`**). See `LICENSE` for full terms and commercial threshold provisions.

---

## Artifact Summary

| Field | Value |
|---|---|
| **Base Model** | [`LiquidAI/LFM2.5-24B-A2B`](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B) |
| **Architecture** | `Lfm2MoeForCausalLM` (`model_type: lfm2_moe`) |
| **Parameters** | 23.97B total (`23,971,585,280`) / ~2B active per token |
| **Quantization Method** | GPTQ (`W4A16` — 4-bit `int4` weights, `bfloat16` activations) |
| **Group Size** | `32` (`desc_act: false`, `act_group_aware: true`, `sym: true`) |
| **Calibration Dataset** | `allenai/c4` (English, 1,024 samples) |
| **Artifact Footprint** | 13.4 GiB on disk |
| **Native Context Window** | 131,072 tokens |
| **Target Runtime** | vLLM (`--quantization gptq`, native `lfm2_moe` support) |
| **Toolchain** | GPTQModel 7.6.0, Transformers, vLLM, PyTorch (CUDA) |

---

## Architecture & Quantization Notes

`LFM2.5-24B-A2B` is a **hybrid short-convolution + full-attention MoE** architecture engineered for high throughput and a minimal KV-cache footprint:

- **40 Total Layers (3:1 Hybrid Stack):** **30 short-convolution layers** interleaved with **10 full-attention GQA layers** (`num_dense_layers: 2` at the base of the stack). Because short-convolution layers are stateless with respect to sequence length, **only 10 layers allocate KV cache**.
- **MoE Topology:** **64 total experts** with **4 experts routed per token** (`hidden_size: 2048`, `moe_intermediate_size: 1536`).
- **Attention Geometry:** 32 query heads, 8 key-value heads (`head_dim: 64`), `max_position_embeddings: 131072`, `vocab_size: 128000` (`tie_word_embeddings: true`).
- **Group Size 32:** Quantized at `group_size=32` (rather than the generic 128 used for dense Llama models) to preserve fidelity across the compact 1536-dim expert projections.
- **Per-expert layout:** GPTQModel's `LFM2MoeQModel` with `W1W3W2MoELifecycleHooks` quantizes the sparse MoE block as discrete per-expert `w1/w3/w2` Linears, the tensor structure vLLM's `lfm2_moe` FusedMoE mapper expects.

### What is quantized

Checked against `model.safetensors.index.json`:

| Precision | Modules |
|---|---|
| **4-bit GPTQ** | All MoE experts `w1/w2/w3` (38 MoE layers × 64 experts = 2,432 each) · dense FFN `w1/w2/w3` (layers 0–1) · short-conv `in_proj`/`out_proj` (30 layers) · attention `q/k/v_proj` (10 layers) |
| **bf16 (unquantized)** | MoE router `gate` (38) · attention `out_proj` (10) · short-conv kernels `conv.conv` (30) · all norms · `embed_tokens` (`lm_head` is tied and not quantized) |

Keeping the router gates in bf16 preserves expert-selection quality. The experts, which hold most of the weights, carry the 4-bit savings.

---

## Serving with vLLM

`lfm2_moe` is a natively supported architecture in vLLM, so the intended load path needs **no `--trust-remote-code`**.

### CLI Server Launch

```bash
vllm serve aiprimebuilds/LFM2.5-24B-A2B-GPTQ \
  --quantization gptq \
  --max-model-len 65536 \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --port 8000
```

*(Tip: On 16 GB – 24 GB consumer/workstation GPUs, add `--kv-cache-dtype fp8` to cut KV cache memory in half.)*

### Python Offline Inference

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="aiprimebuilds/LFM2.5-24B-A2B-GPTQ",
    quantization="gptq",
    max_model_len=32768,
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_k=50,
    repetition_penalty=1.05,
    max_tokens=512,
)

out = llm.generate(["Hello, my name is"], sampling_params)
print(out[0].outputs[0].text)
```

---

## VRAM & KV Cache Scaling

Because only **10 layers** use full attention (`10 layers × 2 [K+V] × 8 KV heads × 64 head_dim × 2 bytes`), the BF16 KV cache consumes only **20 KiB per token** (~0.02 MiB/token)—roughly **75% smaller** than a dense 40-layer model. The table below is **calculated from `config.json`, not measured**:

| `max_model_len` | KV Cache (BF16) | KV Cache (FP8) | Est. Total VRAM (Weights + Runtime + KV) | Fits GPU Tier |
|---|---|---|---|---|
| **8,192** | 0.16 GiB | 0.08 GiB | ~14.7 GiB | 16 GB (RTX 4080 SUPER) |
| **16,384** | 0.31 GiB | 0.16 GiB | ~14.9 GiB | 16 GB |
| **32,768** | 0.63 GiB | 0.31 GiB | ~15.2 GiB | 16 GB – 20 GB |
| **65,536** | 1.25 GiB | 0.63 GiB | ~15.8 GiB | 20 GB – 24 GB (RTX 4000 Ada / 3090 / 4090) |
| **131,072** | **2.50 GiB** | **1.25 GiB** | **~17.1 GiB** | **20 GB – 24 GB (RTX 4000 Ada / 3090 / 4090)** |

- **Recommended Generation Defaults:** `temperature=0.1`, `top_k=50`, `repetition_penalty=1.05` for instruct and tool-use workloads (`temperature=0.6`, `top_p=0.95` for open-ended reasoning evaluations).

---

## Limitations & License Notes

- **4-bit Compression Tradeoff:** `int4` weight quantization is inherently lossy compared to `bfloat16` or `FP8`. Use this artifact when minimizing VRAM footprint and maximizing single-GPU concurrency/throughput are primary goals.
- **vLLM / GPTQModel Dependency:** Loading the quantized MoE expert tree requires a runtime with `lfm2_moe` FusedMoE support (such as current releases of vLLM or GPTQModel >= 7.6.0).
- **License (`lfm1.0`):** Redistribution of derivative works is permitted with attribution under the **LFM Open License v1.0**, subject to commercial revenue threshold limitations (see `LICENSE` §5).

---

## Produced By

**Chris Shipley** ([`aiprimebuilds`](https://huggingface.co/aiprimebuilds)) · Co-Founder & CTO, [Nutmeg Consulting, LLC](https://www.nutmegit.com/) · New York · [LinkedIn](https://www.linkedin.com/in/chrisshipley)
