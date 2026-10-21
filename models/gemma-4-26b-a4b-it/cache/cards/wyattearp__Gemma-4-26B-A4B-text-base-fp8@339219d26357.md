---
language:
- en
license: gemma
license_link: https://ai.google.dev/gemma/terms
base_model: google/gemma-4-26B-A4B-it
pipeline_tag: text-generation
tags:
- gemma4
- google
- text-only
- language-model
- moe
- fp8
- modelopt
- vllm
- ada
- l40s
- rtx-4090
- blackwell
- compressed-tensors
---

# Google Gemma-4-26B-A4B-text-base-fp8 (ModelOpt Calibrated FP8 for Ada & L40S)

**Gemma-4-26B-A4B-text-base-fp8** is a post-training quantized (PTQ) FP8 release of Google's **Gemma-4-26B-A4B-it** Mixture-of-Experts (MoE) model. It is modality-stripped to a pure language model (`Gemma4ForCausalLM`) and quantized using NVIDIA ModelOpt with per-tensor scalar scaling factors (`axis=None`) specifically engineered for native single-GPU serving on **NVIDIA L40S (48 GB GDDR6)**, Ada Lovelace (RTX 4090 / SM 8.9), and Blackwell architectures.

---

## ⚡ Performance & Footprint

* **Checkpoint Footprint**: **24.2 GiB** (down from 50.47 GiB in BF16, a **52% reduction**).
* **Hardware Target**: **NVIDIA L40S (48 GB)**, RTX 4090, RTX 6000 Ada, and Blackwell GB10 / B200.
* **Unified Memory Headroom**: Takes **24.6 GiB** VRAM loaded in vLLM, leaving **~23.4 GiB free for KV cache** on a single 48 GB L40S GPU.
* **Architecture**: Modality-stripped pure text backbone (`Gemma4ForCausalLM`), eliminating multimodal projector and vision tower VRAM overhead.
* **Scales Format**: 23,450 per-tensor scalar scaling factors, matching native Ada Lovelace FP8 Tensor Core hardware GEMM instructions (`torch._scaled_mm`).

---

## 🔬 Quantization Topology

| Component | Target Precision | Technical Rationale |
| :--- | :--- | :--- |
| **MoE Expert MLPs (128 Experts)** | **FP8 (E4M3, Per-Tensor Scales)** | Calibrated across 64 diverse domain prompts. Per-tensor scalar scaling factors match hardware Tensor Core instructions. |
| **Attention Layers ($q, k, v, o$)** | **FP8 (E4M3, Per-Tensor Scales)** | Calibrated activation and weight scaling for high-throughput inference. |
| **MoE Gating & Routers** | **Native BF16** | Preserves high dynamic range router logits to prevent expert routing collapse. |
| **Embeddings & LM Head** | **Native BF16** | Preserves token entropy across the 256,000 vocabulary tokens. |

---

## ⚠️ Important: Tool-Calling Parser Bugfix in vLLM

When deploying Gemma-4 with tool calling in stock vLLM, tool calls can be dropped or fragmented when transitioning directly out of the thinking channel (e.g., `<channel|>call:...` without leading whitespace).

* **Upstream Bug Report**: [vllm-project/vllm#54256](https://github.com/vllm-project/vllm/issues/54256)
* **Official Pull Request**: [vllm-project/vllm#54257](https://github.com/vllm-project/vllm/pull/54257) — *`[Bugfix][Parser] Support bare call: and whitespace-free channel transitions in Gemma4 parser`*

---

## 🚀 Serving with vLLM

This model is in standard `modelopt` FP8 format and runs natively on single-GPU L40S or Blackwell setups:

```bash
vllm serve wyattearp/Gemma-4-26B-A4B-text-base-fp8 \
  --served-model-name google/gemma-4-26B-A4B-text-base-fp8 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype auto \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 262144 \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --enable-auto-tool-choice
```

---

## 📄 License

Inherits upstream **Gemma Terms of Use** from Google DeepMind: [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
