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
- moe
- nvfp4
- fp4
- fp8
- modelopt
- vllm
- blackwell
- compressed-tensors
---

# Google Gemma-4-26B-A4B-it (ModelOpt Blackwell NVFP4)

**Gemma-4-26B-A4B-it-NVFP4** is a post-training quantized (PTQ) mixed-precision release of Google's **Gemma-4-26B-A4B-it** Mixture-of-Experts (MoE) foundation model, optimized specifically for native FP4 Tensor Core execution on **NVIDIA Blackwell architecture (GB10 / DGX Spark, GB200, B200, RTX 50-series)** using NVIDIA ModelOpt and vLLM.

---

## ⚡ Performance & Footprint

* **Checkpoint Footprint**: **16.07 GiB** (down from 49.2 GiB in BF16, a **67.3% reduction** in parameter footprint).
* **Hardware Target**: NVIDIA Blackwell architectures featuring native NVFP4 matrix multiply-accumulate instructions.
* **Unified Memory Headroom**: Fits comfortably on a single 128 GB DGX Spark or RTX 5090 GPU while leaving ample VRAM for extensive KV-cache allocation across long context sequences (up to 262,144 tokens).
* **Throughput**: Accelerates single-stream decode and prefill latency via CUTLASS and FlashInfer FP4 MoE kernels.

---

## 🔬 Quantization Architecture & Precision Topology

Gemma-4-26B-A4B-it utilizes a sparse Mixture-of-Experts (MoE) topology containing 128 routing experts with top-4 active routing per token ($26\text{B}$ total parameters, $\sim 4\text{B}$ active per token). Straightforward uniform quantization of MoE models can lead to representation collapse if router distributions or normalization layers are aggressively truncated. 

This model follows a calibrated mixed-precision topology:

| Component | Target Precision | Technical Rationale |
| :--- | :--- | :--- |
| **MoE Expert MLPs (128 Experts)** | **NVFP4 (E2M1, Block Size 16)** | Represents $>85\%$ of parameter volume. Block-wise quantization with FP8 scaling factors preserves expert specialization with minimal perplexity loss. |
| **Attention Layers ($q, k, v, o$)** | **Native BF16** | Preserves Rotary Position Embedding (RoPE) geometry and query-key dot-product precision across long contexts. |
| **MoE Gating & Routers** | **Native BF16** | Router logits maintain high dynamic range to prevent expert routing drift and load imbalance. |
| **Embeddings & LM Head** | **Native BF16** | Preserves token entropy across the 256,000 vocabulary tokens. |

### Block-Wise Mathematical Quantization Formulation

Linear layer weights are partitioned into contiguous blocks of $B = 16$ elements along the inner channel dimension. For each block $w \in \mathbb{R}^{16}$, an FP8 scaling factor $S_{\text{block}}$ is calculated:

$$S_{\text{block}} = \frac{\max_{i \in \{1 \dots 16\}} (|w_i|)}{V_{\max}}$$

Where $V_{\max} = 6.0$ is the maximum representable dynamic range value in FP4 ($E2M1$ format). The individual weights are quantized via:

$$q_i = \text{clip}\left(\left\lfloor \frac{w_i}{S_{\text{block}}} \right\rceil, -V_{\max}, V_{\max}\right)$$

During GEMM execution, Blackwell FP4 Tensor Cores perform fused matrix multiply-accumulate against the reconstructed weights:

$$\hat{w}_i = q_i \cdot S_{\text{block}}$$

---

## ⚠️ Important: Tool-Calling Parser Bugfix in vLLM

When deploying Gemma-4 with tool calling in stock vLLM, tool calls can be dropped or fragmented when transitioning directly out of the thinking channel (e.g., `<channel|>call:...` without leading whitespace).

* **Upstream Bug Report**: [vllm-project/vllm#54256](https://github.com/vllm-project/vllm/issues/54256)
* **Official Pull Request**: [vllm-project/vllm#54257](https://github.com/vllm-project/vllm/pull/54257) — *`[Bugfix][Parser] Support bare call: and whitespace-free channel transitions in Gemma4 parser`*

### Recommended Serving Configuration
Until PR #54257 is merged into an official vLLM release, you can mount the patched `gemma4_tool_parser.py` into your vLLM container or use the patch from PR #54257 to guarantee 100% reliable tool call extraction.

---

## 🚀 Serving with vLLM

This model is in standard `modelopt` / `compressed-tensors` format and can be served directly using **vLLM**:

```bash
vllm serve wyattearp/Gemma-4-26B-A4B-it-NVFP4 \
  --served-model-name google/gemma-4-26B-A4B-it \
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

This model is released under the **Gemma Terms of Use**. Please review the license conditions before commercial deployment: [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
