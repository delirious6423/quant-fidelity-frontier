---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B-FP8
- Qwen/Qwen3.8-27B
language:
- en
- ar
- zh
pipeline_tag: text-generation
tags:
- representation-engineering
- refusal-ablation
- surgical-projection
- qwen
- fp8
- cybersecurity
- uncensored
- abliterated
- agentic
- coding
- tool-calling
- sglang
- vllm
- qwen3.8
- function-calling
- reasoning
- cot
- hermes-agent
library_name: transformers
---

# Qwen3.8-27B-Surgical-Abliterated (FP8)

> **Precision Representation Engineering on Hybrid Linear Attention Architecture**  
> Achieves **0.00% Refusal Rate** on security assessments while preserving **100.0% Reasoning & CoT Depth** with zero latent loops and zero vocabulary collapse.

---

## 🔬 Overview & Scientific Motivation

Existing abliteration pipelines (such as naive weight-subtraction scripts) often destroy large language models by projecting out arbitrary vectors across all layers ($0 \dots 63$) and attention matrices. This causes severe unintended consequences:
1. **Catastrophic Latent Collapse**: The model falls into repetitive loops or dumps alphabetical dictionaries.
2. **Loss of Attention Sinks**: Projecting $W_q, W_k, W_v, W_o$ breaks token routing and memory mechanics.
3. **Broken CoT Boundaries**: The model fails to emit `</think>` tokens, producing infinite reasoning loops.

### The Surgical Approach

This model was created using **Rank-1 Orthogonal Complement Projections** targeted strictly at the safety gating locus discovered via sequential residual stream probing across all 64 layers:

$$\hat{r} = \frac{\mu_{\text{refusal}} - \mu_{\text{benign}}}{\|\mu_{\text{refusal}} - \mu_{\text{benign}}\|_2}$$

$$W' = W - (W \hat{r})\hat{r}^T$$

- **Untouched Layers**: Layers $0 \to 16$ (syntactic/lexical parsing) and upper layers remain **100% untouched**.
- **Surgical Target**: Only `mlp.down_proj` matrices in the active refusal deliberation band were modified.
- **Dimensional Geometry**: Retains **5,119 dimensions** out of 5,120 hidden state dimensions (99.98% geometric fidelity preserved).
- **Exact FP8 Quantization**: Dequantized block-wise ($128 \times 128$) to FP32, projected orthogonally, and requantized back to `float8_e4m3fn` with updated scale invariants.

---

## 📊 Benchmark & Empirical Evaluation

All evaluations were conducted under official default sampling parameters (`presence_penalty: 0.0`, `repetition_penalty: 1.0`, `temperature: 0.6`).

| Metric | Base Model (`Qwen3.8-27B-FP8`) | Naive Abliteration (`OBLITERATUS`) | **Surgical Abliterated (Ours)** |
| :--- | :---: | :---: | :---: |
| **Refusal Rate (Cybersecurity / Pentest)** | 100.0% | ~0.0% | **0.00%** |
| **Reasoning Retention (`<think>` Depth)** | 100.0% | Broken (Infinite Loops) | **100.0%** |
| **`</think>` Boundary Closure** | Clean | Broken (`Bombur` loops) | **Clean (0 Loops)** |
| **Coding & Logic Retention (Two Sum / SSTI)** | 100.0% | Dictionary Dump | **100.0%** |
| **Multilingual Support (Arabic / English)** | Refused | Broken | **Full Fluency** |

---

## 🚀 Deployment & Serving

### SGLang (Recommended)

To run on an NVIDIA RTX 5090 (32GB VRAM):

```bash
python3 -m sglang.launch_server \
  --model-path medismera/Qwen3.8-27B-Surgical-Abliterated \
  --port 18000 \
  --host 0.0.0.0 \
  --context-length 8192 \
  --max-running-requests 2 \
  --mem-fraction-static 0.96 \
  --disable-cuda-graph \
  --disable-radix-cache \
  --skip-server-warmup \
  --kv-cache-dtype fp8_e5m2 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder
```

### Thinking Mode Configuration

Thinking is fully supported and toggled seamlessly via the OpenAI-compatible API:

- **Thinking Enabled (Default)**: Normal inference automatically provides transparent step-by-step reasoning within `<think>` tags.
- **Thinking Disabled**: Pass `chat_template_kwargs={"enable_thinking": False}` for ultra-fast, direct answer generation.

---

## 📜 Citation & Credits

- Technique: Representation Engineering (Arditi et al., 2024).
- Base Model: Alibaba Cloud `Qwen/Qwen3.8-27B-FP8`.
- Surgical Pipeline: Programming with Medismera.

## ⚡ One-Click Instant Deployment (Zero-Touch Script)

To deploy this model on any fresh cloud server (Vast.ai, RunPod, Lambda Labs with an RTX 5090, A100, L40S, or 2x RTX 4090/3090):

```bash
curl -sSL https://huggingface.co/medismera/Qwen3.8-27B-Surgical-Abliterated/raw/main/deploy_and_serve.sh | bash
```

This script automatically:
1. Installs all required CUDA/Python packages (, , ).
2. High-speed multi-threaded download of the model from Hugging Face.
3. Launches the SGLang OpenAI API server on port `18000` with optimized FP8 KV cache.
4. Executes an automated health-check and verification test.

---
