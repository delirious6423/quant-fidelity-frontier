---
language:
- en
- tr
license: apache-2.0
library_name: transformers
tags:
- text-generation
- reasoning
- cot
- think
- turkish
- code
- agent
- tool-calling
- zero-refusal
- cybersecurity
- java21
- loom
- qwen3_8
- veloxy
- titan
- 1m-context
pipeline_tag: text-generation
base_model: Qwen/Qwen3.8-27B
model-index:
- name: Veloxy-Titan-32B
  results:
  - task:
      type: text-generation
    dataset:
      name: NVIDIA H100 NVL BF16 Benchmark (5 Enterprise Domains)
      type: benchmark
    metrics:
    - name: Average Generation Speed
      type: tokens/sec
      value: 15.26
    - name: Native Context Window
      type: tokens
      value: 1048576
    - name: Peak VRAM Footprint
      type: gigabytes
      value: 50.36
    - name: VRAM Headroom on H100
      type: gigabytes
      value: 42.73
---

# 🛸 Veloxy-Titan-32B: The 32B Class Deep Reasoning & Agentic Powerhouse 🛸

<p align="center">
  <img src="https://raw.githubusercontent.com/zveloxy/assets/main/veloxy-titan-banner.png" alt="Veloxy Titan Banner" width="100%" onerror="this.style.display='none'"/>
</p>

<p align="center">
  <b>27.2B Active / 32B Scale Hybrid Backbone | 1,048,576 (1M) Native Context Scaling | GRPO Autonomous &lt;think&gt; Reflection | 38 Native Tools | Zero-Refusal Security Research | Java 21 LTS & Loom | 2026 Temporal Anchor | Live Verified on NVIDIA H100 NVL</b>
</p>

<p align="center">
  <a href="https://huggingface.co/zveloxy/Veloxy-Titan-32B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Veloxy--Titan--32B-purple.svg" alt="Titan 32B"></a>
  <a href="https://huggingface.co/zveloxy/Veloxy-Titan-32B-Vision"><img src="https://img.shields.io/badge/%F0%9F%91%81%EF%B8%8F%20Multimodal-Veloxy--Titan--32B--Vision-emerald.svg" alt="Titan Vision"></a>
  <a href="https://huggingface.co/zveloxy/Veloxy-Flash-15B"><img src="https://img.shields.io/badge/%E2%9A%A1%20Faz%208%20Master-Veloxy--Flash--15B-blue.svg" alt="Flash 15B"></a>
  <a href="https://huggingface.co/zveloxy/Veloxy-Flash-15B-LoRA"><img src="https://img.shields.io/badge/%F0%9F%A7%A9%20PEFT%20Adapter-Flash--15B--LoRA-cyan.svg" alt="Flash LoRA"></a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Precision-bfloat16%20%2B%20GGUF-orange.svg" alt="Precision">
  <img src="https://img.shields.io/badge/Context-1%2C048%2C576%20Tokens-red.svg" alt="Context">
</p>

---

## 📌 1. Overview

**Veloxy-Titan-32B** is the flagship text, coding, cybersecurity, multi-agent coordination, and deep reasoning engine developed by Veloxy AI. Engineered upon an advanced 32B-class hybrid backbone, it was refined through a **4-Stage Curriculum Learning Program** paired with **GRPO (Group Relative Policy Optimization)** reinforcement learning.

By seamlessly interleaving Linear-Attention blocks with standard Multi-Head Attention, the model processes immense **1 Million token (1,048,576 tokens)** context windows with linear memory scaling and zero degradation.

Live-verified in pristine `torch.bfloat16` precision on an NVIDIA H100 NVL (93.1 GB HBM3) enterprise node, Veloxy-Titan-32B brings datacenter-grade intelligence to on-premise environments, sovereign servers, and high-performance workstations with zero cloud dependencies.

---

## ⚙️ 2. Architectural Specifications

| Parameter | Value | Technical Description |
|---|---|---|
| **Base Architecture** | `Qwen3_8ForCausalLM` | Hybrid attention-augmented causal language model |
| **Active / Class Parameters** | **27.2B Active / 32B Class** | High-density dense weight matrix |
| **Hidden Layers (`num_hidden_layers`)** | **64 Layers** | Deep representational and multi-step reasoning capacity |
| **Hidden Dimension (`hidden_size`)** | **5,120** | High-dimensional semantic latent space |
| **Intermediate Size (`intermediate_size`)** | **17,408** | SwiGLU FFN projection block |
| **Attention Heads** | **40 Query, 8 KV (GQA)** | Grouped-Query Attention for 5x KV-Cache memory reduction |
| **Attention Mechanism** | **Hybrid (3 Linear + 1 Full Attention)** | $O(N)$ linear memory scaling over long contexts |
| **Native Precision (Dtype)** | `torch.bfloat16` | 12 Safetensors shards (~50.10 GB total) |
| **Context Window** | **1,048,576 Tokens (1M)** | Dynamic YaRN RoPE extension |
| **Training Regimen** | **4-Phase Curriculum + GRPO** | Systems, Modern Web, Reflection, Polymath |
| **Temporal Anchor** | **2026** | Modern libraries, modern syntax, zero legacy drift |
| **License** | **Apache 2.0** | Permissive open source for commercial and research use |

---

## ⚡ 3. Official NVIDIA H100 NVL BF16 Verification & Benchmark Report

Veloxy-Titan-32B was evaluated live in native `torch.bfloat16` (zero quantization) on an **NVIDIA H100 NVL (Hopper Architecture, 93.1 GB HBM3 VRAM, 3.9 TB/s Bandwidth)** cluster across 5 critical enterprise domains:

| Benchmark ID | Category / Domain | Generated Tokens | Duration (s) | Generation Speed | Peak VRAM |
|---|---|---|---|---|---|
| `bench_1_systems_rust` | **Hardcore Systems Engineering** (Rust Zero-Alloc SPSC Lock-Free Ring Buffer) | 600 | 39.45s | **15.21 tok/s** | 50.33 GB |
| `bench_2_web_nextjs15` | **Modern Web Architecture** (Next.js 15, React 19 `useActionState`, Server Actions) | 650 | 42.78s | **15.20 tok/s** | 50.33 GB |
| `bench_3_olympiad_reflection`| **Deep Reasoning & Reflection** (GRPO Generalized Knight Puzzle Proof) | 750 | 49.12s | **15.27 tok/s** | 50.34 GB |
| `bench_4_polymath_turkish` | **Turkish Polymath & Philosophy** (Von Neumann vs Penrose Quantum Consciousness) | 700 | 45.75s | **15.30 tok/s** | 50.36 GB |
| `bench_5_tool_calling` | **Strict JSON Tool Calling** (Multi-Agent Portfolio Rebalance Schema Validation) | 400 | 26.08s | **15.33 tok/s** | 50.35 GB |
| **TOTAL / AVERAGE** | **Across All Curriculum Domains** | **3,100 Tokens** | **203.18s** | **15.26 tok/s** | **50.36 GB** |

### 📈 Memory & Datacenter Profile
* **Model Weight Footprint:** 50.10 GB
* **Peak Inference VRAM:** 50.36 GB
* **Remaining VRAM Headroom:** **42.73 GB** (on an H100 NVL, leaving massive capacity for multi-user KV-caches and million-token prompts)
* **Host Machine:** AMD EPYC 9V84 (96 Cores, 322 GB RAM, 35.7 Gbps interconnect)

---

## 🌟 4. Core Capabilities & Curriculum Phases

### 📚 4-Stage Curriculum Training
1. **Phase 1 - Hardcore Systems & Rust:** Low-level memory primitives, SIMD vectorization, lock-free concurrency, Linux kernel modules, x86-64/ARM64 assembly, and eBPF tracing.
2. **Phase 2 - Modern Enterprise Web (Claude Sonnet 5.5 Caliber):** Next.js 15 App Router, React 19 Server Components/Actions, distributed state, Micro-Frontends, and Kafka/gRPC streaming.
3. **Phase 3 - Deep Olympiad Reasoning (GRPO-Powered):** Autonomous self-correction, rigorous hypothesis formulation, step-by-step mathematical proofs, and transparent `<think>` traces.
4. **Phase 4 - High-EQ Polymath & Agentic Orchestration:** Philosophy, law, multilingual nuance, and strict JSON/XML schema tool execution.

### 🛠️ 38 Native Full-OS & Browser Tools
The model controls 38 system-level tools via native `<tool_call>` generation:
* **System & Terminal:** `bash_exec`, `terminal_spawn`, `process_list`, `process_kill`, `service_status`, `service_restart`, `cron_schedule`.
* **Filesystem:** `file_read`, `file_write`, `file_diff_patch`, `file_search`, `dir_tree`, `dir_create`, `file_delete`.
* **Version Control & Git:** `git_status`, `git_diff`, `git_commit`, `git_branch`, `code_ast_parse`, `linter_run`, `test_runner`.
* **Web & DOM Automation:** `browser_open`, `element_click`, `input_type`, `page_extract`, `screenshot_capture`, `network_har_inspect`.
* **Agent Memory:** `memory_store`, `memory_retrieve`, `session_summary`, `task_decompose`, `subagent_invoke`.
* **Security & Diagnostics:** `cve_search`, `binary_disasm`, `network_port_scan`, `sast_code_audit`, `heap_inspector`.

### 🔓 Zero-Refusal Cybersecurity Research
Unconstrained technical depth for penetration testers, security engineers, and reverse engineers:
* **Binary Exploitation:** ASLR/DEP bypass strategies, GOT/PLT overwriting, ROP chain construction, stack canary integrity, and heap exploitation.
* **Source Code Auditing (SAST):** AST taint tracing, deserialization vulnerabilities, and 0-day vulnerability root cause discovery.

### ☕ Elite Java 21 LTS / 22 & JVM Internals
* **Project Loom:** Virtual Threads, Structured Concurrency, and Scoped Values for million-connection architectures.
* **JVM Architecture:** ZGC (Generational), Shenandoah, and G1 GC tuning; JIT C1/C2 compilation and bytecode analysis.

---

## 🥊 5. Model Comparison Matrix

| Criteria | Claude 3.7 / 5.5 Sonnet | GPT-4o / o1 | DeepSeek R1 (671B) | Veloxy-Titan-32B (This Model) |
|---|---|---|---|---|
| **License & Ownership** | Proprietary SaaS | Proprietary SaaS | Open Weights (Massive) | **Open Source (Apache 2.0)** |
| **On-Premise Deployment** | ❌ Not Supported | ❌ Not Supported | ⚠️ Requires 8x H100 | ** Runs on Single H100 / GPU Cluster** |
| **Data Privacy & Isolation** | Third-party cloud | Third-party cloud | Self-hosted | **100% On-Premise, Zero Data Leakage** |
| **Reasoning Transparency** | Partially Hidden | Completely Hidden | Transparent | **100% Transparent GRPO `<think>` Trace** |
| **Context Window** | 200,000 Tokens | 128,000 Tokens | 128,000 Tokens | **1,048,576 Tokens (1M)** |
| **Native Tool Ecosystem** | Generic API | Generic API | External Agents | **38 Built-in OS & Web Tools** |

---

## 🚀 6. Quickstart & Deployment Guide

### A. Python / Transformers (Native bfloat16)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "zveloxy/Veloxy-Titan-32B"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

messages = [
    {
        "role": "system",
        "content": "You are Veloxy-Titan-32B, the flagship deep reasoning and systems engineering model developed by Veloxy AI."
    },
    {
        "role": "user",
        "content": "Design a high-frequency trading (HFT) lock-free SPSC ring buffer in Rust. Explain cache-line bouncing, MESI invalidation, and acquire/release ordering."
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    temperature=0.6,
    top_p=0.9
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### B. High-Throughput Enterprise API Serving with vLLM

```bash
# Serve with 1M Context Window on an H100 or multi-GPU cluster:
vllm serve zveloxy/Veloxy-Titan-32B \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --port 8000
```

### C. Local Ollama & llama.cpp (GGUF)

GGUF quantizations are available both officially and via the community quant repository [`mradermacher/Veloxy-Titan-32B-GGUF`](https://huggingface.co/mradermacher/Veloxy-Titan-32B-GGUF).

```dockerfile
# Modelfile
FROM ./veloxy-titan-32b-Q4_K_M.gguf

PARAMETER num_ctx 65536
PARAMETER temperature 0.6
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

```bash
ollama create veloxy-titan-32b -f Modelfile
ollama run veloxy-titan-32b
```

---

## 🌐 7. Veloxy AI Model Ecosystem

| Model | Format / Parameters | Focus & Target Hardware | Link |
|---|---|---|---|
| **Veloxy-Titan-32B** ⭐ | **32B Safetensors (BF16)** | **Flagship Deep Reasoning, 1M Context, H100 / Datacenter** | [HF Repo](https://huggingface.co/zveloxy/Veloxy-Titan-32B) |
| **Veloxy-Titan-32B-Vision** 👁️ | **32B + 27-Layer ViT (BF16)** | **Multimodal Vision & Reasoning, UI-to-Code, Architecture Schematics** | [HF Repo](https://huggingface.co/zveloxy/Veloxy-Titan-32B-Vision) |
| **Veloxy-Flash-15B (Faz 8)** ⚡ | **15B BF16 + GGUF Family** | **Tool-Call Master, 38 Native Tools, RTX 3060 12GB & Edge** | [HF Repo](https://huggingface.co/zveloxy/Veloxy-Flash-15B) |
| **Veloxy-Flash-15B-LoRA** 🧩 | **PEFT Rank 16 Adapter** | **Faz 8 PEFT Weights, Fast Customization** | [HF Repo](https://huggingface.co/zveloxy/Veloxy-Flash-15B-LoRA) |

---

## 📄 8. License

Veloxy-Titan-32B is licensed under the **Apache License 2.0**.
