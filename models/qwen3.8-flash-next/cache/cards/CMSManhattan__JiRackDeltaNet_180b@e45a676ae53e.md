---
language:
- en
- zh
- ja
- ko
- fr
- es
- pt
- de
- it
- ru
- ar
- vi
- th
tags:
- text-generation
- moe
- mixture-of-experts
- deltanet
- sparse-attention
- hyper-connections
- qwen3.8
- qwen3.8-flash-next
- gguf
- cpu
- efficient
- ternary
- bitnet
- jirack
- routing
- tool-call
- robotics
- ternary-weights
- ternary-transformer
- quantization
- qat
base_model: Qwen/Qwen3.8-Flash-Next
pipeline_tag: text-generation
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
---

# Qwen3.8-Flash-Next 180B migrated to the JiRack Ternary Architecture
- A 180B-parameter Mixture-of-Experts model with only **6B parameters active per token**, which makes high-quality CPU inference realistic at this scale
- Migrated to the JiRack ternary architecture (BitLinear, per-tensor absmean ternary weights) with a lambda-warmup QAT pipeline, targeting TQ2_0 CPU inference on llama.cpp and Ollama
- Robotics, Routing, Coding and Advanced tool calling via the JiRack tool-call workflow
- The Qwen3.8-Flash-Next base model understands images and video; the JiRack build is currently text-only (see [Current status](#current-status))

## PARTNERSHIP
- NVIDIA
- FISERV

# JiRack DeltaNet 180B (CPU)
A 180B MoE model built on the Qwen3.8-Flash-Next architecture: Gated DeltaNet linear attention interleaved with Qwen Sparse Attention (QSA), 512 experts with top-10 routing plus a shared expert, gated residual (hyper-connection) streams, and a 51B-parameter hashed n-gram embedding table. Only ~6B parameters are computed per token, so throughput on CPU is closer to a 6B dense model while quality is that of a much larger one.
- JiRack is a cloud-ready model that helps save money on cloud infrastructure. It can be used as an expert model in RAG deployments, with the ONNX JiRack Java server as an alternative.

## Current status

| Stage | Status |
|---|---|
| Hand-rolled JiRack port of the architecture (`JiRackDeltaNet_180b.py`) | ✅ done |
| Conversion from Qwen/Qwen3.8-Flash-Next and numerical verification against the reference | ✅ done — next-token prediction matches the reference on 100% of positions; logit differences are within the model's own bf16 noise floor |
| BitLinear patch (74,028 layers) at lambda = 0 | ✅ done — bit-exact passthrough (max logit difference 0.000000) |
| Export to Hugging Face layout, llama.cpp GGUF (BF16) | ✅ export verified key-for-key against the original; GGUF conversion in progress |
| Ternary QAT (lambda warmup 0 → 1) | ⏳ not started — requires a machine with ≥ 640 GB RAM |
| TQ2_0 / Q-quantized GGUF, Ollama and Docker builds | ⏳ not yet published |

**Important:** until QAT is complete, the JiRack 180B weights are numerically identical to the base model (lambda = 0 means BitLinear is a pure passthrough). The ternary benefits described on this card apply after QAT.

# JiRack service options
- If you need custom compression or fine-tuning, please write to me and I'll perform QAT from your dataset, tailored specifically to your task.
- Plus double QAT via ONNX QAT.
- Adapt train process to avoid catastrophic forgetting with NDA
- Adapt train process to avoid fast plateau in training with NDA
- MoE-aware QAT: routers are never quantized and stay frozen by default, with built-in router health monitoring (routing agreement, entropy collapse, expert load) during training
- Adapts to agentic or instruct models for tool calling, using the JiRack tokenizer to enable high-quality tool calling — built as a domain-specific tool expert.
- Deployment and scale

---

## JiRack Codding Agent IDE
- It is Agent Coding IDE for JiRack Models to run via Ollama on home PC 
- It good choose for Agent Coding IDE such as Cursor , Windsurf IDE or Devin IDE etc but more safe that ask you to apply changes and review.
- Web site https://www.jirack.com
- Eclise Plugin site https://marketplace.eclipse.org/content/jirack-coding-agent
- Jetbrain Plugin site https://plugins.jetbrains.com/plugin/35073-jirack-agent
- VS Code Plugin site https://marketplace.visualstudio.com/items?itemName=JiRackautomationCodingAgentIDE.jirack-agent
  
## JiRack Codding Agent IDE files
- Eclise Final release version https://huggingface.co/CMSManhattan/JiRackDeltaNet_27b/resolve/main/jirack_ide_final.zip
- Microsoft VS Code plugin https://huggingface.co/spaces/CMSManhattan/README/blob/main/jirack-agent-0.9.0.vsix
- Jetbrain plugin  https://huggingface.co/spaces/CMSManhattan/README/resolve/main/jirack-jetbrains-0.11.0.zip  
- JiRack DeskTop UI  https://huggingface.co/spaces/CMSManhattan/README/resolve/main/windows_v11.zip
  
---

# Ollama support
- 180B Ollama builds are planned after QAT. Ollama needs a llama.cpp version that supports the `qwen4exp` architecture.
- Qwen3.8-Flash-Next thinks by default (`<think>...</think>`); JiRack builds will ship with reasoning disabled by default, as for JiRack 27B.
- Follow https://ollama.com/cmsmanhattan for releases.

# Spring Boot AI tool calls examples for JiRack DeltaNet series
- Tool call library on java for Enterprise  https://github.com/alibaba/spring-ai-alibaba

# GoEx AI tool calls examples for JiRack DeltaNet series
- Tool call library on python  https://github.com/ShishirPatil/gorilla

# JiRack DeltaNet tool calls to boost tool call quality
- Use JiRack Precision tokenizer tags for tool calls with ToolBench https://github.com/OpenBMB/ToolBench
- https://huggingface.co/xalss/Qwen2-7B-Instruct-glaive-function-calling
- https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1
- Add JiRack tool call tags in the dataset and modify tool call processor if needed

# JiRack RoboTech
- Advanced Tokenizer with Robotics & Routing & Tool calls Tokenizer and other
- [CMSManhattan/JiRackDeltaNetTokenizer](https://huggingface.co/CMSManhattan/JiRackDeltaNetTokenizer)
- The 180B build currently uses the original Qwen3.8 tokenizer (vocabulary 248,320).

## Planned GGUF variants

Not yet published. Sizes are estimates from the parameter count (~180B stored); actual files will differ.

| Quant | Est. size | Est. RAM | Description |
|-------|-----------|----------|-------------|
| BF16 | ~360 GB | 384 GB+ | Full precision reference (identical to the base model before QAT) |
| Q8_0 | ~190 GB | ~200–256 GB | Near-lossless |
| Q4_K_M | ~110 GB | ~128 GB | Recommended balance |
| Q3_K_M | ~88 GB | ~96–128 GB | Good quality / size trade-off |
| Q2_K | ~75 GB | ~96 GB | Maximum compression without QAT |
| TQ2_0 (after QAT) | ~65–95 GB | ~96–128 GB | Ternary experts and attention; the size range depends on how the 51B n-gram table is stored |

## Quick Start

## Live  chat with other models 

- https://huggingface.co/spaces/CMSManhattan/JiRack-16B-Chat
- https://huggingface.co/spaces/CMSManhattan/JiRack-8B-Chat
- https://huggingface.co/spaces/CMSManhattan/JiRack-2B-Chat

### Run the JiRack checkpoint directly (PyTorch, CPU)
The JiRack checkpoint is a directory of safetensors shards that is memory-mapped lazily, so it runs even when the model is larger than RAM (slower: pages are read from disk as experts are used).
```bash
python test_jirack_180b_generate.py   # quick generation smoke test
python chat_jirack_180b.py            # interactive chat
```

### Build a GGUF
```bash
python export_180b_to_hf_safetensors.py \
    --checkpoint /data/qwen38_180b_checkpoint_safetensors \
    --out_dir /data/qwen38_180b_hf_export
python convert_hf_to_gguf.py /data/qwen38_180b_hf_export \
    --outfile JiRackDeltaNet_180b.gguf --outtype bf16
```
After QAT, add `--bake_ternary` to the export to write the ternary weights.

### Docker
- 180B Docker images with the JiRack UI will be provided after QAT, or by request.

### Ollama 
- ollama run cmsmanhattan/JiRackDeltaNet_180b-q4
- ollama run cmsmanhattan/JiRackDeltaNet_180b-q3

## Ollama quick Start

### Run with Ollama (recommended — reasoning off by default)
- https://ollama.com/cmsmanhattan
```bash
ollama pull JiRackDeltaNet_180b-q4
ollama run JiRackDeltaNet_180b-q4 "What is the capital of France?"
```


## Recommended sampling parameters
From the base model card:
- **Thinking mode:** `temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0`
- **Instruct (non-thinking) mode:** `temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0`

## Access the UI
Once a JiRack container is running, open your browser at `http://localhost:7869`. This opens the JiRack UI — a clean web interface. The port can be changed from the **Settings** panel.

## Licensing

This repository is dual-licensed: the model weights and the JiRack code are licensed separately.

| Component | License |
|---|---|
| **Model weights** (safetensors, GGUF) — a derivative of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | **Qwen Community License 1.0** — [LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE), Copyright 2026 Qwen |
| **JiRack code and tools** (`JiRackDeltaNet_180b.py`, conversion, QAT training, export and test scripts) | **MIT License** — © 2025-2026 Konstantin Vladimirovich Grabko, CMS Manhattan JiRack Technology |
| **JiRack Docker images with UI, pre-built Ollama quantizations, JiRack UI clients** | Commercial license (see below) |

- **Weights:** the Qwen Community License continues to apply to the weights, including the JiRack-converted and quantized versions. Keep the Qwen copyright and license notice with all copies. Commercial Model-as-a-Service and AI work-assistant (coding / office) products built on these weights require a separate license from Qwen — review the license terms before such use.
- **Code:** the MIT License covers the JiRack source code only. It does not change the license of the weights.
- The Docker image with UI and pre-built Ollama quantizations are separate commercial products.
- All JiRack UI clients are provided under a commercial license. The UI clients can be used for free together with the official JiRack Docker containers, as long as they are not redistributed separately.

For commercial licensing, cluster deployment, or enterprise use of JiRack models, please contact us.
- **JiRack MS Windows 11 Desktop Client (with Ollama API):**
  https://huggingface.co/kgrabko/JiRackTernary_1b/resolve/main/jirack-chat.zip
- **Live email chat with the model:** support@cmsmanhattan.com

## Hardware Recommendations

### Recommended hardware for JiRack DeltaNet 180B
| Use Case | CPU | RAM | Recommended Quant | Notes |
|----------|-----|-----|-------------------|-------|
| Recommended | High-core server CPU (Xeon / EPYC) | 128 GB | Q4_K_M, or TQ2_0 after QAT | Only ~6B active parameters per token |
| High Performance | Dual-socket server | 256–384 GB | Q8_0 / BF16 | Reference quality |
| Low Memory | Modern 16+ core CPU + fast NVMe | 96 GB | Q2_K / Q3_K_M | Usable |
| PyTorch checkpoint | Server CPU + NVMe | 256 GB+ | BF16 (lazy mmap) | Works below model size, slower |
| QAT training | Server CPU | ≥ 640 GB (768 GB recommended) | — | Weights + gradients of ~125B trainable parameters |

## Important Memory Notes
- The whole model must be resident (or memory-mapped) even though only ~6B parameters are computed per token: the router can pick any of the 512 experts in every layer.
- 36 of the 48 layers use Gated DeltaNet with a fixed-size recurrent state, so long contexts need far less KV cache than a full-attention model of this size.
- Leave headroom for runtime buffers and the attention KV cache of the 12 full-attention layers.

## Architecture Notes
- **Qwen3.8-Flash-Next architecture** (`qwen4_exp` in Hugging Face, `qwen4exp` in llama.cpp)
- **Parameters:** 125B core + 51B n-gram embedding + 4B MTP (≈180B stored), **6B activated** per token
- **Layers:** 48 = 12 × (3 × Gated DeltaNet → MoE, 1 × Gated Attention with QSA → MoE)
- **Hidden size** 2560, carried as **4 parallel hyper-connection streams** (gated residual replaces the usual pre/post norms)
- **MoE:** 512 experts (intermediate 640), top-10 routing, plus 1 shared expert with a sigmoid gate
- **Gated DeltaNet:** 16 key heads × 128, 48 value heads × 128, conv kernel 4, sigmoid output gate
- **Gated Attention:** 24 query heads, 2 KV heads, head dim 256, partial rotary 0.25, interleaved mRoPE (11/11/10), θ = 10,000,000
- **QSA (Qwen Sparse Attention):** keys pooled in blocks of 4 tokens; each query attends to the top 2048 tokens' blocks plus the local tail
- **PLE n-gram embedding** on layer 2: hashed bi- and trigrams, 16 heads, 320M-row table (51B parameters) in 128 shards
- **MTP head:** 1 layer for speculative decoding
- RMSNorm ε = 1e-6, vocabulary 248,320, context 262,144 tokens natively (extensible to 1M in the base model)

### JiRack ternary scheme
- **Ternarized (BitLinear):** DeltaNet `in_proj_qkv` / `in_proj_z` / `out_proj`, attention `q/k/v/o_proj`, all routed experts and the shared expert — 74,028 layers
- **Always full precision:** MoE routers and the shared-expert gate (never quantized, frozen by default during QAT), hyper-connections, all norms, conv1d, `A_log` / `dt_bias`, QSA indexer, PLE projections and n-gram table, embeddings, `lm_head`, MTP head
- **Weights:** per-tensor absmean ternary `{-γ, 0, +γ}`; **activations** during training: per-token int8
- **QAT:** continuous lambda warmup from full precision (λ = 0) to fully ternary (λ = 1) with a straight-through estimator

## Benchmarks
JiRack DeltaNet 180B is built on Qwen3.8-Flash-Next. The tables below reproduce the published base-model results from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for reference — they reflect the upstream base model's capabilities, not JiRack-specific QAT or quantization results.

### Language

|                                                          | Qwen3.8-Flash-Next | Qwen3.8-27B | Qwen3.7-Plus | DeepSeek-V4-Flash-0731 | Claude-Opus-4.6 (Max) |
|----------------------------------------------------------|--------------------|-------------|--------------|------------------------|-----------------------|
| # Params                                                 | 125B               | 27B         | 397B         | 284B                   | --                    |
| # Activated params                                       | 6B                 | 27B         | 17B          | 13B                    | --                    |
| # N-gram embedding params                                | 51B                | --          | --           | --                     | --                    |
| **Coding** | | | | | |
| Agentic coding — DeepSWE 1.1                             | **58.7**           | 42.2        | 16.5         | 54.4                   | --                    |
| Agentic coding — SWE-bench Pro                           | **62.5**           | 61.7        | 55.8         | 56.0                   | 53.4                  |
| Multilingual software engineering — SWE-bench Multilingual | **81.0**         | 73.8        | 75.8         | --                     | 77.5                  |
| Repo-level code generation — NL2Repo-Bench               | 48.1               | 42.3        | 41.1         | **54.2**               | 47.6                  |
| **Agent** | | | | | |
| Long-horizon office work — CoWorkBench                   | **73.9**           | 70.7        | 65.1         | 45.1                   | 68.2                  |
| Professional job tasks — JobBench                        | **55.7**           | 33.4        | 27.6         | 41.3                   | 36.6                  |
| Frontier agentic tasks — Agents' Last Exam (Pass@1 / Score) | 24.3 / **51.2** | 20.4 / 42.9 | 13.2 / 33.6  | **25.2** / --          | --                    |
| Real-world tool use — Toolathlon Verified (Pass@1)       | **73.5**           | 67.1        | 50.6         | 70.3                   | --                    |
| **General** | | | | | |
| Instruction following — IFBench                          | **81.3**           | 79.5        | 79.1         | 79.2                   | 62.5                  |
| Scientific reasoning — GPQA Diamond                      | **91.7**           | 89.2        | 90.3         | 90.8                   | 91.3                  |
| Multidisciplinary reasoning — HLE                        | 35.9               | 30.8        | 34.7         | 33.8                   | **40.0**              |
| Competitive coding — LiveCodeBench v6                    | **91.9**           | 90.3        | 89.6         | 90.6                   | 88.8                  |

### Vision-Language (base model)

|                                                          | Qwen3.8-Flash-Next | Qwen3.8-27B | Qwen3.7-Plus | Claude-Opus-4.6 (Max) |
|----------------------------------------------------------|--------------------|-------------|--------------|-----------------------|
| **Agentic Multimodal Intelligence** | | | | |
| Multimodal tool use — ClawEval-MM (Pass@3 / Average)     | **64.4 / 60.4**    | 57.4 / 56.9 | 57.4 / 60.1  | 52.5 / 54.7           |
| Application recreation — RecreationBench                 | **49.9**           | 47.1        | 30.2         | --                    |
| Mobile use — AndroidWorld                                | **84.5**           | 81.9        | 81.0         | 62.0                  |
| Computer use — OSWorld 2.0 (Binary / Partial)            | **19.4 / 52.3**    | 19.4 / 48.0 | 2.8 / 21.5   | --                    |
| Visual web development — Vision2Web                      | **64.0**           | 62.9        | 42.1         | --                    |
| **General Multimodal Intelligence** | | | | |
| Embodied intelligence — ERQA                             | **72.3**           | 65.5        | 69.8         | 40.8                  |
| Long video understanding — LVBench                       | **76.6**           | 72.4        | 76.2         | 63.0                  |
| Real-world perception — RealWorldQA                      | **88.5**           | 85.9        | 86.9         | 73.9                  |
| Visual math — MathVision (w/o CI / w/ CI)                | **90.6 / 95.7**    | 90.0 / 94.6 | 90.3 / 88.7  | 65.5 / --             |
| Scientific chart analysis — CharXiv (RQ) (w/o CI / w/ CI) | 84.6 / **90.6**   | 83.7 / 90.2 | **85.8** / 85.9 | 66.0 / --          |

*Source: [Qwen/Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Best result in each row is bolded. Empty cells (--) indicate results not available or not applicable. See the source card for evaluation harnesses, settings and footnotes.*

## Citation
```bibtex
@techreport{qwen2026design,
    title       = {On the Design of {Qwen3.8-Next} Architecture: Evaluation, Efficiency, and Training Stability},
    author      = {{Qwen Team}},
    institution = {Alibaba Group},
    month       = {August},
    year        = {2026}
}
```

## 📧 Contact & Licensing
For joint venture opportunities, hardware integration, or licensing inquiries:
- **Email:** grabko@cmsmanhattan.com
- **Phone:** +1 (516) 777-0945
- **Location:** New York, USA

## License
- **Model weights:** Qwen Community License 1.0 — see [LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). Copyright 2026 Qwen.
- **JiRack code and tools:** MIT License — see `LICENSE-CODE`. © 2025-2026 Konstantin Vladimirovich Grabko, CMS Manhattan JiRack Technology.


![NVIDIA Inception Program Member](https://huggingface.co/spaces/CMSManhattan/README/resolve/main/badge.jpg)
