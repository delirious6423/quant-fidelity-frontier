---
library_name: gguf
tags:
- gguf
- quantized
- faithful-retrieval
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
---

# Qwen3.8-27B — 13GB (GGUF)

GGUF build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), capability-validated by [baa.ai](https://baa.ai) for **faithful, retrieval-grounded reading** — a full 27B reader that runs on a single 16 GB GPU (e.g. AWS `g4dn.xlarge` / NVIDIA T4).

Built for the case where the retrieved documents are the source of truth and the model's job is to *interpret* them — not to answer from its own memory. It was selected and validated to keep that behaviour intact under aggressive compression.

## Metrics

| Metric | Value |
|--------|-------|
| **Size** | **12.6 GB** |
| Average bits | 3.73 (IQ3_M) |
| Format | llama.cpp (GGUF) |
| Deference to retrieved evidence | **1.00** |
| Reading accuracy from context | **1.00** |
| Multi-hop reasoning (4–5 hop, long context) | 0.92–0.96 |

Validated to lose **nothing** on faithfulness or reading versus the full-precision model: when a retrieved passage contradicts the model's own prior, it follows the passage 100% of the time, and reads the correct answer from context 100% of the time. Fits a 16 GB GPU with room for an 8K-token context window.

## Usage

```bash
brew install llama.cpp huggingface-cli

hf download baa-ai/Qwen3.8-27B-RAM-13GB-GGUF --include "*.gguf" --local-dir ./qwen3.8-27b-ram-13gb

llama-cli -m ./qwen3.8-27b-ram-13gb/Qwen3.8-27B-RAM-13GB.gguf -p "Hello!" -n 256 -ngl 99
```

Or via `llama-server` for an OpenAI-compatible HTTP API (drop-in for a retrieval-augmented reader):

```bash
llama-server -m ./qwen3.8-27b-ram-13gb/Qwen3.8-27B-RAM-13GB.gguf --port 8080 -ngl 99 --ctx-size 8192
```

Also works with Ollama, LM Studio, and `llama-cpp-python`.

---

*Compressed and capability-validated by [baa.ai](https://baa.ai)*

---

## Black Sheep AI Products

**[Shepherd](https://baa.ai/shepherd.html)** — Private AI deployment platform that shrinks frontier models by 50-60% through RAM compression, enabling enterprises to run sophisticated AI on single GPU instances or Apple Silicon hardware. Deploy in your VPC with zero data leaving your infrastructure. Includes CI/CD pipeline integration, fleet deployment across Apple Silicon clusters, air-gapped and sovereign deployment support, and multi-format export (MLX, GGUF). Annual cloud costs from ~$2,700 — or run on a Mac Studio for electricity only.

**[Watchman](https://baa.ai/watchman.html)** — Capability audit and governance platform for compressed AI models. Know exactly what your quantized model can do before it goes live. Watchman predicts which capabilities survive compression in minutes — replacing weeks of benchmarking. Includes compliance-ready reporting for regulated industries, quality valley warnings for counterproductive memory allocations, instant regression diagnosis tracing issues to specific tensors, and 22 adversarial security probes scanning for injection, leakage, hallucination, and code vulnerabilities.

Learn more at **[baa.ai](https://baa.ai)** — Sovereign AI.
