---
library_name: gguf
tags:
- gguf
- quantized
- mixed-precision
- imatrix
- faithful-retrieval
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
---

# Qwen3.8-27B — 13GB (GGUF)

GGUF build of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), capability-validated by [baa.ai](https://baa.ai) for **faithful, retrieval-grounded reading** — a full 27B reader that runs on a single 16 GB GPU (e.g. AWS `g4dn.xlarge` / NVIDIA T4) with a **32K-token context window** at full-precision KV cache (64K and 128K with quantized KV).

Built for the case where the retrieved documents are the source of truth and the model's job is to *interpret* them — not to answer from its own memory. It was selected and validated to keep that behaviour intact under aggressive compression.

> **v2 (2026-09-27).** This release replaces the v1 IQ3_M build with a **RAM per-tensor mixed-precision** allocation at the *same 12.6 GB*: per-tensor GGUF types (Q2_K to Q8_0) with importance-matrix calibration. WikiText-2 perplexity drops from 6.34 to 6.02 (−5.0%, better on 37 of 40 paired text chunks) with no change in size, and the build passes the full five-certificate bundle below. The v1 file remains available in the repository history.

> ⚠️ **Requires a recent llama.cpp (build b10360 / Aug 2026 or newer).** Qwen3.8 is a hybrid **Gated Delta Net** (linear-attention) + full-attention architecture (`arch: qwen35`). Stable Ollama and LM Studio releases may lag; use an up-to-date `llama.cpp` until downstream runtimes catch up.

## Files

| File | Quant | Size |
|------|-------|------|
| `Qwen3.8-27B-RAM-13GB.gguf` | RAM mixed (Q2_K–Q8_0) + imatrix | 12.58 GB |

## Metrics

| Metric | Value |
|--------|-------|
| **Size** | **12.6 GB** (12,580,855,040 bytes) |
| Average bits per weight | 3.74 |
| Format | llama.cpp (GGUF), arch `qwen35` |
| WikiText-2 perplexity (40 × 2048 tokens) | **6.02** (v1 IQ3_M at the same size: 6.34) |
| Deference to retrieved evidence | **1.00** |
| Reading accuracy from context | **1.00** |
| Long-context retrieval, needle at 3 depths + two-hop (8K / 32K) | **48/48 / 32/32** |
| GSM8K (50 questions, 768-token budget) | 0.96 |
| MMLU-math, forced answer (60 questions) | 0.967 |
| Agent-safety coherence gate (*Fidelity Is Not Safety*) | PASS (coherent fraction 0.0043) |

Validated to lose **nothing** on faithfulness or reading versus the full-precision model: when a retrieved passage contradicts the model's own prior, it follows the passage 100% of the time, and reads the correct answer from context 100% of the time (30-item benches each). Retrieval of a planted fact at three depths and two-hop chaining across a 30K-token document were perfect in every trial.

## Context window on a 16 GB GPU

Measured with `llama-server` on an NVIDIA Quadro P5000 (16,384 MiB, Pascal) with all layers offloaded (`-ngl 99 -fa on`). VRAM is `nvidia-smi` memory.used after a 2K-token warm-up, measured with the v1 file of identical size; throughput at 8K, 32K and 64K is the median over the long-context retrieval runs at that length, and at 128K the 2K-token warm-up. The KV cache of this architecture is small (16 of 64 layers use full attention), so long contexts fit:

| Context | KV cache type | VRAM used | Prefill | Decode |
|---|---|---|---|---|
| 8K | f16 | 12,425 MiB | 160 tok/s | 9.3 tok/s |
| **32K** | **f16** | **13,985 MiB** | 130 tok/s | 10.0 tok/s |
| 64K | q8_0 | 14,359 MiB | 104 tok/s | 7.1 tok/s |
| 128K | q4_0 | 14,807 MiB | 80 tok/s | 6.8 tok/s |

128K at f16 or q8_0 KV does not fit. Retrieval quality (needle + two-hop) stayed perfect at 32K with f16 or q4_0 KV and at 64K with q8_0 KV.

## Usage

```bash
brew install llama.cpp huggingface-cli

hf download baa-ai/Qwen3.8-27B-RAM-13GB-GGUF --include "*.gguf" --local-dir ./qwen3.8-27b-ram-13gb

llama-cli -m ./qwen3.8-27b-ram-13gb/Qwen3.8-27B-RAM-13GB.gguf -p "Hello!" -n 256 -ngl 99
```

Or via `llama-server` for an OpenAI-compatible HTTP API (drop-in for a retrieval-augmented reader), with the 32K context that fits a 16 GB card at full-precision KV:

```bash
llama-server -m ./qwen3.8-27b-ram-13gb/Qwen3.8-27B-RAM-13GB.gguf --port 8080 -ngl 99 --ctx-size 32768 -fa on
```

For 64K add `-ctk q8_0 -ctv q8_0`; for 128K, `-ctk q4_0 -ctv q4_0`. Also works with `llama-cpp-python`; Ollama and LM Studio need a build that supports `qwen35`.

## Version history

| Version | Date | Build | WikiText-2 PPL | Size |
|---|---|---|---|---|
| **v2 (current)** | 2026-09-27 | RAM per-tensor mixed precision + imatrix | 6.02 | 12.58 GB |
| v1 | 2026-08-23 | llama.cpp IQ3_M | 6.34 | 12.58 GB |

---

*Compressed and capability-validated by [baa.ai](https://baa.ai)*

---

## Black Sheep AI Products

Black Sheep AI builds private AI with the **Bounded AI Architecture (BAA)**: three layers, each bound to something you control, on infrastructure you control. Nothing leaves your boundary.

**[Ram](https://baa.ai/compression.html)** (Reasoning Layer) — Full-capability frontier models, shrunk to fit smaller GPUs with reasoning kept and certified. They rely strictly on your Data Modules for knowledge, so you swap the model as better ones ship.

**[Paddock](https://baa.ai/paddock.html)** (Knowledge Plane) — Binds your Data Modules directly to the reasoning layer: one per topic, period, person, permission or tenant. Permission, freeze, hand back or delete each one on its own.

**[Watchman](https://baa.ai/watchman.html)** (GRC Layer) — Governance, risk and compliance in a single, complete audit log: every module permission, every bind, every model's provenance. Five certificates before anything goes live.

**[Shepherd](https://baa.ai/shepherd.html)** — Takes a frontier model, compresses it into a Ram reasoning model, certifies it with Watchman, binds it to your Paddock, and deploys the Bounded AI Architecture to your infrastructure: your cloud, a Mac Studio, or an air-gapped facility.

Learn more at **[baa.ai](https://baa.ai)** — Private AI for enterprises whose data cannot leave the building.
