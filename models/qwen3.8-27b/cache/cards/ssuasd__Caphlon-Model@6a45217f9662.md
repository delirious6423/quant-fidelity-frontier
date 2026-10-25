---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
library_name: llama.cpp
language:
  - en
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - qwen3
  - qwen
  - 27b
  - iq2_s
  - quantized
  - lightweight
  - local-llm
  - text-generation
  - hybrid-attention
  - layer-pruning
---

# Caphlon-Model

**A 6.87 GiB GGUF build of Qwen 3.8 27B — full 27B reasoning quality, laptop-sized footprint.**

Built and benchmarked on **25 October 2026**. No distillation, no student model: the
27B topology survives. Footprint was cut by removing representation redundancy —
vocabulary merge-table reduction and hybrid-interval-aware layer pruning — on top of an
Unsloth Dynamic IQ2_S base.

## Model details

| | |
|---|---|
| **Base model** | [Qwen 3.8 27B](https://huggingface.co/Qwen) (Apache-2.0) |
| **Intermediate** | Unsloth Dynamic IQ2_S |
| **Architecture** | 60-layer hybrid: gated-delta-net (majority) + full-attention (minority) |
| **File** | `Caphlon-Model-IQ2_S.gguf` — **6.87 GiB** (7,375,122,272 bytes) |
| **SHA-256** | `8487b4e443834659b230e0bd529674ad65e068fc00380b996aaf11c873d63bc0` |
| **Format** | GGUF (llama.cpp) |
| **Context** | up to 32K tested |
| **Language** | English |

KV cache lives only on the full-attention layers; gated-delta-net layers carry
fixed-size recurrent state, so memory stays small **and** context scales cheaply.

## Changes vs. the base quant

| Step | Change | Saved |
|---|---|---|
| 1 | Vocabulary merge table reduced to a 100k target (closure-preserving selection) | −649 MiB |
| 2 | Layers 12–15 removed (4 layers, aligned to the hybrid interval) | −302 MiB |
| 3 | Weights kept at IQ2_S — no further quantization damage | — |

## Benchmarks

Measured 2026-10-25, Tesla T4 (16 GB), llama.cpp, temperature 0, thinking disabled.

| Test | Caphlon-Model | Base quant | Result |
|---|---|---|---|
| English perplexity (`wiki.test.raw`, 512 ctx × 40 chunks) | **6.6088** | 6.3638 | +3.85% |
| HumanEval, 30 problems, pass@1 | **29/30** | 30/30 | **96.7%** |
| MMLU, 502-question subset | **344/502** | 370/502 | **92.9% retained** |
| File-edit throughput (`--spec-type ngram-mod`) | **29.7 tok/s** | 12.7 tok/s | 2.34× |
| Free-form prose throughput | 12.4 tok/s | 12.7 tok/s | — |

## Hardware requirements

| Setup | Minimum | Comfortable |
|---|---|---|
| GPU, full offload | 8 GB VRAM (8K ctx) | 12 GB VRAM (32K ctx) |
| CPU only | 8 GB RAM (mmap) | 16 GB RAM |
| Apple Silicon | 16 GB unified | 24 GB+ |
| Disk | 7 GB | — |

## Usage

```bash
llama-server -m Caphlon-Model-IQ2_S.gguf \
  -ngl 99 -c 32768 -np 1 \
  -b 4096 -ub 1024 --flash-attn on \
  --spec-type ngram-mod --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 48 --spec-ngram-mod-n-max 64 \
  --jinja --port 8080
```

Send `chat_template_kwargs: {"enable_thinking": false}` on chat requests, otherwise
the model spends the token budget on internal reasoning.

Optional, only when context-bound: `--cache-type-k q8_0 --cache-type-v q4_0`
(measured gain −152 MB, but greedy output is then no longer byte-identical).

## License and attribution

Weights are a derivative of **Qwen 3.8 27B** (Apache-2.0); the upstream license governs
them and permits commercial use with notices retained. Documentation, naming, and
evaluation scripts by the author are Apache-2.0.

"Caphlon" is a project mark. Apache-2.0 grants no trademark rights: you may redistribute
the weights, but you may not present a modified build as official, endorsed, or authored
by this project.

## Citation

```bibtex
@misc{caphlon2026,
  title  = {Caphlon-Model: A 6.87 GiB lightweight build of Qwen 3.8 27B},
  author = {demiralpdev},
  year   = {2026},
  month  = {10},
  note   = {GGUF, IQ2\_S base, vocabulary-reduced and layer-pruned},
  url    = {https://github.com/demiralpdev/Caphlon-Model}
}
```

- **GitHub:** https://github.com/demiralpdev/Caphlon-Model
- **Project page:** https://demiralpdev.github.io/Caphlon-Model/
