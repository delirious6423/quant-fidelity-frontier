---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
---

# Tileward-Qwen3.8-27B-TWX4

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), compressed by
[Tileward](https://tileward.com) to TWX4: 31.0 GB, and level with the uncompressed model in an
agent loop, finishing 43.7% of AgentBench OS's Linux tasks against 45.1%.

![Tasks finished on AgentBench OS for four builds of Qwen3.8-27B](results.png)

| | Tasks finished, two runs of 142 | Share |
|---|---|---|
| Qwen3.8-27B, uncompressed | 64 + 64 | 45.1% |
| Qwen's own FP8 build | 60 + 65 | 44.0% |
| **Tileward-Qwen3.8-27B-TWX4** | **61 + 63** | **43.7%** |
| Bonsai 2 27B, a ternary build | 57 + 64 | 42.6% |

Each task is a Linux job in a real container, passed or failed by the environment; every build had
the same tasks and settings on one 80 GB card. The ternary build is the smallest file, 7.2 GB
against 31.0 GB, and used 28% more prompt tokens than the uncompressed model to get there; this
build used 7% more.

## Research

- [How each figure was measured, on tileward.com/evals](https://tileward.com/evals/#compressed-model-agents)
- [We made Qwen3.8-27B 8.5% smaller than NVIDIA's NVFP4 build, and beyond](https://tileward.com/research/a-smaller-qwen3-8-27b/)
- [Smaller models should keep the behavior you chose](https://tileward.com/research/what-compression-inherits/)
- [All of Tileward's research](https://tileward.com/research/)

## Availability

Weights are not published here. To talk about this build,
[book 30 minutes with an engineer](https://calendar.app.google/hg2DzFTxKJgwFWm19).

Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen team,
Apache-2.0.
