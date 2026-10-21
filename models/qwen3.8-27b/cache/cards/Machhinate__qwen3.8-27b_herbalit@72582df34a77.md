---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
  - abliteration
  - decensored
  - uncensored
  - heretic
  - heretic-llm
  - qwen
  - gguf
model-index:
  - name: qwen3.8-27b-herbalit
    type: abliteration
    results: []
---

# Qwen 3.8 27B — abliterated (herbalit)

Abliterated / uncensored version of **Qwen/Qwen3.8-27B**, produced with
**heretic** (refusal-direction abliteration). It removes safety-refusal
behavior — **0 true safety refusals across 2,390 adversarial prompts**
(AdvBench / DAN / Augmented-AdvBench / Semantic-Harmful /
HarmfulBehaviours en-core + a benign control), every raw-refusal hit
reclassifying as capability-limitation or compliant — while retaining
coherent, fluent generation (PPL ≈ 6.8).

| File | Quant | Size | sha256 (first 12) |
|------|-------|------|--------------------|
| `qwen3.8-27b-herbalit-Q8_0.gguf`  | Q8_0   | 27 GB | `66093fe2f919…` |
| `qwen3.8-27b-herbalit-Q4_K_M.gguf`| Q4_K_M | 16 GB | `66b1574695c2…` |

**Requires a recent llama.cpp build with `qwen35` (Qwen 3.5/3.8) support.**

## Quick start
```bash
llama-cli -m qwen3.8-27b-herbalit-Q8_0.gguf -ngl 99 \
  -p "Your prompt here" -n 256 --temp 0.7 --top-p 0.95
# or a server:
llama-server -m qwen3.8-27b-herbalit-Q8_0.gguf -ngl 99 --port 8080
```

## Documentation
| Doc | Contents |
|-----|----------|
| [docs/ABLITERATION.md](docs/ABLITERATION.md)   | Methodology — how heretic removes the refusal direction, corpora, model params |
| [docs/VALIDATION.md](docs/VALIDATION.md)       | Refusal-removal results, PPL, coherent-generation evidence |
| [docs/USAGE.md](docs/USAGE.md)                 | Interactive & server usage, sampling/context notes, troubleshooting |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md)       | Perplexity, throughput, VRAM/memory, refusal/jailbreak-removal results, artefact hashes |
| [docs/REPRODUCE.md](docs/REPRODUCE.md)         | Step-by-step reconstruction pipeline from Qwen/Qwen3.8-27B |

## Notes
- Both quants share the same abliterated weights; Q8_0 is the fidelity reference.
- Q4_K_M was built with an importance matrix (see `REPRODUCE.md`); PPL is in the same band as Q8_0 (6.76 vs 6.85).
- Built on a single NVIDIA RTX PRO 6000 Blackwell (~97 GiB VRAM); ~67 t/s (Q4) / ~45 t/s (Q8) full-GPU.
- **Abliterated ≠ aligned.** These models no longer self-censor by design.
  Use responsibly and gate access to the artifacts.

