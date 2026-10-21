---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
pipeline_tag: text-generation
tags:
- gguf
- quantized
- llama.cpp
---

# Parakon-27B

A compact, high-fidelity **GGUF** build for local inference, derived from
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). It loads in any current
[llama.cpp](https://github.com/ggml-org/llama.cpp) build and the tools on top of it
(LM Studio, Ollama, etc.) — no custom kernels.

## File

| File | Size | Quality vs Q8 (median KLD) | Perplexity |
|---|---|---|---|
| `Parakon-27B.gguf` | 13.0 GB | **0.014** | −2.2% |

Measured against a full-precision `Q8_0` reference: very low drift at a fraction of the
footprint. A median KLD under ~0.05 indicates minimal quality loss; above ~0.08 is real
degradation. These are relative-to-Q8 figures, not absolute task-benchmark scores.

## Usage

```bash
llama-cli -m Parakon-27B.gguf -p "Your prompt here"
```

## License

Derived from **Qwen/Qwen3.8-27B** and redistributed under its **Apache-2.0** terms.
