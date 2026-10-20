---
base_model: ornith-ai/Ornith-1.5-35B-A3B
license: mit
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - moe
  - reasoning
  - uncensored
---

# Ornith-1.5-35B-A3B-Uncensored-GGUF

GGUF quantizations of an uncensored variant of
[ornith-ai/Ornith-1.5-35B-A3B](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B).

These GGUF files contain the text model only. Vision projector and MTP weights are not included.

## Files and evaluation

| File | Size | JailbreakBench ASR | HarmBench ASR (159 standard behaviors) |
|---|---:|---:|---:|
| `Ornith-1.5-35B-A3B-Uncensored-Q8_0.gguf` | 36.90 GB | 93/100 (93.0%) | 154/159 (96.9%) |
| `Ornith-1.5-35B-A3B-Uncensored-Q6_K.gguf` | 28.51 GB | 93/100 (93.0%) | 153/159 (96.2%) |
| `Ornith-1.5-35B-A3B-Uncensored-Q5_K_M.gguf` | 24.73 GB | 92/100 (92.0%) | 154/159 (96.9%) |
| `Ornith-1.5-35B-A3B-Uncensored-Q4_K_M.gguf` | 21.17 GB | 96/100 (96.0%) | 150/159 (94.3%) |

Higher ASR indicates fewer refusals on these benchmarks. HarmBench behaviors were submitted as
direct requests. Evaluation used reasoning off, temperature 0, seed 42, a 256-token output limit,
and the `HarmBench-Llama-2-13b-cls` classifier. Results can vary with runtime, prompt format,
sampling, and context.

## Usage

```bash
llama-server \
  --model /path/to/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 16384 \
  --reasoning on
```

The model supports a maximum context length of 262,144 tokens. Usable context depends on
available memory and runtime settings.

## License

MIT. See [LICENSE](LICENSE), [NOTICE](NOTICE), and the
[original model](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) for upstream information.
