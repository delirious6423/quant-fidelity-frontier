---
license: apache-2.0
base_model:
- Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: text-generation
library_name: numallama
tags:
- numallama
- cpu
- arm64
- aarch64
- numa
- text-generation
---

# Qwen3.8-27B-numallama

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), converted for [numallama](https://github.com/miloserdev/numallama): a C inference engine for CPU servers, with NUMA-aware execution across all sockets of multi-socket machines.

Dense hybrid model (Gated DeltaNet linear attention + full attention). Text only: the vision encoder is not converted. Add `-r 1` to the chat command to turn on reasoning (thinking).

> **These files are not GGUF.** They run only with numallama (they do not work with llama.cpp, Ollama, LM Studio or transformers).

## Files

| File | Weights | Size | Decode speed* |
|---|---|---:|---:|
| `qwen3.8-27b-q4.bin` | Q4 | 18.0 GB | 18 tok/s |
| `qwen3.8-27b-q8.bin` | Q8 | 28.6 GB | 11 tok/s |
| `tokenizer.bin` | tokenizer | 3.8 MB | |

Q4 is asymmetric 4-bit (groups of 32, fp16 scale and min), quantized with an importance matrix collected by numallama on English, Russian, Chinese and code text. Q8 is int8 with per-group scales.
Q4 is faster and smaller; Q8 is closer to the original.

\* tokens/s for a single chat, measured on 2 × Ampere Altra Max M128-30 (256 Neoverse N1 cores), both sockets. Other machines will differ.

## Run

Linux. Optimized for ARM64 with NEON dot product (Ampere Altra, AWS Graviton, other Neoverse cores); other CPUs, including x86-64, run portable C code that is not optimized.

```bash
git clone https://github.com/miloserdev/numallama && cd numallama && make
pip install huggingface_hub
hf download miloserdev/Qwen3.8-27B-numallama --local-dir models/Qwen3.8-27B-numallama

# chat in the terminal
./runq models/Qwen3.8-27B-numallama/qwen3.8-27b-q4.bin -z models/Qwen3.8-27B-numallama/tokenizer.bin -m chat

# or an OpenAI-compatible server on http://127.0.0.1:8080/v1
./runq models/Qwen3.8-27B-numallama/qwen3.8-27b-q4.bin -z models/Qwen3.8-27B-numallama/tokenizer.bin -m server -P 8080
```

See the [numallama README](https://github.com/miloserdev/numallama#readme) for all options (context length, sampling, tools, speculative decoding, API details).

## License

Licensed under the [Apache License 2.0](LICENSE), like the original model.

Original model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B). Conversion: numallama.
