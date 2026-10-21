---
base_model: deepseek-ai/DeepSeek-V4-Flash-0731
language:
  - en
license: mit
license_link: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/LICENSE
pipeline_tag: text-generation
tags:
  - gguf
  - turboquant
  - config-i
  - moe
quantized_by: thetom-ai
inference: false
---

# DeepSeek-V4-Flash — Config-I (GGUF)





**284B-parameter MoE (21B active) in 95 GiB — fits and runs on 128 GB unified-memory boxes (DGX Spark, Mac).** Config-I hybrid quantization at **2.88 bpw effective**.

Config-I quantization of [deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) — the official 2026-07-31 release, superseding the preview — (43 layers, 256 routed experts top-6, hash routing, MLA attention, mHC hyper-connections, DSA indexer). Sibling release: [Config-I MLX](https://huggingface.co/thetom-ai/DeepSeek-V4-Flash-ConfigI-MLX) for Apple Silicon.

📖 **How to actually run it well:** the [offlabel operating guide](https://github.com/TheTom/offlabel/blob/main/models/deepseek-v4-flash-0731.md) covers the backend traps, sampling, the repetition-loop fix, and the DwarfStar speed comparison in one place.

## ⚠️ Runtime requirement

This GGUF uses the TurboQuant `TQ3_1S` weight type, so it needs the [TurboQuant llama.cpp fork](https://github.com/TheTom/llama-cpp-turboquant) rather than stock llama.cpp:

```bash
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant          # default branch is what you want
cmake -B build -DGGML_CUDA=ON    # or -DGGML_METAL=ON on a Mac
cmake --build build -j
```

Verified against the fork's default branch (`feature/turboquant-kv-cache`): loads and generates correctly on both Metal and CUDA.

## Recipe

| Tensor group | Type | bpw |
|---|---|---|
| Expert gate/up (all layers) | Q2_0 | 2.25 |
| Expert down, attention, indexer, compressor | TQ3_1S (WHT-rotated) | 4.0 |
| Token embeddings | q8_0 | 8.5 |
| Output head | q6_K | 6.6 |
| Router gates, mHC mixing, APE, norms, sinks | f32/bf16 | — |

## Compression

| | Size |
|---|---|
| FP8/FP4 source | 167 GB |
| BF16 equivalent | ~570 GB |
| **Config-I hybrid (2.88 bpw)** | **95 GiB (102 GB), 3 split files ≤45 GB** |

## Quality

- Wikitext PPL: **15.64 ± 0.38** (ctx 512, 60 chunks) · **12.86 ± 0.29** (ctx 2048, 15 chunks) — measured on CUDA (GB10)
- Coherent greedy generation verified on CUDA, CPU, and Metal.

**Observed behaviors at 2.88 bpw** (honesty notes, measured under heavy I/O contention — quiet-condition re-runs pending):
- Use the official sampling settings: **temperature 1.0, top_p 0.95** (DeepSeek's card). Greedy decoding (temp 0) induces repetition loops on this model family and is not representative.
- The post-`</think>` content stream can degenerate into repetition while the correct answer sits intact in the reasoning block — parse `reasoning_content` when serving with reasoning enabled (observed under greedy decoding; official sampling mitigates).
- Repetition on hard symbolic math was observed under greedy decoding only — being re-evaluated at official sampling settings.
- Broader MMLU/NIAH numbers land here after quiet-condition runs complete.

## Example

```bash
TQ_NO_ROTATE=1 ./build/bin/llama-cli -m DeepSeek-V4-Flash-0731-ConfigI-00001-of-00003.gguf \
  -ngl 99 -c 8192 --temp 1.0 --top-p 0.95 -p "your prompt" -st < /dev/null
```

## License

Inherits the DeepSeek model license from the base model. Quantization by [thetom-ai](https://huggingface.co/thetom-ai).
