---
license: mit
library_name: gguf
tags:
- gguf
- quantized
- deepseek
- mixture-of-experts
- coding
- imatrix
- dwarfstar
- rocm
- strix-halo
- long-context
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
---

# DeepSeek-V4-Flash-0731 DS4 Antirez Coder Halo 128K MixQ

**A mixed-precision GGUF for Antirez’s DwarfStar (`ds4`) runtime, tested on a 128 GB AMD Strix Halo.** The 84.15 GiB model scored **26/50** on a frozen coding suite, compared with **21/50** for the unmodified reference build. That is a measured difference on this suite, not proof of a general coding improvement.

[![Buy Me A Coffee](https://cdn.buymeacoffee.com/buttons/v2/default-green.png)](https://www.buymeacoffee.com/Datayoda)

> **Check fit and compatibility before downloading:** The GGUF is 84.15 GiB and was tested on a 128 GB Strix Halo with DwarfStar (`antirez/ds4`). It used about 90 GiB of resident model memory in documented runs, plus runtime and context overhead; the system had approximately 14–17 GiB of host-memory headroom. Lower-memory systems may not have enough room. Compatibility with mainline `llama.cpp` is not guaranteed.

| | |
|---|---:|
| Frozen public coding suite | **26/50** (unmodified reference: 21/50) |
| Tested context | 131,072 tokens |
| GGUF size | 84.15 GiB (90,343,990,176 bytes) |
| Runtime | [`antirez/ds4`](https://github.com/antirez/ds4) |
| Tested hardware | 128 GB AMD Strix Halo |
| Speed figures | Not published; no prefill/decode throughput measurements included |

## What makes this build different

This is **not a fine-tune**. It uses the reference quantization layout, with only the routed-expert gate, up, and down tensors in `blk.0` and `blk.1` promoted to `Q4_K`.

That two-block recipe was the simplest tested layout to reach the best score in a 12-round search. Promoting more layers did not reliably improve results: promoting `blk.0`, `blk.1`, and `blk.2` scored 16/50, while promoting `blk.0`, `blk.1`, and `blk.5` tied at 26/50.

![Public coding-suite scores and layer promotions across the C-R00–C-R11 search](c-r02-search-summary.png)

## Run it

Built for DwarfStar. Compatibility with mainline `llama.cpp` is not guaranteed.

```bash
ds4-server --rocm \
  --model DeepSeek-V4-Flash-0731-DS4-Antirez-Coder-Halo-128K-MixQ.gguf \
  --ctx 131072 \
  --port 8080
```

If you are unsure about available memory, start with a shorter context. Actual memory use depends on context length, runtime revision, launch settings, and other active applications.

## Quantization and file details

- Routed-expert gate/up: `IQ2_XXS`, except `blk.0` and `blk.1` at `Q4_K`.
- Routed-expert down: `Q2_K`, except `blk.0` and `blk.1` at `Q4_K`.
- Attention, shared experts, and output head: `Q8_0`.
- Importance matrix: the published reference imatrix, not a coding-specific one.
- File SHA-256: `2d9c653eacbf4f730d0b2ef0d75e56307b0832659b91b54d7a7d07e901799d4b`

## Evaluation notes

The coding result is from 50 frozen public tasks with deterministic decoding at temperature 0: this build passed 26; the unmodified reference passed 21. It is a result on this task set and runtime configuration—not a general guarantee of better coding performance. A separate 15-task private confirmation suite scored 0/15 for both this build and the baseline, so it did not distinguish them.

A 130,029-token prompt was accepted at a 131,072-token context setting and generated 512 tokens without observed truncation. This verifies the tested configuration, not equal quality across the full context window.

## Credits

Thanks to the DeepSeek team, Antirez and DwarfStar contributors, and the llama.cpp/GGML community. If you test the model, sharing your runtime, hardware, context length, memory use, and reproducible results is welcome.