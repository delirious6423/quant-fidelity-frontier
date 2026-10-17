---
license: other
license_name: qwen-community-license-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
tags:
  - gguf
  - qwen
  - moe
  - mixed-precision
  - speculative-decoding
pipeline_tag: text-generation
---

# Qwen3.8-Flash-Next, IQ3_S-expert GGUF

A GGUF quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a
512-expert mixture-of-experts model with a 262 144-token context and image input. It averages 4.081 bits
per weight.

It is the same file as
[boxrunner/Qwen3.8-Flash-Next-mixC-GGUF](https://huggingface.co/boxrunner/Qwen3.8-Flash-Next-mixC-GGUF)
(map C) with one change: the routed experts' gate and up projections are **IQ3_S** instead of IQ2_XXS.
Every other tensor is byte-identical to map C. Routed-expert down is IQ4_NL, the n-gram table is IQ4_NL,
and the rest is Q5_K, Q6_K and Q8_0. All 24 576 experts together take 53.3 GiB, so on a 64 GB Mac they do
not all fit in memory at once. Map C's do (40.4 GiB).

## Files

| file | bytes | sha256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf` | 45 950 320 320 | `c7f7de6263e37dbd0740d052c756bdc925cbf2d4464ddd60e6f7da11b668baac` |
| `Qwen3.8-Flash-Next-IQ3_S-00002-of-00002.gguf` | 44 312 166 688 | `29e45f1312c121f98bcd2ee54b1b14d7ae03ecba04be3c8a1eaf39ccade810b2` |
| `mmproj-Qwen3.8-Flash-Next-F16.gguf` (image projector) | 904 003 840 | `c67bb0f275b28b48a6096e0ae6408dab6189c82447f8d5932469bc4b856f0341` |
| `mtp-head-IQ4_NL.gguf` (draft head, optional) | 2 362 171 520 | `132d3883361eb5fb2b3b9f5051deb9963e202271f07c809a4a69c96a741b0666` |
| `add-draft-head.py` | small | puts the draft head into part 1 for llama.cpp |

The two parts were cut with llama.cpp's `gguf-split`. Put them in one folder and open part 1; the engine
finds part 2 itself. `llama-gguf-split --merge` turns them back into one file. The projector and the draft
head are the same files as in the map C repository.

## Run it in llama.cpp

You need llama.cpp at commit `73a43d1f` or later, the first with this architecture (`qwen4exp`).

```
llama-server -m Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-F16.gguf -c 32768 -ngl 99 -fa on --jinja
```

**On a 64 GB Mac this file does not fit in llama.cpp's GPU memory.** It loads, then the first computation
fails with `Insufficient Memory`, with or without the draft head. We have not run it on a Mac with more
memory. On a 64 GB Mac, use map C with llama.cpp.

**With the draft head.** This needs a llama.cpp build that contains pull request
[#28243](https://github.com/ggml-org/llama.cpp/pull/28243), tested at `53b1389d`. That build reads the draft
layer from the model file itself:

```
pip install gguf
python3 add-draft-head.py Qwen3.8-Flash-Next-IQ3_S-00001-of-00002.gguf mtp-head-IQ4_NL.gguf IQ3_S-mtp
llama-server -m IQ3_S-mtp-00001-of-00002.gguf -ngl 99 -fa on --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2
```

The script copies part 1 with the head added and links part 2 under the new name. Keep the original
parts: a llama.cpp without that pull request refuses the file with the head.

## Run it in Boxrunner

Boxrunner is a macOS app for local models (not yet publicly released). Choose
**Qwen3.8-Flash-Next 512x56B (experimental IQ3_S)** in its model library. It downloads these files and sets
context, cache and memory for your Mac. On a 64 GB Mac it keeps most experts in memory and reads the rest
from the SSD as needed. The draft head is off by default.

## Measured speed

Boxrunner only; llama.cpp did not fit (above). One Apple M5 Max with 64 GB (macOS 27), one request at a
time, greedy, no image loaded, 40 448 MiB expert cache (18 206 of 24 576 experts in memory,
the rest read from the SSD). Other machines will differ.

| date, build | prompt | context, cache | prompt processing | generation |
|---|---|---|---:|---:|
| 2026-09-24, `8a97a2fe` | 14 993 tokens, 320 generated, thinking off | 32 768, f16 | 801 tokens/s | 33.5 tokens/s |
| 2026-09-25, `0788e6d21` | ~83 000 tokens, thinking on | 98 304, compressed | — | 25.7 tokens/s, 26.3 with the draft head |

No quality benchmark is reported here.

## How it was made

Map C's file, with the routed experts' gate and up requantized to IQ3_S from our Q8_0 conversion of
`Qwen/Qwen3.8-Flash-Next` (revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`). The importance matrix is
AtomicChat's, the same one map C used (its hash is in `NOTICE`).

## Licence

The **Qwen Community License 1.0**, not Apache-2.0: see `LICENSE` and `NOTICE`. It allows copying,
modifying and redistributing the weights, subject to two conditions. You must keep the notice, and very
large commercial products must display the model name. Model-as-a-Service and AI Work Assistant
businesses need a separate licence from Qwen for commercial use. Read `LICENSE` for the binding text.

Thanks to the Qwen team for the model and to the llama.cpp maintainers for the format and the tools.
Neither endorses this repository.
