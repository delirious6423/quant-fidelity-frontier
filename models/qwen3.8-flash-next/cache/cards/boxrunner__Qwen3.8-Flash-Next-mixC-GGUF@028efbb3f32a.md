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

# Qwen3.8-Flash-Next, mixed-precision GGUF (map C)

A GGUF quantization of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next), a
512-expert mixture-of-experts model with a 262 144-token context and image input. It averages 3.455 bits
per weight, and on a 64 GB Apple silicon Mac all of its routed experts fit in memory at once.

- Routed experts: gate and up in **IQ2_XXS**, down in **IQ4_NL**, 40.4 GiB for all 24 576 experts.
- The per-layer n-gram table (26.8 GiB) is **IQ4_NL** and sits alone in part 2, which keeps it out of
  llama.cpp's GPU memory on a 64 GB Mac.
- The rest is Q5_K (token embedding, output), Q6_K and Q8_0.

## Files

| file | bytes | sha256 |
|---|---:|---|
| `Qwen3.8-Flash-Next-oQ2.0-mixC-00001-of-00002.gguf` | 47 621 145 408 | `cda764d3d33285da202a057d2180435054dd6a064a68de2e150dc451a779c8a7` |
| `Qwen3.8-Flash-Next-oQ2.0-mixC-00002-of-00002.gguf` | 28 800 138 464 | `61fedfa3c73ce4764bf46957a534c6072f0313d2c5bc85312f777bf252e31da5` |
| `mmproj-Qwen3.8-Flash-Next-F16.gguf` (image projector) | 904 003 840 | `c67bb0f275b28b48a6096e0ae6408dab6189c82447f8d5932469bc4b856f0341` |
| `mtp-head-IQ4_NL.gguf` (draft head, optional) | 2 362 171 520 | `132d3883361eb5fb2b3b9f5051deb9963e202271f07c809a4a69c96a741b0666` |
| `add-draft-head.py` | small | puts the draft head into part 1 for llama.cpp |

Put both parts in one folder and open part 1; the engine finds part 2 itself. The draft head is the
model's own multi-token-prediction layer, with its experts quantized to IQ4_NL. It is only needed for
speculative decoding.

## Run it in llama.cpp

You need llama.cpp at commit `73a43d1f` or later, the first with this architecture (`qwen4exp`).

```
llama-server -m Qwen3.8-Flash-Next-oQ2.0-mixC-00001-of-00002.gguf \
  --mmproj mmproj-Qwen3.8-Flash-Next-F16.gguf -c 131072 -ngl 99 -fa on --jinja
```

On a 64 GB Mac, a 131 072-token context with the projector fits with the default f16 cache. The full
262 144 does not: with f16 it fails at the first computation. It loads with `-ctk q8_0 -ctv q8_0` (text
only) or `q4_0`, but we have not measured what a quantized cache does to quality.

**With the draft head (speculative decoding).** This needs a llama.cpp build that contains pull request
[#28243](https://github.com/ggml-org/llama.cpp/pull/28243). It was measured at `53b1389d`. That build
reads the draft layer from the model file itself, so first write a part 1 that contains it:

```
pip install gguf
python3 add-draft-head.py Qwen3.8-Flash-Next-oQ2.0-mixC-00001-of-00002.gguf mtp-head-IQ4_NL.gguf mixC-mtp
llama-server -m mixC-mtp-00001-of-00002.gguf -c 65536 -ngl 99 -fa on --jinja \
  --spec-type draft-mtp --spec-draft-n-max 2
```

The script copies part 1 with the head added (about 49 GB) and links part 2 under the new name. Keep the
original parts: a llama.cpp without that pull request refuses the file with the head. Use a draft length
of 1 or 2; longer was not faster. On a 64 GB Mac the head fits at a 65 536-token context with the
projector, and not at 131 072.

## Run it in Boxrunner

Boxrunner is a macOS app for local models (not yet publicly released).
Choose **Qwen3.8-Flash-Next 512x56B (ours, map C)** in its model library. It downloads these files and
sets context, cache and memory for your Mac. The draft head is off by default.

## Measured speed

All measured on one Apple M5 Max with 64 GB (macOS 27), one request at a time, greedy, thinking off,
f16 cache, no image loaded. Other machines will differ.

**llama.cpp without the draft head**, master `60b06ab9` (2026-09-19), 32 768-token context:

| prompt | prompt processing | generation |
|---|---:|---:|
| short, 400 tokens generated | — | 50.1 tokens/s |
| ~15 000 tokens | 915 tokens/s | 42.2 tokens/s |
| 125 363 tokens, 131 072 context, projector loaded | 382 tokens/s | 20.0 tokens/s |

**Both engines, with and without the draft head**, 2026-09-26, 88 064-token context, generation in
tokens/s. llama.cpp is PR #28243 at `53b1389d`, Boxrunner is build `adb8d70e3`:

| prompt | llama.cpp | llama.cpp + head | Boxrunner | Boxrunner + head |
|---|---:|---:|---:|---:|
| short (~90 tokens), 256–1 024 generated | 44.1–44.5 | 55.3–74.9 (length 2) | 53.9–55.5 | 64.8–71.9 |
| ~15 000 tokens | 38.1 | 50.5 (length 1) | 44.8 | 53.9 |
| ~83 000 tokens | 23.7 | 29.6 (length 2) | 46.6 | 40.4 |

Boxrunner drafts one token per step. Its drafting numbers were taken after one long request. Right
after start, drafting ran at about half that speed until the first long request.

No quality benchmark is reported here.

## How it was made

Converted from `Qwen/Qwen3.8-Flash-Next` at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` to Q8_0,
then quantized with upstream llama.cpp `73a43d1f` and a per-tensor type map. The importance matrix is
AtomicChat's (its hash is in `NOTICE`). `reproduce.sh` records the commands.

## Licence

The **Qwen Community License 1.0**, not Apache-2.0: see `LICENSE` and `NOTICE`. It allows copying,
modifying and redistributing the weights, subject to two conditions. You must keep the notice, and very
large commercial products must display the model name. Model-as-a-Service and AI Work Assistant
businesses need a separate licence from Qwen for commercial use. Read `LICENSE` for the binding text.

Thanks to the Qwen team for the model and to the llama.cpp maintainers for the format and the tools.
Neither endorses this repository.
