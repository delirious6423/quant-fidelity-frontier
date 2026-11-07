---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language: [en]
datasets: [nvidia/OpenCodeInstruct, EleutherAI/hendrycks_math, openai/gsm8k, OpenAssistant/oasst2]
tags: [tr-quantisation, qwen3.8, moe, trit, ternary, quantized, gguf, llama.cpp, apple-silicon]
inference: false
---


# Qwen3.8-Flash-Next TR3-G (GGUF, for llama.cpp on Mac) — EXPERIMENTAL

**Qwen3.8-Flash-Next translated with TR Quantisation**: linear-layer weights and routed experts at 3 trits (27 levels, one 16-bit scale
per group of 128 weights); the n-gram embedding table rounded to 3-trit values and stored exactly in a Q5_0 container; hyper-connection
weights at 8 bits; packed as GGUF for llama.cpp. No retraining. **It needs our patched llama.cpp**: the
tensor type `TQ27_128` does not exist in upstream llama.cpp, so stock builds, LM Studio and Ollama cannot load it.

**Experimental.** It runs on Macs from 16 GB up by keeping the experts on the SSD behind a fixed cache, and it is slow on the smaller
ones. Every Mac speed below was measured on one Apple M5 Max (128 GB) with the model's memory capped to emulate the smaller size; **no
real 16, 24, 32, 48 or 64 GB Mac has run it.**

This is not the same file as the vLLM model `Ternii/Qwen3.8-Flash-Next-TR2.57-G`: that one keeps less-used experts at 2 trits, which
the Mac runtime does not support yet. Results do not transfer between the two.

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `de4b8e4d`), Qwen Community License
  1.0, by the Qwen team. About 180B parameters in total; in Qwen's breakdown, 125B with 6B activated, plus 51B n-gram embedding and
  4B MTP. Built from Qwen's BF16 weights; not derived from any NVIDIA checkpoint.
- Runtime: patch series on ggml-org/llama.cpp @ff5a233, [github.com/ternii-ai/tritq-vllm/tree/main/runtime/llama.cpp](https://github.com/ternii-ai/tritq-vllm/tree/main/runtime/llama.cpp).
  Metal (Apple silicon) is supported; CUDA is not.
- Ready-built binaries of that series: [llama.cpp-tr-b11721](https://github.com/ternii-ai/tritq-vllm/releases/tag/llama.cpp-tr-b11721) — macOS arm64 (Metal); the Linux arm64 build in the same release was not tested with this model.
- Text model only (no vision encoder, no MTP head). The n-gram table is inside the GGUF and is read from the SSD through the page cache.
- Name: TR3-G = TR Quantisation at 3 trits per weight, codes calibrated with GPTQ (-G).
- Publisher: Wai Chung.

## Quick start (Mac)

```bash
# Ready-built llama.cpp-tr for Apple silicon (macOS 13.3+), checksums on the release page
curl -LO https://github.com/ternii-ai/tritq-vllm/releases/download/llama.cpp-tr-b11721/llama.cpp-tr-macos-arm64.tar.gz
tar xzf llama.cpp-tr-macos-arm64.tar.gz && cd llama.cpp-tr-macos-arm64
hf download Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF --local-dir qwen38-tr3
# Pick the cache size for your Mac (table below): 35000 (64 GB and up), 13501 (32-48 GB), 6750 (24 GB), 5900 (16 GB)
LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB=35000 LLAMA_MOE_CACHE_MAX_BATCH=512 LLAMA_MOE_ASYNC=1 GGML_SCHED_COPY_CB_FIRST=1 \
GGML_METAL_OPT_GDNPREP=1 GGML_METAL_OPT_NGATED=1 GGML_TQ_NO_REPACK=1 \
./llama-server -m qwen38-tr3/Qwen3.8-Flash-Next-TR3-G-00001-of-00010.gguf -c 4096 -fa on --jinja -np 1 --cache-ram 0
```

To build from source instead (same patch series), replace the first three lines with the lines below and run `./build/bin/llama-server`
in place of `./llama-server`:

```bash
git clone https://github.com/ternii-ai/tritq-vllm && cd tritq-vllm/runtime/llama.cpp
./apply.sh llama.cpp-tr && cd llama.cpp-tr
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build build -j --target llama-server
```

The figures on this card come from source builds. The ready-built binary was checked on this model with two short chat requests only.

Point `-m` at the first shard; llama.cpp finds the other nine next to it. `GGML_METAL_OPT_GDNPREP` / `GGML_METAL_OPT_NGATED` turn on two
decode fusions for this model (patch 0003; off by default, on in every measurement below). The cache needs
`LLAMA_MOE_CACHE_MAX_BATCH=512` (without it, it aborts at start) and was measured single-sequence only (hence `-np 1`; llama-server
otherwise defaults to 4 slots). `--cache-ram 0` turns off llama-server's host-memory prompt cache (default up to 8 GiB): with it on,
the server's own memory grew by about 235 MiB per request until that cache filled (about 35 requests). That is ordinary memory on top
of the wired memory in the table below, up to 8 GiB more: half the RAM of a 16 GB Mac, and at a 38,254 MiB expert cache the server's
footprint reached 50.0 GB on the 128 GB test Mac. With it off, the server's footprint stayed flat. The cost is that a repeated prompt
prefix is processed again.

Qwen3.8 runs in thinking mode by default with `--jinja`. For short answers, start the server with `--reasoning off`, or send
`"chat_template_kwargs": {"enable_thinking": false}` in the request, or raise `max_tokens`. These switches are standard llama.cpp
options; the speed runs below sent `enable_thinking: false` per request; the accuracy run used a thinking-off chat template.

## Which Mac

Measured on an Apple M5 Max (128 GB) with this patch series and this file, emulating each size by capping the expert cache inside the
runtime. Only the expert cache is capped: the n-gram table is still read through the page cache of a 128 GB machine, so a real
smaller Mac will read the SSD more often, and it also has a slower chip and SSD. Expect lower speeds than these. 4k context, one user.
"Speed" = decode tokens/s, two ways: the quick-start server answering GSM8K questions (thinking off, up to 512 tokens each; 15
questions, 60 at the 64 GB setting), and a 25-prompt replay (the prompt, then the reference answer fed one token at a time). Memory =
wired memory (the kind the GPU budget limits) added while the quick-start server ran with `--cache-ram 0`, measured as the whole Mac's
wired memory above its level just before the server started, logged every second, one run: the steady level and, in brackets, the peak of
the brief bursts (5-10 s each, several per run). The Mac was otherwise idle but not empty, so this is mostly, not only, the server.

| Mac memory | expert cache (`LLAMA_MOE_CACHE_MIB`) | speed, quick-start server (GSM8K answers) | speed, replay | wired memory added: steady (brief peak) | fits the assumed GPU budget? |
|---|---|---|---|---|---|
| 64 GB and up | 35,000 MiB | **24.6** (one run) | not run at this size | 40.3 GB (42.3) | yes on 64 GB (45.81 GB): about 5.5 GB to spare, 3.5 at the peak |
| 32 GB, 48 GB | 13,501 MiB | **10.3** (10.32 / 10.40) | 13.75 (one pass) | 18.0-18.3 GB (20.1) | yes (22.91 GB on 32 GB, 34.36 GB on 48 GB) |
| 24 GB | 6,750 MiB | **7.6** (7.62 / 7.59) | 7.74 / 7.65 | 10.9 GB (13.3) | yes (17.18 GB) |
| 16 GB | 5,900 MiB | **7.4** (7.36 / 7.25); slow | 9.78 / 9.80 | 10.0-10.3 GB (12.4) | tight: about 1.2-1.4 GB to spare when steady, but the brief bursts went up to about 0.9 GB over the assumed budget (~11.45 GB), about 5 % of the time; untested on a real 16 GB Mac |

The 13,501, 6,750 and 5,900 MiB settings were each run twice with the server, the first time with the prompt cache on (otherwise the
same command); decode speed was unchanged by it. The 64 GB setting was first measured at 38,254 MiB (server 23.1, replay 22.1 / 27.3),
but its wired memory peaked at 46.5 GB, over the 64 GB budget, so the table uses 35,000 MiB. The server's resident memory (which also
counts the pages of the model file macOS has read, and which it can drop under pressure) was higher, 10.87 GB at the 16 GB setting and
42.15 GB at the 64 GB setting at the end, and it kept rising slowly through every run (about 35-50 MB per request); the wired memory did
not.

- **The GPU budget is our assumption:** two-thirds of the Mac's RAM (conservative; about 75 % is commonly reported for larger Macs). It
  has not been measured on a 16-64 GB machine.
- **Speeds swing from pass to pass**: the same setting moved by about 25 % between passes on the same day (7.6-9.6 tokens/s with this build at about the
  6,750 MiB setting across that day's runs, including an earlier build of the file), the Mac was in use during the measurements, and the cause is not known. Read the rows as a range,
  not a point.
- On 48 GB and 96-128 GB Macs a larger cache (or no cache at all on 128 GB) is expected to be faster, but with this file only the
  cache sizes above have been timed, so we give no figure for it.
- The original model at 8 bits (Q8_0) is 188.8 GB and does not fit any Mac; this file is 114.3 GB on disk.
- **The cache does not change the output:** raw logits are bit-identical with the cache at 6,750, 13,501 and 38,254 MiB and with every
  expert cached. The 5,900 and 35,000 MiB settings were not logit-checked separately (5,900 produced the same number of tokens on all
  25 replay prompts).
- The 16 GB row's replay speed is not evidence that a smaller cache is faster: it was timed on a different day from the 24 GB row, and
  the pass-to-pass swing above is larger than the difference; with the server (same session for all four) 16 GB is the slowest.
- The replay and accuracy runs used patches 0001-0004 of the series; the server runs used the full series 0001-0005 with
  `GGML_TQ_NO_REPACK=1`. Patch 0005 (Arm CPU kernels, on by default on aarch64) has not been timed with this model on a Mac, so the
  quick start turns it off, which gives the measured configuration.

**Accuracy on the Mac** (this file, Metal, 38,254 MiB expert cache, server with 4 slots; greedy, thinking off; GSM8K first 400,
HumanEval 164, MBPP sanitized 257; paired exact McNemar vs the original, Holm over 3 comparisons). That server used 4 slots, a
16k context and the prompt cache, and its memory reached about 55 GB (footprint; about 70 GB resident): do not copy that
configuration onto a 64 GB Mac.

| | GSM8K | HumanEval | MBPP | |
|---|---|---|---|---|
| TR3-G (this file) | 95.25 | 96.34 | 90.27 | no significant difference (Holm p 0.54 / 1.00 / 0.54) |
| Original (Q8_0) | 96.50 | 96.34 | 91.83 | reference, measured in an earlier session |

"No significant difference" is not a claim of equality: GSM8K and MBPP are nominally lower (1.25 and 1.56 points), and the original was
run in another session, where serving nondeterminism of about the same size applies. Multilingual accuracy was not measured for this
file.

## Limitations

- **Experimental.** The expert cache was measured at 4k context only, micro-batch <= 512 tokens, single sequence for speed (the accuracy
  run used 4 slots); concurrency, cancelled requests and long prompts are untested.
- **No real 16-64 GB Mac has run it**; every speed is the M5 Max with memory capped. At 16 GB the quick-start server held about
  10.0-10.3 GB of wired memory, with a brief peak of 12.4 GB, against an assumed GPU budget of ~11.45 GB; a real 16 GB Mac running other
  apps may have less. Memory figures assume `--cache-ram 0`.
- **Slow below 64 GB** (about 10-14 tokens/s at the 32-48 GB setting and about 7-10 at 16-24 GB on the M5 Max), and replay speeds swing by
  about 25 % between passes for reasons not yet known.
- **Needs the patched llama.cpp.** GGUF type ids 44-46 are specific to this series.
- Not the vLLM recipe (TR2.57-G); its benchmarks against NVIDIA's NVFP4 do not apply to this file.
- Same safety properties and limitations as the base model; no additional alignment or filtering was applied.

## Licence (Qwen Community License 1.0)

Derived from [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `de4b8e4d`, Copyright (c) 2026 Qwen)
and distributed under the **Qwen Community License 1.0** (`LICENSE`, unmodified; changes in `NOTICE`). Two conditions pass to you:
(1) above 100,000,000 monthly active users or US$20,000,000 monthly revenue, the model name must be displayed prominently in the
product's user interface; (2) if you or your affiliates run a Model-as-a-Service or AI Work Assistant business (as defined in the
licence), you need a separate licence from Qwen before commercial use. Questions: model-business@notice.qwencloud.com. Built from
Qwen's BF16 weights; not derived from any NVIDIA checkpoint. Modified by Wai Chung; no training was performed. Calibration prompts were
drawn from NVIDIA's OpenCodeInstruct (CC BY 4.0), the MATH and GSM8K training sets (MIT) and OpenAssistant oasst2 (Apache-2.0); see
`NOTICE`. Not an official Qwen release; not endorsed by Alibaba Cloud.

**Trademarks.** Qwen is a trademark of Alibaba Cloud / Alibaba Group. Mac, MacBook Pro and Metal are trademarks of Apple Inc.,
registered in the U.S. and other countries and regions. NVIDIA and CUDA are trademarks and/or registered trademarks of NVIDIA
Corporation in the U.S. and other countries. Used only to identify the products we run on or compare with. This project is independent
and is not affiliated with, sponsored or endorsed by Alibaba, Apple, NVIDIA or Hugging Face.

## Contact

hello@ternii.com (questions, bug reports, licence and takedown requests).

## Files and checksums

10 shards, 114,314,279,712 bytes in total. Shard 2 holds the n-gram table.

| file | size (bytes) | sha256 |
|---|---|---|
| `Qwen3.8-Flash-Next-TR3-G-00001-of-00010.gguf` | 3,298,513,120 | `0cefed7b4bf102865baff941b5b1fdb8834daef1201e4baf9fc3f9099d39d17c` |
| `Qwen3.8-Flash-Next-TR3-G-00002-of-00010.gguf` | 35,200,169,152 | `e2b53954a7161797f29fc6dc9c50735096b77a0717813edbc967c3deb77e6408` |
| `Qwen3.8-Flash-Next-TR3-G-00003-of-00010.gguf` | 9,830,679,904 | `67b7ae3addac69d2a59a1f396d5c8d5b3dc2dea2e17f30b8f9b68e5d67dc3927` |
| `Qwen3.8-Flash-Next-TR3-G-00004-of-00010.gguf` | 9,787,378,592 | `ed233dc6c6fded07e4140fe5229e3953cca5bd1acb773028ba7d527b2515eba7` |
| `Qwen3.8-Flash-Next-TR3-G-00005-of-00010.gguf` | 9,750,677,024 | `c1756080e8da196632796b59e58c908b30921088d881b75141346aebc4bdc871` |
| `Qwen3.8-Flash-Next-TR3-G-00006-of-00010.gguf` | 9,814,991,584 | `57ca0373c4dccba93b4f13f245447f7a59e053aabda44c8e3624c7a45d1b64a5` |
| `Qwen3.8-Flash-Next-TR3-G-00007-of-00010.gguf` | 9,787,378,624 | `aec1f96930267734ff8bd767598e69774f3eb231f3682ae0f49a7ad3959444bf` |
| `Qwen3.8-Flash-Next-TR3-G-00008-of-00010.gguf` | 9,750,676,992 | `95804c21982e24560596b9a82422a9dce76ee09cdc3d266ed513ad61f7dcf5b4` |
| `Qwen3.8-Flash-Next-TR3-G-00009-of-00010.gguf` | 9,819,535,872 | `8419ec2cdf9f6d6ec75185bfa173a036f671b0a0632d6b31f89a158bea03340f` |
| `Qwen3.8-Flash-Next-TR3-G-00010-of-00010.gguf` | 7,274,278,848 | `de47ce52c59ee69c318d5eceb57bd4d638747381fc65f00072df2bf4b1451d2b` |

Check a download with `shasum -a 256` on each file.
