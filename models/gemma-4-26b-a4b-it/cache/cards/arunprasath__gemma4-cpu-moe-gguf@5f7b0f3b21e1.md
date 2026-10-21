---
license: gemma
base_model:
  - google/gemma-4-26B-A4B-it
  - google/gemma-4-26B-A4B-it-assistant
pipeline_tag: text-generation
library_name: gguf
language:
  - en
tags:
  - gguf
  - llama.cpp
  - cpu-inference
  - mixture-of-experts
  - speculative-decoding
  - quantized
  - gemma-4
model_creator: Google
quantized_by: Arun Prasath E G
---

# Gemma-4 26B on a CPU, no GPU: the benchmarked GGUFs

These are the exact model files behind a measured result: a 26-billion-parameter mixture-of-experts model running at about **40 tokens per second single-stream** (output identical to plain Q4_0) and about **124 tokens per second aggregate at batch 32**, on an i9-13900K with 64 GB of DDR5-4800 and no graphics card. Plain Q4_0 with no tricks does about 25.

The write-up with the byte budget, the roofline, the dead ends and the exact commands: **https://apeg.dev/writing/running-gemma4-26b-on-a-cpu/**

Scripts, results and a one-command reproduce: **https://github.com/arun-prasath2005/gemma4-cpu-moe**, archived as DOI [10.5281/zenodo.22762963](https://doi.org/10.5281/zenodo.22762963).

By [Arun Prasath E G](https://apeg.dev/about/), agentic AI architect and forward deployed engineer ([apeg.dev](https://apeg.dev)).

## Files

| file | size | what it is |
|---|---|---|
| `gemma-4-26B_q4_0-it.gguf` | 14.4 GB | Google's Gemma-4-26B-A4B instruction-tuned release, converted and quantized to Q4_0 with llama.cpp. The base of both recipes. |
| `mtp-q4.gguf` | 321 MB | Google's official MTP drafter for Gemma-4, quantized to Q4. Used for speculative decoding, so the output stays identical to the base model. |
| `gemma-4-26B_q4_0_headQ2K.gguf` | 14.3 GB | The same model with the output head (the projection to the 262K-token vocabulary) requantized from 6.5 to 2.4 bits per weight. Per token the head is 32 percent of the bytes read and the experts only 16 percent, so this is where the bytes are. No changed outputs were found on the test prompts. Needs the ik_llama.cpp kernels. |

The base model and the drafter are Google releases under the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Nothing here is fine-tuned; these are conversions and quantizations of the official weights so that the numbers reproduce exactly.

## Run it

Two public forks of llama.cpp, pinned to the commits the numbers were measured on: [atomic-llama](https://github.com/arun-prasath2005/gemma4-cpu-moe#try-it) at `d86eb0b` for the speculative-decode path and the batched benchmark, and [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) at `f96eadd` for the low-bit quantization kernels. The reproduce script builds both.

```bash
git clone https://github.com/arun-prasath2005/gemma4-cpu-moe
cd gemma4-cpu-moe
bash scripts/reproduce.sh      # builds both engines, downloads these GGUFs, runs both recipes
```

Single-stream recipe: 3 of 8 experts plus MTP speculative decoding, 8 threads, because single-stream decode is bound by memory bandwidth and more threads do nothing.

```bash
llama-cli -m models/gemma-4-26B_q4_0-it.gguf \
  --override-kv gemma4.expert_used_count=int:3 \
  --model-draft models/mtp-q4.gguf --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-threads 8 \
  -t 8 -f prompt.txt -n 200 --temp 0 --seed 1 -st --simple-io --no-display-prompt
```

Batched recipe: read the `S_TG` column for aggregate generation tokens per second, 24 threads, because batching moves the bottleneck to compute.

```bash
llama-batched-bench -m models/gemma-4-26B_q4_0-it.gguf \
  --override-kv gemma4.expert_used_count=int:3 \
  -c 8192 -npp 64 -ntg 128 -npl 1,2,4,8,16,32 -t 24
```

## What was measured

| setting | tokens per second |
|---|---|
| Q4_0, single stream, no tricks | about 25 |
| Q4_0, 3 of 8 experts, MTP speculative decoding, single stream | about 40 (37.7 careful, about 41 on a cool machine) |
| Same, batch 32, aggregate | about 124 |

Measured RAM bandwidth was 64.5 GB/s achievable against 76.8 on paper, and single-stream decode already used about 48 of it. That is the wall: tokens per second is bandwidth divided by bytes read per token. Thread count from 8 to 24 changed nothing single-stream.

Running 3 of 8 experts is an approximation. It was checked by generating on real prompts (a word problem, code, a factual question) at top-3 and top-8 and comparing the answers, which matched. Raw-text perplexity rose 1.6x, which is expected for a chat-tuned model on raw text and is not a quality signal here.

Numbers drift with turbo and temperature. What should reproduce is the shape: threads flat for latency, threads helping for throughput, the byte-budget proportions, and the head compressing for free. If the shape does not hold on your machine, that is a bug: open an issue on the repository, and add a row to its community results file.

## Cite

```
Arun Prasath E G. Running Gemma-4 26B at 124 tokens/sec on a CPU, no GPU: recipe, byte budget, roofline.
Version 1.0, September 2026. https://doi.org/10.5281/zenodo.22762963
```
