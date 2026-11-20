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
tags: [tr-quantisation, qwen3.8, moe, trit, ternary, quantized, gguf, llama.cpp, apple-silicon, macbook]
inference: false
---

# Qwen3.8-Flash-Next TR2.31-G (GGUF) — for 16 GB Macs

**Qwen3.8-Flash-Next on a 16 GB MacBook Air at about 12 tokens/s.** This is Qwen's 177B-parameter MoE model (the 180B checkpoint
without its 4B multi-token-prediction head; 6B active per token) translated with TR Quantisation: the routed experts at 2 trits
(9 levels), the other linear layers at 3 trits (27 levels), one 16-bit scale per group of 128 weights; the n-gram embedding table
rounded to 3-trit values and stored exactly in a Q5_0 container; hyper-connection weights at 8 bits. 89.8 GB on disk; the experts
stay on the SSD and a fixed cache in memory holds the busy ones. No retraining. **It needs our patched llama.cpp**
(release `llama.cpp-tr-b11721-r3` or later): the tensor types `TQ9_128` / `TQ27_128` do not exist in upstream llama.cpp, so stock builds, LM Studio
and Ollama cannot load it. [r4](https://github.com/ternii-ai/tritq-vllm/releases/tag/llama.cpp-tr-b11721-r4) adds a Linux arm64 build with CUDA for the DGX Spark;
there the tensor-core prompt path changes outputs, with GSM8K level (92.80 vs 93.25 with `GGML_CUDA_TQ_MM=0`, n = 1,319, not significant).

**Measured on a real 16 GB Mac:** MacBook Air, Apple M5, 16 GB, macOS, on mains power, nothing else open. Speed and accuracy figures
below; an independent re-test on the same machine reproduced them.

- Base model: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (revision `de4b8e4d`), Qwen Community License
  1.0, by the Qwen team. Built from Qwen's BF16 weights; not derived from any NVIDIA checkpoint.
- Same calibration and the same non-expert tensors as [Qwen3.8-Flash-Next-TR3-G-GGUF](https://huggingface.co/Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF)
  (1,080 of 1,224 tensors byte-identical); only the 144 routed-expert tensors differ (2 trits instead of 3). Sparse attention on
  (`compress_ratios` 4 on the 12 full-attention layers), as in that repo's 2026-11-13 shard 1.
- `Qwen3.8-Flash-Next-TR2.31-G.trpw` (115 MiB): a small learned predictor that the runtime uses to read the next layer's experts from
  the SSD early. It never changes the output (byte-identical text with it on or off); it only makes decode faster. Keep it next to the
  shards; without it the runtime falls back to the model's router for read-ahead and decode is about 5 % slower (measured with 10 experts).
- Runtime: patch series on ggml-org/llama.cpp @ff5a233, [github.com/ternii-ai/tritq-vllm/tree/main/runtime/llama.cpp](https://github.com/ternii-ai/tritq-vllm/tree/main/runtime/llama.cpp)
  (patch 0007 = the 16 GB defaults). Metal (Apple silicon) only for this file.
- Text model only (no vision encoder, no MTP head).
- Name: TR2.31-G = TR Quantisation, 2.31 trits per weight on average over the 176.2B translated weights (experts 120.8B at 2 trits;
  other linear layers 4.2B and the 51.2B n-gram table at 3), codes calibrated with GPTQ (-G).
- Publisher: Wai Chung.

## Quick start: one line

Open Terminal and paste:

```bash
curl -fsSL https://raw.githubusercontent.com/ternii-ai/tritq-vllm/main/mac/qwen38-16gb.sh | bash
```

It checks the Mac (Apple silicon, macOS 13.3+, 16 GB+, free disk), downloads the ready-built llama.cpp-tr and this model (about 90 GB)
into `~/qwen38-tr`, checks every file's SHA-256, starts a local chat server and opens it in the browser. Run the same line again to
restart; finished downloads are kept and interrupted ones resume. Read the script first if you like:
[`mac/qwen38-16gb.sh`](https://github.com/ternii-ai/tritq-vllm/blob/main/mac/qwen38-16gb.sh). Everything runs on the Mac; nothing is
sent anywhere.

Below 48 GB the installer uses a 32k context by default (`TR_CTX` changes it, up to 262144). It reads prompts in batches of 2,048 tokens
and makes the expert cache smaller as the context grows (6,600 MiB at 32k, 5,950 at 64k, 4,750 at 128k), so that every layer stays on
the GPU. Measured on the MacBook Air M5 16 GB with 9k-23k-token prompts: at 32k, prompts read at 118-123 tokens/s and decode runs at
10.9-11.8 tokens/s (13.4 GB wired); at 128k, a 64k-token prompt reads at 113 tokens/s and decodes at 6.8 tokens/s.

## Quick start: by hand

```bash
curl -LO https://github.com/ternii-ai/tritq-vllm/releases/download/llama.cpp-tr-b11721-r3/llama.cpp-tr-macos-arm64.tar.gz
tar xzf llama.cpp-tr-macos-arm64.tar.gz && cd llama.cpp-tr-macos-arm64
hf download Ternii/Qwen3.8-Flash-Next-TR2.31-G-GGUF --local-dir qwen38
caffeinate -ims env LLAMA_MOE_SSD=1 LLAMA_MOE_CACHE_MIB=6750 LLAMA_MOE_CACHE_MAX_BATCH=512 LLAMA_MOE_ASYNC=1 \
  GGML_SCHED_COPY_CB_FIRST=1 GGML_METAL_OPT_GDNPREP=1 GGML_METAL_OPT_NGATED=1 GGML_TQ_NO_REPACK=1 \
  ./llama-server -m qwen38/Qwen3.8-Flash-Next-TR2.31-G-00001-of-00008.gguf -c 4096 -fa on --jinja -np 1 --cache-ram 0 --reasoning off
# then open http://127.0.0.1:8080
```

- `caffeinate -ims` matters: without a power assertion the base M5's GPU clocks down while it waits for the SSD and decode is much
  slower.
- With `LLAMA_MOE_SSD=1` the runtime uses **8 routed experts per token** instead of the model's 10 (measured accuracy-neutral, below)
  and loads the `.trpw` next to the first shard. `LLAMA_MOE_EXPERTS=0` keeps the model's 10; `LLAMA_MOE_PRED_W=` (empty) turns the
  predictor off.
- **Bigger cache, optional:** add `-fit off` and set `LLAMA_MOE_CACHE_MIB=8100` for about 5 % more speed. It leaves under 1 GB free on
  a 16 GB Mac, so only with nothing else open.
- `--reasoning off` gives direct answers; drop it to let the model think first (much longer replies at the same tokens/s).
- 24 GB and larger Macs can run the same command; only 16 GB has been measured, and a larger cache there has not been timed with this
  file.

## Speed (MacBook Air M5, 16 GB)

Quick-start settings (cache 6,750 MiB, 4k context, one user, thinking off), decode tokens/s, server-side median over chat prompts
(GSM8K and general writing / code / explanation prompts, up to 512 tokens each), 180 s idle before each run.

| configuration | decode tokens/s | note |
|---|---|---|
| **this file, release defaults** (8 experts, learned predictor) | **about 12** (11.9–12.6) | 6 runs over two days, ours and an independent re-test |
| same, bigger cache (`-fit off`, 8,100 MiB) | 12.6–13.1 | 5 runs |
| same, 9,709-token prompt at `-c 16384` | 12.6 | one run; prefill 75 tokens/s |
| this file, model's 10 experts, router read-ahead | 9.7–10.1 | 3 runs |
| [TR3-G](https://huggingface.co/Ternii/Qwen3.8-Flash-Next-TR3-G-GGUF) with release r2 (no read-ahead) | 3.6 | same Mac, 11-10 |

- **Prompt processing is slow**: about 75–78 tokens/s on long prompts (a 4,000-token document takes close to a minute before the
  first answer token).
- Memory: wired memory up to 12.9 GB with the defaults at 4k context (at least 1.1 GB left free); 14.2–14.3 GB with the bigger cache.
- The SSD does most of the work: about 255–265 MB is read per token; about 86 % of expert reads hit the cache.

## Accuracy

Greedy, thinking off; paired exact McNemar tests (lost / gained items, p).

**8 experts (the default here) vs the model's 10**, this file, CUDA (DGX Spark), llama-server with 4 slots; GSM8K first 400, HumanEval 164,
MBPP sanitized 257, IFEval 541 (prompt-level strict):

| | GSM8K | HumanEval | MBPP | IFEval |
|---|---|---|---|---|
| 8 experts | 96.5 | 95.1 | 91.1 | 66.0 |
| 10 experts | 95.0 | 95.7 | 91.1 | 65.8 |
| 8 vs 10 | 2 / 8, p 0.11 | 1 / 0, p 1.00 | 3 / 3, p 1.00 | 15 / 16, p 1.00 |

On the Mac (Metal), 8 vs 10 experts on 120 further GSM8K items: 92 / 120 both.

**This file vs the original and vs TR3-G** (10 experts; Metal on the 16 GB MacBook Air for this file and TR3-G; original Q8_0 on
CUDA in an earlier session):

| | GSM8K 400 | HumanEval | MBPP | |
|---|---|---|---|---|
| this file (Metal) | 95.5 | 95.1 | 89.9 | |
| TR3-G (Metal) | 95.25 | 96.34 | 90.27 | this file vs TR3-G pooled 15 / 13, p 0.85 |
| original (Q8_0) | 96.50 | 96.34 | 91.83 | this file vs original pooled 18 / 7, p 0.043 |

- **Level with TR3-G on the Mac.** Against the original the pooled loss is about one point and nominally significant (uncorrected);
  TR3-G on the Mac shows the same against the original (13 / 4, p ≈ 0.049), so most of it is Metal-vs-CUDA arithmetic, not the 2-trit
  experts.
- **At the token level the 2-trit experts do cost something:** WikiText-2 perplexity on the Mac (100 × 512 tokens) 3.69 vs 3.41 for
  TR3-G (+8 %). The task benchmarks above do not show it.
- 8 experts change the text (only 4–59 % of answers, depending on the set, are word-for-word the same as with 10) but not the scores.
- Not measured: multilingual tasks, long-context accuracy, thinking mode, and HumanEval / MBPP / IFEval with 8 experts on Metal.

## Limitations

- **Experimental**, single sequence (`-np 1`), measured at 4k context (one 16k run). Concurrency and cancelled requests are untested.
- **One Mac measured**: MacBook Air M5 16 GB. Other chips and SSDs will differ; older or slower SSDs will be slower.
- Close other large apps: the model uses about 13 GB of the 16.
- **Needs the patched llama.cpp** (`llama.cpp-tr-b11721-r3` or later for the 8-expert default and the predictor; earlier releases load the file but run
  at about a third of the speed).
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

**Trademarks.** Qwen is a trademark of Alibaba Cloud / Alibaba Group. Mac, MacBook Air and Metal are trademarks of Apple Inc.,
registered in the U.S. and other countries and regions. NVIDIA and CUDA are trademarks and/or registered trademarks of NVIDIA
Corporation in the U.S. and other countries. Used only to identify the products we run on or compare with. This project is independent
and is not affiliated with, sponsored or endorsed by Alibaba, Apple, NVIDIA or Hugging Face.

## Contact

hello@ternii.com (questions, bug reports, licence and takedown requests).

## Files and checksums

8 shards, 89,777,601,088 bytes in total, plus the predictor `.trpw` (120,680,652 bytes). Shard 2 holds the n-gram table.

| file | size (bytes) | sha256 |
|---|---|---|
| `Qwen3.8-Flash-Next-TR2.31-G-00001-of-00008.gguf` | 2,276,151,520 | `c05ce4861f626550581bc45d587a7f23b3e4d1124d3a3466081e6e756938f124` |
| `Qwen3.8-Flash-Next-TR2.31-G-00002-of-00008.gguf` | 35,200,169,152 | `3a542dae48de68e443f2830cd081148a8298498bce178d614f33ae216e4a9022` |
| `Qwen3.8-Flash-Next-TR2.31-G-00003-of-00008.gguf` | 9,712,996,704 | `f2b7e7dbed9990754ad980f6d63a948e2a29b5d5f902e0ca257c2762c833a8c5` |
| `Qwen3.8-Flash-Next-TR2.31-G-00004-of-00008.gguf` | 9,724,274,016 | `55ccb38251247de3ba3e59a514ff65147928b32cb422dacd56504c196f0d55e8` |
| `Qwen3.8-Flash-Next-TR2.31-G-00005-of-00008.gguf` | 9,719,729,728 | `82ed8c32ca773b3e0da3304193a6dca15836435bf48d456bfccc679119906eb7` |
| `Qwen3.8-Flash-Next-TR2.31-G-00006-of-00008.gguf` | 9,665,151,104 | `5e33924c0f22e192dc1168bb96e25878517cda244b78117e3f1542cd25a73c91` |
| `Qwen3.8-Flash-Next-TR2.31-G-00007-of-00008.gguf` | 9,687,572,416 | `29236756f165e5159f2443b5ecab2af1d5fc7292babb19626135da2b1615a740` |
| `Qwen3.8-Flash-Next-TR2.31-G-00008-of-00008.gguf` | 3,791,556,448 | `84a0cdc9a06010aedacd2801c1183e65ac128e85d10e09623ef31f577e0290bf` |
| `Qwen3.8-Flash-Next-TR2.31-G.trpw` | 120,680,652 | `79cc1c8258622362688196b1832bd352339d4c3b51e152126888a23aaa866ddf` |
| `SHA256SUMS` | | all of the above, for `shasum -a 256 -c SHA256SUMS` |

Check a download with `shasum -a 256` on each file.
