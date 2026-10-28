---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/stealth-model/Qwen4/blob/main/Qwen3.8-Flash-Next/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - qwen3.8-flash-next
  - qwen4
  - qwen-4
  - qwen
  - qwen4-27b
  - qwen4-max
  - qwen4-flash
  - alibaba
  - stealthmark
  - stealth-model
thumbnail: https://stealthmodels.com/qwen-4/assets/social-card-v2.png
---

<a href="https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page"><img src="https://huggingface.co/stealth-model/Qwen4/resolve/main/panels/hero.webp" alt="Qwen 4: release date, models and benchmarks. Max, 27B, Flash, Plus and a likely 35B-A3B. The Jade Atlas guide by Stealth Models."></a>

# Qwen 4: release date, models and benchmarks

**Qwen 4 is not out yet.** On 28 October 2026 there are no Qwen4 weights on Qwen's Hugging Face account. This page tracks every confirmed sign and what Qwen 4 has to beat. Until it ships, this repo hosts GGUF files of Qwen's Qwen 4 preview, Qwen3.8-Flash-Next, with setup guides for 16 to 32 GB GPUs and 128 GB unified-memory machines.

*Independent Qwen 4 page by [Stealth Models](https://stealthmodels.com/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page). Not affiliated with Alibaba or Qwen. The official Qwen 4 weights will appear at [huggingface.co/Qwen](https://huggingface.co/Qwen); on release day our Qwen 4 GGUF files follow here.*

## Run the Qwen 4 preview today: Qwen3.8-Flash-Next GGUF

Qwen3.8-Flash-Next is Qwen's early preview of the Qwen 4 architecture: a mixture-of-experts model with 512 experts (10 active per token), a hybrid of Gated DeltaNet and full-attention layers, and a learned n-gram table. The files in [`Qwen3.8-Flash-Next/`](https://huggingface.co/stealth-model/Qwen4/tree/main/Qwen3.8-Flash-Next) are dynamic quants, with an 8-bit vision projector we made with llama.cpp. They need llama.cpp build b11669 or newer.

| File | Size | Best for |
|---|---|---|
| `Q2_K_XL/` | 79 GB | 16 to 32 GB GPUs with 64 GB of RAM |
| `Q4_K_XL/` (uploading) | 111 GB | 24 to 32 GB GPUs with 96 to 128 GB of RAM, AMD Strix Halo, NVIDIA DGX Spark |
| `Q5_K_XL/` (uploading) | 158 GB | 32 GB GPUs with 128 GB of RAM, machines with 192 GB or more |
| `mmproj-Q8_0.gguf` | 0.6 GB | image input, 8-bit |
| `mmproj-F16.gguf` | 0.9 GB | image input, full precision |

Q2_K_XL and both vision projectors are ready now. Q4_K_XL and Q5_K_XL are uploading and will appear in the folder within hours.

**Why it runs on less memory than its size.** Three facts from the files' own headers decide the settings below.
1. Only about 4 to 5.5 GB has to live on the GPU. Everything else is expert weights, and each token uses 10 of 512 experts, so experts can sit in system RAM layer by layer (`--n-cpu-moe`).
2. The n-gram table (28.8 GB in Q2 and Q4, 54.4 GB in Q5) is read a few rows at a time. llama.cpp streams it from disk on demand by default (`--lazy-mode auto`), so it needs neither RAM nor VRAM. Keep the files on an NVMe SSD.
3. Only 12 of the 48 layers keep a KV cache: 1.7 GB at 128k tokens in `q8_0`, 3.2 GB in `f16`.

### Download one size

```bash
hf download stealth-model/Qwen4 --local-dir qwen4 \
  --include "Qwen3.8-Flash-Next/Q2_K_XL/*" "Qwen3.8-Flash-Next/mmproj-Q8_0.gguf"
```

Replace `Q2_K_XL` with the size from the table below. Point `-m` at the first shard (`...-00001-of-0000N.gguf`); llama.cpp finds the others.

### Settings for your machine

| Your machine | File | `--n-cpu-moe` | KV cache | Context | System RAM used by experts |
|---|---|---|---|---|---|
| 16 GB GPU, 64 GB RAM | Q2_K_XL | 41 | `q8_0` | 64k | about 39 GB |
| 24 GB GPU, 64 GB RAM | Q2_K_XL | 34 | `q8_0` | 128k | about 33 GB |
| 24 GB GPU, 96 GB RAM | Q4_K_XL | 40 | `q8_0` | 64k | about 64 GB |
| 32 GB GPU, 64 GB RAM | Q2_K_XL | 27 | `f16` | 128k | about 26 GB |
| 32 GB GPU, 96 to 128 GB RAM | Q4_K_XL | 37 | `f16` | 128k | about 59 GB |
| 32 GB GPU, 128 GB RAM, best quality | Q5_K_XL | 39 | `f16` | 128k | about 80 GB |
| AMD Strix Halo or NVIDIA DGX Spark, 128 GB unified | Q4_K_XL | none, all on the GPU | `f16` | 128k | none |
| 192 GB or more, unified or GPU plus RAM | Q5_K_XL | none on unified memory | `f16` | 128k | none |

The `--n-cpu-moe` value keeps the experts of the first N of 48 layers in system RAM. Each value is the highest that leaves room on the GPU for the core weights, the KV cache, the vision projector and about 3.5 GB of working buffers. If you see an out-of-memory error, raise it by 2; if you have VRAM to spare, lower it by 1 for more speed.

**GPU with system RAM** (example: 24 GB GPU, 64 GB RAM):

```bash
llama-server \
  -m qwen4/Qwen3.8-Flash-Next/Q2_K_XL/Qwen3.8-Flash-Next-Q2_K_XL-00001-of-00003.gguf \
  --mmproj qwen4/Qwen3.8-Flash-Next/mmproj-Q8_0.gguf \
  -ngl all --n-cpu-moe 34 \
  -ot "^per_layer_token_embd\.weight$=CPU" \
  -fa on -ctk q8_0 -ctv q8_0 -c 131072 \
  -ub 2048 -b 2048 -t 8 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

- `-ngl all` puts every non-expert layer on the GPU; `--n-cpu-moe` then moves the chosen expert layers to RAM.
- `-ot "...=CPU"` keeps the n-gram table off the GPU, so it streams from your SSD.
- `-t` is your number of physical CPU cores. The CPU computes the experts kept in RAM, so fast RAM and all physical cores matter most.
- `-ub 2048 -b 2048` speeds up reading long prompts. On a 16 GB GPU use `-ub 1024`.
- With 16 GB, use `-c 65536`.

**Unified memory** (AMD Strix Halo, NVIDIA DGX Spark, 128 GB):

```bash
llama-server \
  -m qwen4/Qwen3.8-Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-Q4_K_XL-00001-of-00004.gguf \
  --mmproj qwen4/Qwen3.8-Flash-Next/mmproj-Q8_0.gguf \
  -ngl all -ot "^per_layer_token_embd\.weight$=CPU" \
  -fa on -c 131072 -ub 2048 -b 2048 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

Q4_K_XL needs about 89 GB of GPU memory this way. Q5_K_XL also fits, at about 112 GB, if your system lets the GPU use that much. With 192 GB or more, add `--lazy-mode off` to keep the n-gram table in memory as well.

**Sampling** (Qwen's recommendations): thinking mode `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`; instruct mode `--temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5`. Set the reasoning effort with `--chat-template-kwargs '{"reasoning_effort":"medium"}'`.

*These settings are calculated from the exact tensor sizes in the files. We have not yet measured them on every machine; tell us your results in the [Community tab](https://huggingface.co/stealth-model/Qwen4/discussions).*

<a href="https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page"><img src="https://huggingface.co/stealth-model/Qwen4/resolve/main/panels/radar.webp" alt="Qwen 4 release radar, checked 28 October 2026: not released yet. Found: first Qwen4 model code (26 Aug), the Qwen3.8-Flash-Next architecture preview (27 Aug), Max, Plus, Flash and 27B named at Apsara (22 Sep), kernel code for Qwen4-27B dense and Qwen4.0-Max (23 Oct). Unconfirmed: Kalmia on Arena (21 Oct). Not yet: weights, API and price."></a>

<a href="https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page"><img src="https://huggingface.co/stealth-model/Qwen4/resolve/main/panels/bar.webp" alt="What Qwen 4 has to beat on StealthMark 1.8.8: GPT-6 Astra 74.7, Qwen3.8-Max-0902 63.3, DeepSeek-V4.1-Flash-1017 59.2, Qwen3.8-27B 51.7. Our forecast: Qwen4-27B about 61.7, Qwen4-Max about 73.3."></a>

## When will Qwen 4 be released?
Qwen has not announced a date. Alibaba said on 22 September 2026 that Qwen4 is in training. Qwen Developers teased "big or small?" and "let's do both!" on 10 October, and by 23 October hardware teams were preparing kernels for Qwen4-27B and Qwen4.0-Max. The next date to watch is Alibaba Cloud's first European Apsara Conference in Amsterdam on 29 October; its official programme features Qwen3.8-Max and announces no Qwen4 launch.

## Which Qwen 4 models are coming?
Alibaba's Apsara screen showed four names: **Qwen4-Max, Qwen4-Plus, Qwen4-Flash and Qwen4-27B**. Conference reporting puts Qwen4-27B first in line for open weights. A secondhand testing lead points to a small mixture-of-experts model, likely named Qwen4-35B-A3B, with about three billion active parameters per token.

## Is Qwen4-27B dense, and will it run on my PC?
Kernel code merged on 23 October adds support for "Qwen4-27B dense", with INT4 on one or two GPUs and FP8 on two. Its predecessor, Qwen3.8-27B, is also dense, and it leads the local models in our StealthMark benchmark at 51.7. On release day we will publish GGUF files in this repo for the open Qwen 4 models that fit consumer hardware, and add our StealthMark results as they finish. [Follow us on Hugging Face](https://huggingface.co/stealth-model) to get them.

## Is Kalmia Qwen 4?
Unconfirmed. Kalmia is an anonymous model on Arena, first seen on 21 October. A tokenizer test by AI Tracker Bot found Qwen's quirks in 6 of 8 test tokens and in none of 40 tokens from other labs. That points to the Qwen family, but it cannot tell Qwen 4 from a Qwen3.8 refresh. Alibaba, Qwen and Arena have not commented.

## Are the leaked Qwen 4 benchmarks real?
None traces back to a primary source. The "Qwen 4 Max (Preview) 53" chart is absent from Artificial Analysis's own leaderboard, the 82% SWE-bench figure for "Qwen4-Coder" comes from a page that calls it illustrative, and the Terminal-Bench numbers shared beside Kalmia come from a paper on Qwen3.8-27B.

## How good will Qwen 4 be?
Nobody outside Alibaba has measured it yet. Our forecast adds about ten points to each measured Qwen 3.8 score, with a central range of seven to thirteen points. That would put Qwen4-27B near 61.7, ahead of today's best open model, DeepSeek-V4.1-Flash-1017 at 59.2, and Qwen4-Max near 73.3, close to GPT-6 Astra at 74.7. We will test every Qwen 4 model on StealthMark as soon as it is available.

## Watch
<a href="https://www.youtube.com/watch?v=MFeGdHbvMc0"><img src="https://huggingface.co/stealth-model/Qwen4/resolve/main/panels/film.webp" width="420" alt="Film: Qwen 4 isn't out yet. Here's the gap it has to close."></a>

---
[The full Qwen 4 guide](https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page) · [StealthMark leaderboard on Hugging Face](https://huggingface.co/spaces/stealth-model/stealthmark) · [StealthMark](https://stealthmodels.com/stealthmark?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page) · [Follow for the release](https://huggingface.co/stealth-model)<br>
This page and its images: CC BY 4.0, credit stealthmodels.com. The GGUF files: Qwen Community License 1.0 ([LICENSE](https://huggingface.co/stealth-model/Qwen4/blob/main/Qwen3.8-Flash-Next/LICENSE)). Updated 29 October 2026.
