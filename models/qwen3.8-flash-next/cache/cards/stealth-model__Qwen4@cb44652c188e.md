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

Qwen 4 isn't out yet, but you can already run its architecture. Qwen3.8-Flash-Next is Qwen's preview of it, and it's ready here for your hardware. On release day, the Qwen 4 GGUF files land here too.

By [Stealth Models](https://stealthmodels.com/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page).

## Run Qwen on your hardware

| Model | What it is | Setup guide |
|---|---|---|
| **Qwen3.8-Flash-Next** | Qwen's preview of the Qwen 4 architecture | [Open the guide](#qwen38-flash-next-the-qwen-4-preview) |
| Qwen4-27B | dense, first in line for open weights | on release day |
| Qwen4-35B-A3B | small mixture of experts, likely | on release day |
| Qwen4-Flash | the efficient Qwen 4 tier | on release day |

<img src="https://huggingface.co/stealth-model/Qwen4/resolve/main/panels/flash-next-header.webp" alt="Run it locally: Qwen3.8-Flash-Next, the Qwen 4 preview, on your hardware. 16 GB to 128 GB machines, up to 1M context, Q2, Q4, Q5, MTP and vision.">

## Qwen3.8-Flash-Next: the Qwen 4 preview

Qwen3.8-Flash-Next already uses the new design: a learned n-gram memory next to sparse attention, with 512 experts of which 10 work on each token. It reads 262,144 tokens natively and up to 1 million with YaRN.

| File | Size | Best for |
|---|---|---|
| `Q2_K_XL/` | 79 GB | 16 to 32 GB GPUs with 64 GB of RAM |
| `Q4_K_XL/` | 111 GB | 24 to 32 GB GPUs with 96 to 128 GB of RAM, AMD Strix Halo, NVIDIA DGX Spark |
| `Q5_K_XL/` | 158 GB | 32 GB GPUs with 128 GB of RAM, 192 GB machines |
| `mmproj-Q8_0.gguf` | 0.6 GB | vision, near lossless, a third less VRAM |
| `mmproj-F16.gguf` | 0.9 GB | vision, full precision |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.8 GB | faster output with MTP |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | 1.9 GB | MTP when VRAM is tight |

It fits on far less than its size: only 4 to 5.5 GB has to live on the GPU. The experts wait in system RAM, and the n-gram table (29 GB, 54 GB in Q5) stays on your SSD. Use a current llama.cpp build: b11971 from 28 October or newer.

### Pick your settings

All rows use a 4-bit KV cache (`-ctk q4_0 -ctv q4_0`): the whole cache is only about 2 GB at 256k and 8 GB at 1M.

| Your machine | File | `--n-cpu-moe` at 256k | `--n-cpu-moe` at 1M | RAM for experts (256k / 1M) |
|---|---|---|---|---|
| 16 GB GPU, 64 GB RAM | Q2_K_XL | 42 | 45 (512k max) | 40 / 43 GB |
| 24 GB GPU, 64 GB RAM | Q2_K_XL | 34 | 42 | 33 / 40 GB |
| 24 GB GPU, 96 GB RAM | Q4_K_XL | 41 | 46 | 66 / 73 GB |
| 32 GB GPU, 64 GB RAM | Q2_K_XL | 26 | 33 | 25 / 32 GB |
| 32 GB GPU, 96 to 128 GB RAM | Q4_K_XL | 36 | 41 | 58 / 66 GB |
| 32 GB GPU, 128 GB RAM, best quality | Q5_K_XL | 39 | 42 | 80 / 86 GB |
| AMD Strix Halo or NVIDIA DGX Spark, 128 GB unified | Q4_K_XL | all on the GPU | all on the GPU | none |
| 192 GB or more | Q5_K_XL | all on the GPU on unified memory | all on the GPU | none |

`--n-cpu-moe` is how many of the 48 expert layers stay in system RAM. Out of memory? Raise it by 2. VRAM to spare? Lower it by 1.

### Speed and memory options

- **n-gram table on SSD or in RAM.** `-ot "^per_layer_token_embd\.weight$=CPU" --lazy-mode on` keeps it on your SSD: no VRAM, no RAM, a few small reads per token, so use a fast NVMe drive. With RAM to spare (29 GB, 54 GB for Q5), `--lazy-mode off` keeps it in memory for slightly faster lookups.
- **ngram-mod, free speed.** `--spec-type ngram-mod` drafts tokens from text already in your context. It costs no VRAM or RAM and helps most when the answer repeats your input: code edits, summaries, long documents.
- **MTP, the model's own draft head.** Qwen trained one multi-token prediction layer, and Unsloth measures 1.3 to 1.7 times faster output with it. It takes 2.8 GB of VRAM (1.9 GB with the Q4_K_M file), so raise `--n-cpu-moe` by 3 on Q2 and by 2 on Q4 and Q5. It runs today on Unsloth's llama.cpp branch `qwen4exp/mtp`; support in official llama.cpp is still in review. Add `-md qwen4/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 5`, or combine both with `--spec-type draft-mtp,ngram-mod`.
- **1M context.** Add `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 -c 1000000 -ub 512`. For 512k use `--rope-scale 2 -c 524288`. Qwen advises YaRN only when you need the length.

### Download and run

```bash
hf download stealth-model/Qwen4 --local-dir qwen4 \
  --include "Qwen3.8-Flash-Next/Q2_K_XL/*" "Qwen3.8-Flash-Next/mmproj-Q8_0.gguf"
```

GPU plus system RAM (here a 24 GB GPU with 64 GB of RAM, 256k context):

```bash
llama-server \
  -m qwen4/Qwen3.8-Flash-Next/Q2_K_XL/Qwen3.8-Flash-Next-Q2_K_XL-00001-of-00003.gguf \
  --mmproj qwen4/Qwen3.8-Flash-Next/mmproj-Q8_0.gguf \
  -ngl all --n-cpu-moe 34 \
  -ot "^per_layer_token_embd\.weight$=CPU" --lazy-mode on \
  -fa on -ctk q4_0 -ctv q4_0 -c 262144 \
  -ub 2048 -b 2048 -t 8 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

- `-t` is your number of physical cores; fast RAM matters most here.
- On a 16 GB GPU, use `-ub 1024`.

AMD Strix Halo or NVIDIA DGX Spark (128 GB unified, 256k context):

```bash
llama-server \
  -m qwen4/Qwen3.8-Flash-Next/Q4_K_XL/Qwen3.8-Flash-Next-Q4_K_XL-00001-of-00004.gguf \
  --mmproj qwen4/Qwen3.8-Flash-Next/mmproj-Q8_0.gguf \
  -ngl all -ot "^per_layer_token_embd\.weight$=CPU" --lazy-mode on \
  -fa on -ctk q4_0 -ctv q4_0 -c 262144 -ub 2048 -b 2048 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

Q4_K_XL takes about 89 GB this way, and fits 1M context too (about 96 GB). Q5_K_XL fits at 256k if your system gives the GPU about 110 GB. With 192 GB or more, swap `--lazy-mode on` for `--lazy-mode off`.

Qwen's sampling: thinking `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`, instruct `--temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5`.

What's your setup? Tell us your GPU, RAM and speed in the [Community tab](https://huggingface.co/stealth-model/Qwen4/discussions), and we'll tune the table.

---

<a href="https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page"><img src="https://huggingface.co/stealth-model/Qwen4/resolve/main/panels/radar.webp" alt="Qwen 4 release radar, checked 28 October 2026: not released yet. Found: first Qwen4 model code (26 Aug), the Qwen3.8-Flash-Next architecture preview (27 Aug), Max, Plus, Flash and 27B named at Apsara (22 Sep), kernel code for Qwen4-27B dense and Qwen4.0-Max (23 Oct). Unconfirmed: Kalmia on Arena (21 Oct). Not yet: weights, API and price."></a>

<a href="https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page"><img src="https://huggingface.co/stealth-model/Qwen4/resolve/main/panels/bar.webp" alt="What Qwen 4 has to beat on StealthMark 1.8.8: GPT-6 Astra 74.7, Qwen3.8-Max-0902 63.3, DeepSeek-V4.1-Flash-1017 59.2, Qwen3.8-27B 51.7. Our forecast: Qwen4-27B about 61.7, Qwen4-Max about 73.3."></a>

See where every model stands on our [StealthMark leaderboard](https://huggingface.co/spaces/stealth-model/stealthmark).

## When will Qwen 4 be released?
No date yet. Alibaba confirmed on 22 September that Qwen 4 is in training, and by 23 October hardware teams were already preparing kernels for Qwen4-27B and Qwen4.0-Max. All eyes are on Apsara Europe in Amsterdam on 29 October.

## Which Qwen 4 models are coming?
Four names were on Alibaba's screen: Qwen4-Max, Qwen4-Plus, Qwen4-Flash and Qwen4-27B, with 27B first in line for open weights. A small 35B-A3B looks likely too.

## Will Qwen4-27B run on my PC?
Kernel code calls it dense, built for one or two GPUs at 4-bit. Its predecessor, Qwen3.8-27B, is the best local model in our StealthMark tests today. [Compare it with four other open models](https://stealthmodels.com/benchmarks?models=qwen3.8-27b,deepseek-v4.1-flash-1017,glm-5.3-flash,kimi-k3,deepseek-v4.1-flash&utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page)

## Is Kalmia Qwen 4?
Maybe. Kalmia is an anonymous model on Arena that trips over the same tokens Qwen models do. That points to the Qwen family, not yet to Qwen 4 itself. [See the tokenizer evidence](https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page#evidence)

## Are the leaked Qwen 4 benchmarks real?
None of them traces back to a real source. One chart never appeared on Artificial Analysis, and one number comes from a paper about Qwen3.8-27B.

## How good will Qwen 4 be?
Our forecast: about ten points above each Qwen 3.8 model. That puts Qwen4-27B ahead of today's best open model and Qwen4-Max within reach of GPT-6 Astra. [Our benchmark forecast, model by model](https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page#qwen4-outlook-map)



---
[The full Qwen 4 guide](https://stealthmodels.com/qwen-4/?utm_source=huggingface&utm_medium=referral&utm_campaign=qwen4-page) · [StealthMark leaderboard](https://huggingface.co/spaces/stealth-model/stealthmark) · [Follow us for the release](https://huggingface.co/stealth-model)<br>
Text and images: CC BY 4.0, stealthmodels.com. GGUF files: [Qwen Community License 1.0](https://huggingface.co/stealth-model/Qwen4/blob/main/Qwen3.8-Flash-Next/LICENSE).
