---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/stealth-model/Qwen3.8-Flash-Next__Qwen4-news/blob/main/Qwen3.8-Flash-Next/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - qwen3.8-flash-next
  - qwen4
  - qwen-4
  - qwen
  - alibaba
  - stealthmark
  - stealth-model
thumbnail: https://stealthmodels.com/qwen-4/assets/social-card-v2.png
---

<img src="https://huggingface.co/stealth-model/Qwen3.8-Flash-Next__Qwen4-news/resolve/main/panels/hero.webp" alt="The Qwen 4 lineup: Max, 27B, Flash, Plus and a likely 35B-A3B, with the Jade Atlas capybara.">

# Qwen3.8-Flash-Next GGUF and Qwen 4 news

Qwen4-Max came out on 16 November, but only as an API. Alibaba says smaller Qwen 4 models will get open weights, with no date yet. Until then, you can already run the Qwen 4 architecture yourself: Qwen3.8-Flash-Next is Qwen's own preview of it, and the GGUF files are here, with settings for your hardware. When open Qwen 4 weights come out, we plan to add GGUFs for them here too.

By [Stealth Models](https://stealthmodels.com).

<img src="https://huggingface.co/stealth-model/Qwen3.8-Flash-Next__Qwen4-news/resolve/main/panels/flash-next-header.webp" alt="Qwen3.8-Flash-Next, the Qwen 4 preview, on your hardware: 16 GB to 128 GB machines, up to 1M context, Q2, Q4, Q5, MTP and vision.">

## Qwen3.8-Flash-Next, the Qwen 4 preview

Qwen3.8-Flash-Next already uses the new design: a learned n-gram memory next to sparse attention, and 512 experts, of which 10 work on each token. It reads 262,144 tokens natively and up to 1 million with YaRN.

| File | Size | Best for |
|---|---|---|
| `Q2_K_XL/` | 79 GB | 16 to 32 GB GPUs with 64 GB of RAM |
| `Q4_K_XL/` | 111 GB | 24 to 32 GB GPUs with 96 to 128 GB of RAM, AMD Strix Halo, NVIDIA DGX Spark |
| `Q5_K_XL/` | 158 GB | 32 GB GPUs with 128 GB of RAM, 192 GB machines |
| `mmproj-Q8_0.gguf` | 0.6 GB | vision, near lossless, a third less VRAM |
| `mmproj-F16.gguf` | 0.9 GB | vision, full precision |
| `mmproj-Q6_K.gguf` | 0.52 GB | vision, smaller |
| `mmproj-Q4_K.gguf` | 0.46 GB | vision, smallest |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.8 GB | faster output with MTP |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf` | 1.9 GB | MTP when VRAM is tight |

It needs far less VRAM than its size suggests. Only 4 to 5.5 GB has to sit on the GPU; the experts stay in system RAM, and the n-gram table (29 GB, 54 GB in Q5) can stay on your SSD. You need llama.cpp b11971 (28 October) or newer.

### Pick your settings

All rows use a 4-bit KV cache (`-ctk q4_0 -ctv q4_0`), so the whole cache is only about 2 GB at 256k and 8 GB at 1M.

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

`--n-cpu-moe` is how many of the 48 expert layers stay in system RAM. If you run out of memory, raise it by 2; with VRAM to spare, lower it by 1.

### Speed and memory options

- **n-gram table on SSD or in RAM.** `-ot "^per_layer_token_embd\.weight$=CPU" --lazy-mode on` keeps it on your SSD. That costs no VRAM or RAM, just a few small reads per token, so a fast NVMe drive helps. With RAM to spare (29 GB, 54 GB for Q5), `--lazy-mode off` keeps it in memory for slightly faster lookups.
- **ngram-mod.** `--spec-type ngram-mod` drafts tokens from text that's already in your context. It costs no memory and helps most when the answer repeats your input, like code edits, summaries or long documents.
- **MTP.** Qwen trained one multi-token prediction layer, and Unsloth measured 1.3 to 1.7x faster output with it. It takes 2.8 GB of VRAM (1.9 GB with the Q4_K_M file), so raise `--n-cpu-moe` by 3 on Q2 and by 2 on Q4 and Q5. For now it runs on Unsloth's llama.cpp branch `qwen4exp/mtp`; support in official llama.cpp is still in review. Add `-md qwen4/Qwen3.8-Flash-Next/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 5`, or combine both with `--spec-type draft-mtp,ngram-mod`.
- **1M context.** Add `--rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 262144 -c 1000000 -ub 512`. For 512k, use `--rope-scale 2 -c 524288`. Qwen recommends YaRN only when you actually need the length.

### Download and run

```bash
hf download stealth-model/Qwen3.8-Flash-Next__Qwen4-news --local-dir qwen4 \
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

- Set `-t` to your number of physical cores; fast RAM matters most here.
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

Q4_K_XL takes about 89 GB this way and fits 1M context too (about 96 GB). Q5_K_XL fits at 256k if your system gives the GPU about 110 GB. With 192 GB or more, swap `--lazy-mode on` for `--lazy-mode off`.

Qwen's recommended sampling for thinking mode is `--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0`, and for instruct `--temp 0.7 --top-p 0.8 --top-k 20 --presence-penalty 1.5`.

If you run it, tell us your GPU, RAM and speed in the [Community tab](https://huggingface.co/stealth-model/Qwen3.8-Flash-Next__Qwen4-news/discussions) and we'll tune the table.

---

## Qwen 4 timeline and updates

<img src="https://huggingface.co/stealth-model/Qwen3.8-Flash-Next__Qwen4-news/resolve/main/panels/radar.webp" alt="Qwen 4 release radar, checked 24 November 2026: Qwen4-Max is out, API only. Found: first Qwen4 model code (26 Aug), the Qwen3.8-Flash-Next architecture preview (27 Aug), Max, Plus, Flash and 27B named at Apsara (22 Sep), kernel code for Qwen4-27B dense and Qwen4.0-Max (23 Oct), the lineup shown again in Amsterdam as coming soon (29 Oct), the Qwen4-Max preview (11 Nov). Released: Qwen4-Max (16 Nov). Not yet: open weights for 27B, Flash and Plus.">

<img src="https://huggingface.co/stealth-model/Qwen3.8-Flash-Next__Qwen4-news/resolve/main/panels/bar.webp" alt="Where Qwen 4 stands on StealthMark 1.9.0: Claude Fable 5.2 79.1, Qwen4-Max 62.2, Qwen3.8-Max-0902 58.6, Kimi K3.1 Pro 56.4 (open-weights leader), Qwen3.8-27B 49.0 (local leader). Our forecast for Qwen4-27B: 56.0 to 62.0.">

All scores are on our [StealthMark leaderboard](https://huggingface.co/spaces/stealth-model/stealthmark).

### Is Qwen 4 out?
Partly. Qwen4-Max opened as a preview on 11 November and was released on 16 November, as an API only: its weights are not public. Qwen4-Plus, Qwen4-Flash and Qwen4-27B have no date yet. Alibaba says smaller Qwen 4 models will get open weights. As of 24 November, Qwen's Hugging Face account has no Qwen4 weights.

### Which Qwen 4 models are coming?
Alibaba named four: Qwen4-Max, Qwen4-Plus, Qwen4-Flash and Qwen4-27B, and Max is the first one out. Qwen's lead described 27B as the model local developers have been waiting for, and it's reported to be first in line for open weights. A small 35B-A3B looks likely too.

### Will Qwen4-27B run on my PC?
Kernel code calls it dense, built for one or two GPUs at 4-bit. Its predecessor, Qwen3.8-27B, is the best local model in our StealthMark tests today. When the Qwen4-27B weights come out, we'll add GGUFs and per-GPU settings here.

### How good is Qwen4-Max?
It scores 62.2 on StealthMark 1.9.0, the best Qwen model we have measured and 3.6 points above Qwen3.8-Max-0902. Its results task by task are in [our Qwen4-Max results](https://stealthmodels.com/benchmarks/models/qwen4-max).

### How good will Qwen4-27B be?
Our forecast is Qwen3.8-27B plus seven to thirteen points, the pace of earlier Qwen version steps: 56.0 to 62.0. Most of that range would put it ahead of today's best open model, Kimi K3.1 Pro at 56.4. The forecast and every Qwen 4 trace so far are in our [Qwen 4 guide](https://stealthmodels.com/qwen-4/).

---
Text and images: CC BY 4.0, stealthmodels.com. GGUF files: [Qwen Community License 1.0](https://huggingface.co/stealth-model/Qwen3.8-Flash-Next__Qwen4-news/blob/main/Qwen3.8-Flash-Next/LICENSE).
