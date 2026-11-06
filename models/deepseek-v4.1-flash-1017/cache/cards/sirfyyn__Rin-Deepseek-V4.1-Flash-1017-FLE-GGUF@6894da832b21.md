---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
tags:
- gguf
- moe
- deepseek
- fle
- flelink
- fle-link
- fastlocalexperts
- expert-offloading
- speculative-decoding
---

[![FLE Link: run FLE in one command](https://flelink.run/media/banner-engine.jpg)](https://flelink.run)

# Rin - Deepseek V4.1 Flash 1017 - FLE - GGUF

Rin is a 2-bit GGUF of DeepSeek V4.1 Flash that runs at about **53 tokens/s on one 32 GB GPU**, with **FLE** (FastLocalExperts). It is built from ISTA DASLab's GSQ-RCO quants and split into a "hot" and a "cold" half along how often experts are used.

## Run it in one click with FLE Link

[**FLE Link**](https://flelink.run) is the desktop app for FLE: pick the Rin recipe, press start, chat. It sets the engine flags, the cache budget and the DSpark drafter for you.

- App and downloads: https://flelink.run , source: https://github.com/sirfyyn/flelink
- Engine: https://github.com/sirfyyn/FastLocalExperts (needs **FLE v1.2.6 or newer**; plain llama.cpp cannot load the split layout)
- Speculative drafter for this model: https://huggingface.co/sirfyyn/DeepSeek-V4.1-Flash-1017-DSpark-FLE-GGUF (the file is also in this repo)

## What is in this repo

| File | What |
|---|---|
| `Rin-Deepseek-V4.1-Flash-1017-FLE-0000{1..6}-of-00006.gguf` | the model, 6 parts (part 6 is the Engram table, taken unchanged from ISTA; it is in this repo, so one download is enough) |
| `FLE-DSpark-V4.1-1017-Q2K-Q8.gguf` | DSpark drafter (byte-identical to the DSpark repo) |
| `fle-usage-profile-ista-iq2_xxs.prof` | usage profile for the RAM expert store's warm start. Without it the first minutes are much slower (30.9 instead of 53 tok/s in our run) |

## How it is made

- **Hot half** (the experts a usage profile asks for most, 1860 experts at a 15.5 GiB budget, 35 to 58 per layer): byte-identical to ISTA's IQ2_XXS.
- **Cold half** (the rest, served from RAM): per tensor either ISTA's IQ2_XXS or ISTA's 1.75 bpw build. 47 of 120 cold tensors use the 1.75 bpw build, picked by gain per cost.
- The split follows FLE's own budget plan, so the engine loads the hot half to the GPU and the cold half to RAM without re-planning. Layout and KV keys: `fle.expert_split.*`.
- The bit distribution follows a usage profile from mixed texts. Other workloads (long documents, code) touch cold experts more often (share 0.73 against 0.37 in the profile), so the saving shrinks there.

## Numbers (against the original, not against ISTA)

KL divergence, top-1 agreement and perplexity ratio are measured against the original weights (Q8/MXFP4 reference), on two texts.

| Text | | KLD | Top-1 | PPL ratio |
|---|---|---|---|---|
| mixed | ISTA IQ2_XXS | 0.248 | 83.5 % | 1.171 |
| mixed | **Rin** | 0.292 | 82.1 % | 1.220 |
| agent | ISTA IQ2_XXS | 0.210 | 85.2 % | 1.013 |
| agent | **Rin** | 0.234 | 84.4 % | 1.021 |

Speed (RTX PRO 4500 32 GB, Ryzen 9950X3D, 120 GB DDR5, Linux, FLE v1.2.6): Rin 53.4 tok/s (median of three fresh starts, DSpark on) against 50.4 for ISTA IQ2_XXS with DSpark, **+6 %**.

Rin is lossy. It is a bit worse than the quant it is made from; the trade is speed. Best performance measured on Linux.

## Recipe with DSpark

```bash
FLE_DRAFT_UBATCH=512 FLE_MOE_PROFILE=fle-usage-profile-ista-iq2_xxs.prof FLE_MOE_PROFILE_READONLY=1 \
FLE_MMAP_NO_RANDOM=1 FLE_MOE_WILLNEED_NOW=1 FLE_MC_OVERLAP=1 FLE_SCHED_ASYNC_CPU=1 \
FLE_UBATCH_LEND=1 FLE_UBATCH_PREFETCH=1 FLE_UBATCH_PREFETCH_SLOTS=3 FLE_PREFILL_LEAN=1 FLE_HC_FUSED=1 \
FLE_MOE_STORE_GB=90 FLE_MOE_STORE_ASYNC=1 FLE_MOE_FUSED_CPU=1 FLE_MOE_POOL=1 \
FLE_MOE_STORE_UPLOAD=1 FLE_PREFILL_WINDOW=8 FLE_PREFILL_WINDOW_MAX=2047 \
llama-server -m Rin-Deepseek-V4.1-Flash-1017-FLE-00001-of-00006.gguf -c 65536 --jinja -ngl 99 -cmoe -t 16 \
  -ctk q8_0 -ctv q8_0 -fit on --fit-target 1115 --moe-expert-cache-inserts 0 -b 4096 -ub 4096 \
  -md FLE-DSpark-V4.1-1017-Q2K-Q8.gguf -ngld 99 --spec-type draft-dspark --spec-draft-p-min 0.7 --spec-draft-n-max 5 \
  --reasoning-budget 8000
```

FLE Link has this as the "Rin" recipe. Do not set `FLE_MOE_CACHE_BUDGET_MIB` with Rin: the hot half already is the GPU cache. `--reasoning-budget 8000` matters when thinking is on: without it the model thought 60,000 tokens on the three.js test tasks and never answered.

## Credits

Base model: DeepSeek ([deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017), MIT). Quantized from: ISTA DASLab's GSQ-RCO quants ([ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF), MIT): Rin's hot half is their IQ2_XXS, the cold half partly their 1.75 bpw build. Both licenses allow redistribution. Engine, split layout, drafter and FLE Link: sirfyyn. I (sirfyyn) make FLE and FLE Link.
