---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
language: [en, zh]
tags: [exl3, quantized, glm, moe, rocm, strix-halo, gfx1151]
---
<p align="center"><img src="assets/banner.png" alt="Yamz Labs" width="100%"></p>

# GLM-5.3-Flash, EXL3

**Model size:** 320B parameters (18B active). The "50B params" figure in the Hugging Face sidebar is not a parameter count: the site counts the packed 16-bit words of the EXL3 format.

**Hardware:** sized to fit one 128 GB machine. The pack is in the standard EXL3 format, so it is not tied to one GPU; it has been tested only on AMD Strix Halo (Ryzen AI Max+ 395) with the Kyojin engine, and other hardware is untested.

An EXL3 quantisation of [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (MoE, 320B total, 18B active) for one 128 GB unified-memory machine (AMD Ryzen AI Max+ 395, Radeon 8060S, gfx1151). It includes the MTP head for speculative decoding. It runs on the [Kyojin engine, built on ExLlamaV3](https://github.com/Yamz-Labs/kyojin), not on stock ExLlamaV3. Quantised tensors by turboderp; layer mix, scale-vector tuning and packaging by Yamz. Not affiliated with Z.ai or turboderp.

## Highlights
- **Quality.** Against the official FP8 model, KLD (how far the output distribution is from FP8; lower is closer) is 0.151 and top-1 agreement is 89.31 % on 30 held-out rows. On 126 of the 129 held-out rows top-1 agreement is 90.11 %.
- **Size.** 99.73 GB for a 320B-total MoE: one 128 GB machine holds it, with MTP and long context.
- **Speed.** Prefill (reading the prompt) 652 tok/s at 8K, 649 at 128K, 619 at 256K and 567 at 500K; decode 27.4 to 36.8 tok/s with MTP (2 drafts), depending on the text, and 26.4 at 500K of context. One Ryzen AI Max+ 395.
- **MTP projection included.** The pack ships `mtp_eh_proj.st` (64 MB), an unquantized copy of the MTP projection. The Kyojin server loads it automatically from the model folder (engine main from 2026-10-05, no flag). It only changes the draft, not the model's answers: 31.6 tok/s greedy with it against 26.7 without, on three prompts of 400 tokens.
- **Engine.** Kyojin, the Yamz engine built on ExLlamaV3, runs it on gfx1151 with ROCm, MTP head included. Use Kyojin 1.5.10 or later: flat speed up to 500K tokens of context, and its MTP drafter picks the draft depth per round (no weights change, pull the tag and restart the server).
- **Image input.** The Kyojin server reads images (OpenAI `image_url` parts) with the vision tower already in this pack (engine main from 2026-10-08, +0.35 GB of GPU memory). First version, improvements coming.
- **Feedback wanted.** Run it on your Strix Halo / Ryzen AI Max machine and tell us your tok/s and hardware: open an issue at https://github.com/Yamz-Labs/kyojin/issues or start a discussion on this page.
![GLM-5.3-Flash against llama.cpp on the same machine](assets/datasheet-vs-llamacpp.png)
![KLD and top-1 agreement against the official FP8 model](assets/datasheet-quality.png)

## How it was made
The quantised tensors come from turboderp's public EXL3 quantisations of this model ([turboderp/GLM-5.3-Flash-exl3](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3)): the 2.05 bpw pack, with the expert layers 29 to 44 taken from the 3.05 bpw pack. Yamz chose that per-layer mix, then ran one short tuning stage: small per-layer scale vectors (`su`/`sv`) of layers 21-44 are tuned against the official FP8 model on 526 text rows of our own. The tuned vectors are written into the normal shards: the pack is a plain EXL3 pack with 12 shards and no override file. Size: 99.73 GB.
## Run
```bash
git clone https://github.com/Yamz-Labs/kyojin && cd kyojin
./build.sh                                   # builds the ROCm extension for gfx1151
source tools/strix_halo/env.sh
python tools/glm/serve.py --model /path/to/this-pack --port 8000 -c 131072 --num-draft 2
```
OpenAI-style HTTP API. `--num-draft 2` turns on MTP. `-c 524288` loads (see Hardware). Exact flags: `tools/glm/SERVE.md` in the engine repo. These steps were run from a clean install on 2026-10-05.

The pack also ships `draft2s.st`, a two-stage draft head. Kyojin 1.5.2 and later pick it up from the model folder automatically; it only speeds up drafting, never changes the answers. `EXL3_DRAFT2S=off` turns it off.

## Hardware
| Machine | Status |
|---|---|
| Ryzen AI Max+ 395, 128 GB LPDDR5X, Radeon 8060S (gfx1151), ROCm | Tested. All numbers below. |
| Other AMD GPUs, NVIDIA GPUs, less than 128 GB | Not tested. The pack is 99 GB: it does not fit in less memory. |

Memory: at `-c 524288` the server holds about 118 GiB of the 124 GiB the system shows, and leaves 1 to 5 GiB free. Treat 128 GB as the floor for this pack at that context: close every other large process, and start clients outside any memory-limited session. Lower `-c` reduces the KV cache; the footprint at `-c 262144` was not measured.

## Quality (reference = official FP8)
| Rows | Engine | KLD vs FP8 | Top-1 agreement with FP8 |
|---|---|---|---|
| 30 held-out rows, 30,720 tokens | this repository's engine (stock path) | 0.15081 | 89.31 % |
| 4 long rows of 4,096 tokens, never trained on | this repository's engine (stock path) | 0.22737 | 88.74 % |
| 126 of the 129 held-out rows (the longest were left out by a memory guard) | served engine | not run | 90.11 % |

The 30 held-out rows and the long rows are disjoint from the training rows (exact-row and 12-gram token-shingle checks against all 129 held-out rows and the long rows; 36 of our rows that shared any 12-gram were removed from training). We report no figure measured on our own training domain. KLD on all 129 rows was not run. The shipped pack (12 shards, 151,554 tensors) reproduces the 30-row result bit for bit on the stock engine: KLD 0.150812.

Quality is measured against FP8, not against other formats. No same-size GGUF quality comparison was run. Task scores of this exact pack were not run; reasoning mode was not part of any task test (non-thinking mode only).
## Speed
Speculative decoding (MTP, 2 drafts) is the default server mode and returns the same tokens as plain decoding. Method: the public engine code, an empty home directory, this published pack downloaded from the hub, the default environment, greedy decoding, thinking off. Prefill is the speed the server reports for the request; decode uses the server's own counters (`/metrics`). Cold prompts of 8K to 256K tokens, plus one of 500K (499,712 tokens, a separate start with `-c 524288`), an essay request of 256 tokens after one discarded 8K request, one request per depth. Full method and sources: [doc/benchmarks.md](https://github.com/Yamz-Labs/kyojin/blob/main/doc/benchmarks.md) in the engine repository.

One Ryzen AI Max+ 395, 128 GB. Tokens per second.

| | 8K | 32K | 64K | 128K | 256K | 500K |
|---|---|---|---|---|---|---|
| Prefill | 652 | 634 | 629 | 649 | 619 | 567 |
| Speculative decode | 35.0 | 32.1 | 33.5 | 30.8 | 34.3 | 26.4 |

By kind of text (card prompts, 256 tokens, median of 3): prose 27.4, chat 30.0, code 36.8 tok/s. On a second Ryzen AI Max+ 395 with 128 GB (Kyojin 1.5): prefill 623 / 629 / 602 at 8K / 32K / 64K; decode 26.3 / 30.8 / 33.0 on prose / chat / code. With `--sessions 4`: 30 tok/s for one request, 37.5 in total for two at once, 40.6 for four.

![GLM-5.3-Flash speed against context](assets/datasheet-speed.png)

**First launch.** The engine tunes its dense GEMM kernels on the first requests and keeps the result in a cache. Since Kyojin 1.5 the GLM server starts without the dense GEMM tuner (first start about 95 s instead of about 13 minutes); `EXL3_DENSE_GEMM_TUNE=1` brings it back. The 8K to 256K row above was taken with the tuner. Without it, the first GLM prefills run at 200 to 240 tok/s on a fresh install and reach the figures above within a few requests.

For scale: llama.cpp (ROCm, UD-IQ1_S 1.56 bpw, `-fa 1 -ub 2048`, no MTP) on the same machine: pp4096 197.9, pp16384 158.3, tg128 16.74, tg at 64K 7.19 tok/s. That quant is coarser than ours, so the comparison covers engine and format together.

### Earlier runs, kept as published
Measured on development builds of the engine, before the figures above. Context settings differ between rows (`-c 98304`, `-c 131072`, `-c 524288`): compare rows within the same setup. Repetitions inside one setup differ by up to 20 %.

| Setup | Prefill tok/s (prompt size) | Decode tok/s |
|---|---|---|
| `-c 98304`, steering preset on, medium-reasoning chat template, `--max-history 2`; prefill mean of 3, decode mean of 6 | 580 (3.5K), 584 (14K), 546 (64K) | 29.0 / 30.3 / 27.6 prose / chat / code at temperature 0; 26.0 / 28.5 / 26.4 at temperature 1.0, top-p 0.95 |
| `-c 131072`, 24K prompt | 609.0 | 28.1 (28.12, 28.09) |
| `-c 524288`, raw engine server, steering preset loaded | 611.9 (3.7K), 585.0 (14.2K) | 27.1 default sampling, 28.1 greedy |
| `-c 524288`, through an OpenAI-style proxy in front of the server | 590.0 (3.7K), 589.8 (14.4K); 22.8K cold prompt: first token after 39.0 s, about 585 | 29.1 default sampling, 28.1 greedy |

A 185K-token recall test found its needle, but the cold prefill of that prompt was not timed.

## Optional steering presets
Optional steering presets for the optional runtime steering hook of the engine are published separately in the Yamz presets repository (https://github.com/yamz-labs/yamz-presets); they are not part of these weights. A ready-made variant of this pack, with the preset bundled as a file the engine applies at load, is [yamz-labs/GLM-5.3-Flash-EXL3-Yamz-Uncensored](https://huggingface.co/yamz-labs/GLM-5.3-Flash-EXL3-Yamz-Uncensored) (same shards).

## Sampling
Use the base-model card values: temperature 1.0, top-p 0.95, no repetition penalty. Greedy decoding is not recommended.

## Limits
- Sampling. Use the card values above; greedy decoding is not recommended, especially for long outputs.
- Only gfx1151 with ROCm is tested.
- The conversion tooling is not published. The engine and these weights are.
- Reasoning mode was not part of the task tests (non-thinking mode only).

## Licence and credits
MIT. Base model GLM-5.3-Flash, Copyright (c) 2026 Z.AI Co., Ltd, MIT, licence file included as `LICENSE`. Our additions: Copyright (c) 2026 Yamz Labs, MIT. Provided as is, without warranty. Engine: built on ExLlamaV3 by turboderp (MIT). GLM is a name of Z.ai; this release is unofficial.

Training data of the base model: see the upstream card. Our tuning stage used 526 text rows of our own (mixed domains: chat, code, agent, math, French); the held-out proof is above.

Quantised tensors by turboderp; layer mix, scale-vector tuning and packaging by Yamz. Not affiliated with Z.ai or turboderp.
