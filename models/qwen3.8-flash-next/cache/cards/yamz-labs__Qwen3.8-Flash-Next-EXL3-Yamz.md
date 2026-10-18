---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: image-text-to-text
language: [en, zh]
tags: [exl3, quantized, qwen, moe, rocm, strix-halo, gfx1151]
---
<p align="center"><img src="assets/banner.png" alt="Yamz Labs" width="100%"></p>

# Qwen3.8-Flash-Next, EXL3

**125B parameters, 6B active, 95 GB.** Runs on one 128 GB AMD Strix Halo machine with the Kyojin engine, with exact speculative decoding; speed measured to 500K tokens of context, recall tested to 128K.

EXL3 build of [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by Yamz Labs. Kyojin is our engine, built on ExLlamaV3: https://github.com/Yamz-Labs/kyojin. Not affiliated with Qwen.

## Highlights
- **Fidelity.** Against the original model's FP8 build: top-1 agreement 93.81 % and KL divergence 0.0216 on 200 public neutral texts (3084 scored positions), one scorer for all engines. Strata v0.1.40 scores 93.64 % and 0.0316 on the same texts, Gufo v0.8.0 95.36 % and 0.0156 (111.3 GB pack). Table below.
- **Exact speculative decoding.** The speculative path returns the same tokens as plain decoding.
- **Long context.** Prefill stays flat with depth: 1621 tok/s at 8K, 1505 at 128K, 1399 at 256K, 1224 at 500K. Speculative decode: 52.2 tok/s at 8K, 46.9 at 256K, 48.1 at 500K. Needle recall 10/10 at 64K and 128K (Kyojin 1.5.3), speculative output identical to plain.
- **Speed.** Speculative decode 47.5 / 52.8 / 73.2 tok/s on prose / chat / code. On the same machine and method, measured by us: Gufo v0.8.0 32.5 / 37.3 / 52.3 and Strata v0.1.40 41.5 / 46.7 / 61.6. Prefill 1621 tok/s at 8K against 1250 for Gufo. Tables below.
- **Reliability.** A batch of server fixes ships with it: failed or abandoned requests do not leak into the next one, dead clients stop generating, prompt-cache restore is exact.
- **Open engine.** Kyojin is open: https://github.com/Yamz-Labs/kyojin
- **Engine.** Use Kyojin 1.5.10 or later (no weights change: pull the tag and restart the server). It also warms the first prefill at load, so the first request no longer pays a one-time delay of about 1.3 s.
- **First version.** Speed and quality updates are coming.
- **Feedback wanted.** Run it and send tok/s plus hardware: https://github.com/Yamz-Labs/kyojin/issues, or a discussion on this page.

![Summary](assets/datasheet-summary.png)
![Fidelity to the original model](assets/datasheet-fidelity.png)
![Long context](assets/datasheet-context.png)
![Speed against Gufo and Strata on the same mini PC](assets/datasheet-speed.png)

## Run
First install the engine (ROCm torch for gfx1151, build): follow the Quickstart in https://github.com/Yamz-Labs/kyojin#quickstart (about 10 minutes). Then, from the engine folder:
```bash
source tools/strix_halo/env.sh
hf download yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz --local-dir ~/models/qwen38
python tools/qwen/serve.py --model ~/models/qwen38 --port 8000 -c 131072
```
OpenAI-style HTTP API on port 8000. For prompts up to 500K tokens start with `-c 524288 --cache-bits 8` (past 262K the server uses the 8-bit cache). Speculative decoding is on by default. Details: `tools/qwen/SERVE.md` in the engine repo.

The repo also holds `draft2s.st`, a two-stage draft head. Kyojin 1.5.2 and later pick it up from the model folder automatically (download it with the model); it only speeds up drafting, never changes the answers. `EXL3_DRAFT2S=off` turns it off.

## Hardware and first load
| Machine | Status |
|---|---|
| Ryzen AI Max+ 395, 128 GB, Radeon 8060S (gfx1151), ROCm | Tested. All numbers here. |
| Other GPUs, less than 128 GB | Not tested. The pack is 95 GB and does not fit in less memory. |

The pack needs about 113 GiB of the 128 GB; at `-c 131072` about 14 GiB stayed free in our runs. Close other large programs first. The default context is 65536. Start to ready takes about 90 s with the pack in the page cache; the server warms its kernels before it answers, so the first request is not slow.

## Speed
Same machine, one method for every engine: non-thinking chat, an essay request on cold prompts of 8K to 256K tokens (Kyojin also at 500K, 499,712 tokens, a separate start with `-c 524288 --cache-bits 8`), 256 greedy tokens, speculation on, engine-reported prefill, defaults. Kyojin runs this pack; rival rows were measured by us on this machine with Strata v0.1.40 (UD-IQ4_XS pack) and Gufo v0.8.0 (UD-Q4_K_XL pack). Each project also publishes its own figures, on its own machine and with its own method. Full method, sources and raw tables: [doc/benchmarks.md](https://github.com/Yamz-Labs/kyojin/blob/main/doc/benchmarks.md) in the engine repository.

Prefill, tokens per second:

| | 8K | 32K | 64K | 128K | 256K | 500K |
|---|---|---|---|---|---|---|
| Kyojin (this pack) | 1621 | 1548 | 1542 | 1505 | 1399 | 1224 |
| Gufo v0.8.0 | 1250 | 1308 | 1289 | 1230 | - | - |

Speculative decode, tokens per second:

| | 8K | 32K | 64K | 128K | 256K | 500K |
|---|---|---|---|---|---|---|
| Kyojin (this pack) | 52.2 | 51.1 | 46.6 | 54.8 | 46.9 | 48.1 |
| Gufo v0.8.0 | 35.3 | 31.9 | 35.8 | 31.4 | - | - |
| Strata v0.1.40 | 39.0 | 38.7 | 36.4 | 35.0 | 37.5 | - |

By kind of text (card prompts, chat / prose / code): Kyojin 52.8 / 47.5 / 73.2; Gufo 37.3 / 32.5 / 52.3; Strata 46.7 / 41.5 / 61.6 (client-timed). Gufo could not take the 256K prompt (about 264K tokens by its count, over its 262144-token window).

Strata prefill: we do not report a figure of our own. An independent measurement by Ciru Inference Lab ([report](https://llm.ciru.ai/strataflash/), 6 October 2026; cold prompts of 32K, 64K and 120K tokens, a different machine and different prompts) gives 795 / 762 / 709 tok/s at 32K / 64K / 120K; the lab states that it did not reproduce the publisher's exact workload.

![Qwen3.8-Flash-Next against Gufo and Strata](assets/datasheet-speed.png)

Fidelity: KL divergence and top-1 agreement against the official FP8 release, 200 public neutral texts, 3084 scored positions, fixed-seed points, one scorer for all three engines, row-cluster bootstrap. The rival engines were patched only to write out their top-20 log-probabilities; their GGUF packs come from the BF16 weights. Run-to-run is the KL between two runs of the same engine on the same points.

| Engine | Pack | Top-1 | KL | KL run-to-run |
|---|---|---|---|---|
| Kyojin (this pack) | 95 GB | 93.81 % | 0.0216 | 0.0008 |
| Strata v0.1.40 | 93.7 GB | 93.64 % | 0.0316 | 0.0074 |
| Gufo v0.8.0 | 111.3 GB | 95.36 % | 0.0156 | 0.0000 |

## Sampling
Use the original card values: temperature 1.0, top-p 0.95, top-k 20. Greedy decoding is not recommended.
## Optional uncensor preset (off by default)
The folder `uncensor-preset/` holds an optional runtime preset for Kyojin. It edits nothing in the weights and does nothing unless you switch it on:
```bash
EXL3_ABLIT_RUNTIME=~/models/qwen38/uncensor-preset/uncensor_spec.json python tools/qwen/serve.py --model ~/models/qwen38 ...
```
The preset is applied by the Kyojin engine; stock ExLlamaV3 and TabbyAPI ignore it. Do not copy those files next to the weights: the engine applies a spec found there automatically. Measured with the preset on: harmful prompts refused 4 of 100 (plain: 99 of 100); on harmless prompts the top-1 agreement with the plain model is 98.8 %; general tasks hold (MMLU-Pro 65.5 vs 65.0, GSM8K 96.7 vs 98.0, IFEval 91.0 vs 89.0). Tool calling: -1.3 points on 300 held-out items (95 % interval -5.7 to +2.7), lower on an older 100-item set (74 vs 81). **For tool-heavy agent use, keep it off.**

## Limits
- Tested on AMD Strix Halo (gfx1151, ROCm) only.
- Thinking and tool use were tested in the engine's own checks, not on public leaderboards.
- The conversion tooling is not published. The engine and these weights are.
- Speeds come from the engine release that goes with this card; the first requests after a fresh install tune some kernels and can be slower.

## Licence and credits
Qwen Community License 1.0, inherited from the original model, file `LICENSE` included. Copyright (c) 2026 Qwen. Please read its conditions (notably on hosted model-as-a-service use) before you deploy. Our additions: Copyright (c) 2026 Yamz Labs. Provided as is, without warranty. Engine: built on ExLlamaV3 by turboderp (MIT). Qwen is a name of its owners; this release is unofficial.

<details><summary>Licence text, verbatim</summary>

```
Qwen Community License 1.0

Copyright (c) 2026 Qwen

Permission is hereby granted, free of charge, to any person obtaining a copy of this software, including the model weights, parameters, configuration files, inference code and associated documentation files (collectively, the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, sell, deploy, host, fine-tune, and create derivative works from (collectively, "Use" or "Using") copies of the Software; and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. If the Software (or any derivative works thereof) is Used for any of the licensee's commercial products or services that have more than 100,000,000 monthly active users or US$ 20,000,000 (or equivalent in other currencies) monthly revenue, respective model name must be prominently displayed on the user interface of such product or service; and,

2. If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works for any commercial purpose. The foregoing requirement shall not apply to the licensee's internal Use of the Software, provided that such Use does not make the Software, its outputs, or its underlying model capabilities available to any third party.

"Model as a Service" means giving a third party access to language model inference or fine-tuning (e.g., via API or a hosted endpoint) in a manner that allows such third parties to exercise meaningful control over the inputs, parameters, or training data. This does not include the mere relaying of requests to models hosted by other third parties. 
“AI Work Assistant” means an independent AI-powered product primarily designed for AI-assisted coding or office productivity (e.g., Qoder and QwenWork). It does not include: (a) a single-purpose AI tool (such as an AI translation tool); (b) an AI assistant primarily designed for a domain other than coding or office productivity (such as Taobao AI Shopping Assistant or AMap AI Chat); or (c) an AI assistant that is a feature of a product whose primary purpose is not AI-assisted coding or office productivity.

THE SOFTWARE AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL QWEN, ITS AFFILIATES OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. THE USE OF THE SOFTWARE MUST COMPLY WITH APPLICABLE LAWS AND REGULATIONS, AND MUST NOT INFRINGE THE INTELLECTUAL PROPERTY RIGHTS OF ANY THIRD PARTY.

For any questions regarding this license, please contact model-business@notice.qwencloud.com.
```
</details>

Quantisation and packaging by Yamz Labs.
