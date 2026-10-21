---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: exllamav3
pipeline_tag: text-generation
language: [en, zh]
tags: [exl3, quantized, deepseek, moe, rocm, strix-halo, gfx1151]
---
<p align="center"><img src="assets/banner.png" alt="Yamz Labs" width="100%"></p>

# DeepSeek-V4.1-Flash-1017, EXL3 (Yamz)

**EXPERIMENTAL.** A demonstration that a 763B-parameter model runs exactly on one 128 GB machine: nothing pruned, no routing approximated. It is slow and not meant for daily agent use. Real decode is about 5 to 11 tokens per second. A tool turn that adds about 2K tokens takes about 20 to 35 seconds before the first token (SSD floor about 24 s).

**Fidelity.** Against the unquantized model: top-1 agreement 0.94, KL divergence 0.07. The 115 GiB SAGE quant, same method: 0.78 and 0.56.

An EXL3 quantisation of [deepseek-ai/DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (763B parameters, about 16B active, 40 layers, 384 routed experts per layer, 6 active), made for the [Kyojin](https://github.com/Yamz-Labs/kyojin) engine on one 128 GB machine with an NVMe.

The parameter count in the Hugging Face sidebar is not the model's: the site counts the packed 16-bit words of the EXL3 format.

## What is in this repo

- The pack: 172 GiB, 2.49 bits per weight on average. Routed experts at 2 or 3 bit, attention and shared experts at 5 bit, MTP/DSpark stages at 4 bit, output head at 6 bit.
- The model's three DSpark draft stages (speculative decoding).
- **Not in this repo: the Engram tables** (FP8, about 190 GB). They are the original files, unchanged. Download them from the source repository, see below.

## Quantisation recipe

- Format: EXL3 (trellis codebook `mul1`, output scales always on), converted with the Kyojin converter from the official checkpoint (routed experts are FP4 in the checkpoint, the rest FP8).
- Bit allocation: every module is quantised at its candidate bit widths; a Hessian-weighted optimiser then picks the width of each routed expert under the size target. Result: 7,525 experts at 3 bit and 7,835 at 2 bit (49 % / 51 %), none removed. Nothing is pruned and no routing is changed.
- Attention and shared experts: 5 bit. MTP experts: 2 bit, MTP other modules: 4 bit. Output head: 6 bit.
- The held-out rows used for the quality table below are never used for calibration.
- The per-expert bit widths are in `config.json` (`quantization_config`).

## Quality

Measured against the unquantized model: top-256 KL divergence and top-1 agreement with its logits, on 20,470 positions (first 10 rows of the held-out agentic set, chunk 4096, the model run in Kyojin with the SSD expert tier).

| Measure | This pack (172 GiB) | SAGE quant (115 GiB) |
|---|---|---|
| Top-1 agreement with the unquantized model | 0.9399 | 0.7766 |
| KL divergence, mean (top-256) | 0.0675 | 0.5609 |
| KL median / 95th percentile | 0.0200 / 0.2307 | 0.1086 / 2.6404 |
| NLL (unquantized model: 1.0279) | 1.0355 | 1.3980 |
| Non-finite positions | 0 | 0 |

The SAGE quant keeps the attention in FP8 and quantizes only the routed experts (1.62 bit); sizes and what is quantized differ.

No task scores were run.

## Run it with Kyojin

Hardware: AMD Ryzen AI Max+ 395, 128 GB unified memory, Radeon 8060S (gfx1151), ROCm, an NVMe with room for 365 GB (this pack plus the Engram tables). Only this machine is tested.

```bash
git clone https://github.com/Yamz-Labs/kyojin && cd kyojin && ./build.sh
hf download yamz-labs/DeepSeek-V4.1-Flash-EXL3-Yamz --local-dir ./ds41-pack
# Engram tables: original files from the DeepSeek repository (two shards, about 190 GB)
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors config.json model.safetensors.index.json \
  --local-dir ./engram_fp8
source tools/strix_halo/env.sh
DS41_PACK=./ds41-pack DS41_ENGRAM_DIR=./engram_fp8 tools/ds41/ds41_serve.sh
```

The server speaks the OpenAI API on port 8000 (chat, completions, streaming, thinking, tool calls, image input) and uses the model's DSpark stages for speculative decoding: every drafted token is verified by the full model (on rare fp16 near-ties the verified text can differ from plain decoding). The first start validates the pack, tunes and warms up before it reports `ready`.

**How a 172 GiB pack runs in 128 GB.** The engine keeps a pool of expert slots in memory and reads the other experts from the NVMe when the router picks them (SSD expert tier). Every expert is read exactly as stored. Setup, knobs and the Engram layers: [doc/DS41.md](https://github.com/Yamz-Labs/kyojin/blob/main/doc/DS41.md).

## Speed

Kyojin 1.6.0, one machine as above, greedy, thinking off. Measured on 20 October 2026 on a machine that shared GPU and NVMe with other jobs, so these are ranges, not a benchmark. A cold long prompt reads most of the pack from the NVMe once (about 3 GB/s), so prefill and the first tokens after a large tool result depend on NVMe speed.

| Case | Measured |
|---|---|
| Decode, plain, warm session on real tokens | 9 to 11 tok/s |
| Decode, 5 workloads (chat, code, tool, after tool, follow-up), plain or DSpark depth 4, mean of 2 rounds | 4.7 to 11.2 tok/s |
| Decode, DSpark session, random token rows (acceptance not representative) | 2.8 to 19 tok/s |
| Prefill 2K / 8K / 16K prompt (cold / warm run) | 60 / 104, 257 / 187, 270 / 307 tok/s |
| Follow-up turn, few new tokens, plain with prompt cache | first token in 1 to 4 s |
| Agent tool turn, about 2K new tokens | first token in 21 to 35 s, 65 to 74 GB read |
| First call with a persisted 20K-token system prompt | 10.3 s (108 s cold) |

- Memory: with a 78 GiB expert pool the process stays under the 128 GB (peak 117.2 GiB in a quality run with a 68 GiB pool). Keep 512 MiB free.
- With DSpark on, follow-up turns re-prefill the whole history in this build (30 to 88 s). For chat use `DS41_DRAFT=0`.
- No depth curve beyond 16K was measured for this release.

## Sampling

Use the base-model values: temperature 1.0, top-p 1.0.

## Limits

- Only gfx1151 with ROCm is tested. The EXL3 format is not tied to one GPU, but this pack does not fit in less than about 128 GB of memory plus NVMe.
- The conversion tooling is not published. The engine and these weights are.

## Feedback

Run it on your Strix Halo or Ryzen AI Max machine and tell us your tok/s and hardware: [issues](https://github.com/Yamz-Labs/kyojin/issues) or this page's discussions.

## Licence and credits

MIT. Base model DeepSeek-V4.1-Flash-1017 by DeepSeek, MIT; its licence file is included as `LICENSE`. Our additions: Copyright (c) 2026 Yamz Labs, MIT. Provided as is, without warranty. Engine: built on ExLlamaV3 by turboderp.
