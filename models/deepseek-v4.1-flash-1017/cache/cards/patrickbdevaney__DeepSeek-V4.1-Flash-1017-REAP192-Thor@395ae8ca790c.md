---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: text-generation
language:
  - en
tags:
  - deepseek
  - deepseek-v4.1-flash
  - moe
  - reap
  - expert-pruning
  - jetson-thor
  - cuda
  - edge-ai
  - engram
---

<div align="center">

# DeepSeek-V4.1-Flash-1017 — REAP-192 (`thor-v1` Pure-CUDA Build)

[![Base Model](https://img.shields.io/badge/Base_Model-DeepSeek--V4.1--Flash--1017-blue)](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
[![Pipeline & CUDA Repo](https://img.shields.io/badge/GitHub-deepseek--v4.1--flash--reap--thor-black?logo=github)](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor)
[![Compression](https://img.shields.io/badge/REAP--192-50%25_Routed_Experts_Pruned-orange)](#format)
[![Target](https://img.shields.io/badge/Hardware-128GB_Jetson_AGX_Thor-76b900)](#memory-on-the-thor)
[![CUDA server](https://img.shields.io/badge/GitHub-deepseek--v4.1--flash--cuda--server-black?logo=github)](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server)
[![Decode](https://img.shields.io/badge/Decode-25--40_tok%2Fs-brightgreen)](#-measured-on-jetson-agx-thor-2026-10-30)
[![Prefill](https://img.shields.io/badge/Prefill-739_tok%2Fs-brightgreen)](#-measured-on-jetson-agx-thor-2026-10-30)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**106.9 GiB pure-CUDA `thor-v1` build of 50%-pruned `DeepSeek-V4.1-Flash-1017` (192 of 384 routed experts) for a single 128 GB NVIDIA Jetson AGX Thor (`sm_110a`) with O_DIRECT NVMe Engram streaming, served by [`deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server) at 25.0 tok/s decode and 739 tok/s prefill.**

| Sibling Artifact | Format | Size | Target Hardware / Runtime |
|---|---|---:|---|
| **[`DeepSeek-V4.1-Flash-1017-REAP192-MXFP4`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4)** | Native MXFP4 + FP8 (`safetensors`) | **146.4 GiB** | Bit-exact 192-expert master checkpoint |
| **[`DeepSeek-V4.1-Flash-1017-REAP192-Thor`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor)** *(this repo)* | `thor-v1` (2.828 bpw K-quant + FP8/BF16) | **106.9 GiB** | Single 128 GB NVIDIA Jetson AGX Thor ([`v41_server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server), **25.0 tok/s** decode, 739 tok/s prefill) |
| **[`DeepSeek-V4.1-Flash-1017-REAP192-GGUF`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF)** | 3-shard GGUF `L3` (`2.46 bpw` I-quant + `Q8_0`/`Q6_K`) | **85.55 GiB** | [`llama.cpp` branch `deepseek-v4.1-reap-thor`](https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor) (12.4 tok/s, 19.9 with DSpark, + vision) |
| **[`patrickbdevaney/deepseek-v4.1-flash-reap-thor`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor)** | Pipeline, `cuda/` server & `llama.cpp` patches | — | Full build pipeline (`ops/s01`..`s12`), [`cuda/`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/tree/main/cuda), and gates |
| **[`patrickbdevaney/deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server)** | The pure-CUDA server for this build | — | `./start.sh`; OpenAI API, DSML tools, web UI |

</div>

---

**DeepSeek-V4.1-Flash-1017-REAP192-Thor** is [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) pruned to **192 of 384 routed experts** (keep-set `1ace9b33134a2ca7`, recorded in [`reap_keepset.json`](./reap_keepset.json)), in the `thor-v1` layout ([`thor.json`](./thor.json), [`cuda/FORMAT_PROPOSAL.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/cuda/FORMAT_PROPOSAL.md)). That layout is built for one **128 GB NVIDIA Jetson Thor** running a pure-CUDA decoder ([`deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server)), with the Engram tables read from NVMe rather than held in memory.

**Text only for now.** The vision tower and aligner ([`vision.safetensors`](./vision.safetensors)) are in this build unchanged, and the pruning protected the experts image tokens route to, but the CUDA server does not run the vision tower yet: image inputs are not accepted. For images, use the [GGUF](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF) with its mmproj on the [`llama.cpp` branch](https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor).

---

## ✨ Highlights

- **Single-Box 128 GB Jetson AGX Thor Fit (`114.9 / 117.0 GiB`):** Resident footprint is **106.9 GiB** (`89.5 GiB` routed experts at `2.828 bpw` across [`layer-00.safetensors`](./layer-00.safetensors)..[`layer-39.safetensors`](./layer-39.safetensors) + native FP8/BF16 dense MLA/DSA/shared-expert/router/mHC path in [`dense.safetensors`](./dense.safetensors)), leaving budget for the **2.0 GiB 1M-token context state** (FP8 KV + indexer keys) and **6.0 GiB decoder workspace**.
- **25–40 tok/s Pure-CUDA Decode (DSpark Speculation On by Default), 739 tok/s Prefill, with O_DIRECT NVMe Engram Streaming:** Measured 2026-10-30 through the server's `start.sh` ([details](#-measured-on-jetson-agx-thor-2026-10-30)), with Engram rows streamed per token from local NVMe via `O_DIRECT` ([`ops/s09_engram_rowfile.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_engram_rowfile.py)). Speculation (the checkpoint's own DSpark drafter, 3 drafts per verify) averages 1.345× over plain decoding on 27 held-out prompts, never below 1.0×, with greedy output token-identical. Plain decoding is 25.0 tok/s; the release figure was 8.09 tok/s.
- **GPTQ Error-Feedback K-Quant Experts with Folded Healing Gain:** Routed experts were requantized once from the MXFP4 source with GPTQ error-feedback rounding ([`ops/s07_quant.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s07_quant.py)) into ggml K-quant blocks (`w13:Q2_K 31, w13:Q3_K 9, w2:Q2_K 28, w2:Q3_K 12`), with the per-layer healing gain folded into `w2` before rounding.
- **Verified Teacher-Forced Fidelity:** Achieves **96.87% top-1 agreement (`kl32 = 0.0607`) on code**, **92.94% (`kl32 = 0.1509`) on math**, **86.46% (`kl32 = 0.3663`) on vision**, and **84.56% (`kl32 = 0.2265`) on agentic tasks** (`0.7756` overall across 185,565 confirmation tokens), outperforming the bit-exact 128-expert alternative (`top-1 = 0.7092`, `kl32 = 0.8026`) by a wide margin.

---

## ⚡ Measured on Jetson AGX Thor (2026-10-30)

Launched with [`./start.sh`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server) at its defaults (32K context, NVFP4 dense overlay, Engram rows on NVMe), queried over `/v1/chat/completions`, greedy, one stream, CUDA 13.0.

| | `v41_server` (this build) | llama.cpp + GGUF L3, same box and prompts |
|---|---:|---:|
| **Decode**, 256-token chat answers (code / explanation / math) | **33.0 / 25.0 / 40.2 tok/s** (speculation on, default); 25.0 plain | 12.3–12.5 tok/s; 17.3–20.7 with `DSPARK=1` |
| **Prefill**, 1,827-token prompt | **398 tok/s** | 159 tok/s (1,800 tokens) |
| **Prefill**, 7,182-token prompt | **739 tok/s** | 147 tok/s (5,871 tokens) |
| Unified memory still free while serving | 9 GiB of 122 (17 with `SPEC=0`) | 25 GiB |

- **Decode** runs at 78% of the memory-bandwidth roofline for the 8.57 GB it reads per token ([ledger](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/docs/zenith/LEDGER.md)). That is up from 8.09 tok/s at release, and every speedup along the way kept the greedy token ids identical.
- **Greedy output** is deterministic: run to run, graph versus no-graph, and across prefill chunk sizes.

---

## 📋 Model Summary

| Property | Value |
|---|---|
| **Base Model** | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`1bae0e1088993b9b82de08f59de5be5a34e42170`, [`config.json`](./config.json)) |
| **Master Checkpoint** | [`patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) |
| **Layout / Format** | `thor-v1` ([`thor.json`](./thor.json) manifest with SHA256 hashes for every file) |
| **Routed Experts** | 192 / 384 per layer ([`reap_keepset.json`](./reap_keepset.json), `1ace9b33134a2ca7`), 2.828 bpw (`w13:Q2_K 31, Q3_K 9; w2:Q2_K 28, Q3_K 12`), **89.5 GiB** |
| **Dense Path** | Native base checkpoint FP8/BF16 tensors ([`dense.safetensors`](./dense.safetensors), [`mtp-0.safetensors`](./mtp-0.safetensors)..[`mtp-2.safetensors`](./mtp-2.safetensors), [`vision.safetensors`](./vision.safetensors)) |
| **Total Build Size** | **106.9 GiB** resident (`114.9 GiB` total with 2.0 GiB 1M-ctx state + 6.0 GiB workspace on 128 GB Jetson Thor) |
| **Measured Speed (2026-10-30)** | **33.0 / 25.0 / 40.2 tok/s** decode (code / explanation / math, speculation on; 25.0 plain), **398 / 739 tok/s** prefill (1.8K / 7.2K-token prompts), greedy batch=1, NVFP4 overlay, Engram on NVMe |
| **Runtime** | `v41_server` ([`deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server)) |
| **License** | MIT ([`LICENSE`](./LICENSE)) |

---

## Format

Experts were requantized once from the MXFP4 source with GPTQ error-feedback rounding into ggml K-quant blocks ([`ops/s07_quant.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s07_quant.py); per-layer mix, protected layers higher; counts of layer x projection-group: w13:Q2_K 31, w13:Q3_K 9, w2:Q2_K 28, w2:Q3_K 12). The heal gain was folded into w2 before rounding.

**NVFP4 dense overlay (`overlay_nvfp4/`, 3.5 GiB, added 2026-10-30).** This is a separate NVFP4 copy of the dense attention and shared-expert weights (`ops/s07_quant.py`). When it is set (`DSV4_NVFP4_OVERLAY`, on by default in `start.sh`), the server reads it instead of their FP8 twins, which roughly halves the dense bytes per token. The FP8/bf16 weights in this repository and the MXFP4 master are unchanged. Set `OVERLAY=0` to serve them.

- **Experts:** 2.828 bits per weight on average, 89.5 GiB in total ([`layer-00.safetensors`](./layer-00.safetensors)..[`layer-39.safetensors`](./layer-39.safetensors)).
- **Whole build:** 106.9 GiB ([`thor.json`](./thor.json) lists every file with its sha256). The dense path ([`dense.safetensors`](./dense.safetensors): attention, indexer, compressor, shared expert, router, hyper-connections) is the base checkpoint's own FP8/BF16 tensors. The vision tower and aligner ([`vision.safetensors`](./vision.safetensors)) are unpruned. The checkpoint's own speculative-decoding (`mtp`) blocks ([`mtp-0.safetensors`](./mtp-0.safetensors), [`mtp-1.safetensors`](./mtp-1.safetensors), [`mtp-2.safetensors`](./mtp-2.safetensors)) are included unmodified.
- **Keep-set:** [`reap_keepset.json`](./reap_keepset.json) lists, per layer, the routed experts that were kept, in their original ids. The router rows and biases were sliced to the same ids.

---

## Memory on the Thor

| item | GiB |
|---|---|
| this build, resident | 106.9 |
| main 1M-token context state (FP8 KV + indexer keys) | 2.0 |
| decoder workspace | 6.0 |
| **total** | **114.9** of a 117.0 GiB budget: fits |

The context-state figure is the pre-registered default (not measured yet). The Engram tables are not in memory: they are read per token from row files on NVMe (below).

**Measured 2026-10-30** (with `./start.sh`, 32K context, NVFP4 overlay on): 17 GiB of the 122 GiB of unified memory are still free while the server is serving. The release figure of 8.09 tok/s (s08 decode gate, 2026-10-25) was the first working engine. See below for current speed.

---

## How it was evaluated

Every number below comes from `s10/final/results.json` of the build pipeline ([`ops/s10_eval.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s10_eval.py), [`RESULTS.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/RESULTS.md)). It is a paired, teacher-forced comparison against the unpruned deepseek-ai/DeepSeek-V4.1-Flash-1017 at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, run in its own MXFP4/FP8 numerics. The comparison uses 185565 scored tokens from 26 held-out documents in the confirmation split. That split was never used to choose the model. The intervals are ninety-five percent document-cluster bootstrap intervals.

- **kl32** is the mean over tokens of the KL divergence on the teacher's top-32 support.
- **top-1 agreement** is the share of tokens where this model's argmax equals the teacher's.
- **dNLL** is this model's next-token NLL minus the teacher's.

The pass/fail thresholds were pre-registered before any result existed ([`ops/s10_gates.json`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s10_gates.json), sha256 `f95b26721124c6a3`). Any threshold amended after a result is listed below with its date, and the result under the original rule is shown beside it ([`DECISIONS.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/DECISIONS.md)).

| arm | kl32 (95% CI) | top-1 agreement | dNLL | confident-flip rate |
|---|---|---|---|---|
| control | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 |
| mxfp4_A | 0.4405 [0.2639, 0.6465] | 0.7980 [0.7431, 0.8481] | 0.2718 [0.1219, 0.4585] | 0.0524 |
| thorA | 0.5194 [0.3261, 0.7423] | 0.7756 [0.7179, 0.8284] | 0.3091 [0.1426, 0.5119] | 0.0619 |
| b128 | 0.8026 [0.5627, 1.0801] | 0.7092 [0.6443, 0.7701] | 0.5469 [0.3271, 0.8010] | 0.0995 |

**Known limitation: multilingual.** Top-1 agreement with the unpruned model on the held-out multilingual documents is lower than in every other domain (confirmation split: mxfp4_A 0.5247, thorA 0.4935, b128 0.4219). The release floor for every other domain is 0.65; for multilingual it was lowered to a collapse guard of 0.4 (2026-10-22, owner decision, before any s10 result). During expert selection, the shipped keep-set scored 0.5233 on the screen split against 0.5592-0.5741 for random keep-sets of the same size. Keep-sets that protected more multilingual experts recovered multilingual only by losing general, finance and science text, so the better all-round keep-set ships. For multilingual use, prefer the unpruned model.

**Gate amended after the result (2026-10-24).** The pre-registered rule required the pruned model to beat a random keep-set of the same size on kl32. Over all domains it did not reach significance (paired difference -0.1014, interval -0.2196 to +0.0400), so this release FAILED its gate as originally registered. The cause is multilingual alone. After seeing that result, the owner amended the rule to exclude multilingual, whose weakness is already documented above. Without it the difference is -0.2144 (interval -0.2633 to -0.1676). Reason: Owner decision AFTER the first s10 final result (confirm split). As registered, mxfp4_A vs random0 paired d_kl32 was -0.1014 [-0.2196, +0.0400]: not significant, entirely because of the multilingual bucket (kl32 1.534 vs random 0.894), the weakness the owner had already accepted on 2026-10-22 with a lowered top-1 floor. In every other bucket the keep-set has roughly half of random's kl32. The as-registered comparison stays in results.json and on the cards beside the amended one; no other threshold changes (thorA top-1 0.78 floor still applies).

**Thor build: top-1 floor waived (2026-10-24, after the result).** The low-bit Thor build scored top-1 0.7756 (interval 0.7179 to 0.8284) against a pre-registered floor of 0.78, so it FAILED that floor; the MXFP4 master scores 0.7980. The owner waived this one clause for the Thor build. Every other clause still applies and passed. Reason: Owner decision after the s10 final result. thorA top-1 0.7756 [0.7179, 0.8283] misses the 0.78 sanity floor (borrowed from GLM REAP50 v2's 0.842) by 0.0044, well inside its interval. It is the only 192-expert build that fits Thor; the bit-exact 128-expert alternative is far worse (top-1 0.7092, paired kl32 +0.283 vs thorA). Requantization costs kl32 +0.079 vs the MXFP4 master (limit 0.25) and top-1 -0.022. Every other clause still applies.

`control` keeps all experts and is re-run through the same evaluation machinery as the others. It measures that machinery, not the model.

This build is the `thorA` arm (numbers from `s10/final/results.json`):

| arm | kl32 (95% CI) | top-1 agreement | dNLL | confident-flip rate |
|---|---|---|---|---|
| thorA | 0.5194 [0.3261, 0.7423] | 0.7756 [0.7179, 0.8284] | 0.3091 [0.1426, 0.5119] | 0.0619 |

By domain:

| domain | tokens | kl32 | top-1 agreement | dNLL |
|---|---|---|---|---|
| agentic | 40955 | 0.2265 [0.1998, 0.2526] | 0.8456 [0.8369, 0.8558] | 0.0916 [0.0623, 0.1168] |
| code | 24573 | 0.0607 [0.0403, 0.0937] | 0.9687 [0.9569, 0.9759] | 0.0439 [0.0277, 0.0698] |
| finance | 24573 | 0.4582 [0.2725, 0.7006] | 0.7440 [0.6859, 0.7810] | 0.2803 [0.1667, 0.4332] |
| general | 32764 | 0.5545 [0.4610, 0.6687] | 0.7523 [0.7078, 0.7882] | 0.1475 [-0.1085, 0.4034] |
| math | 8191 | 0.1509 [0.1509, 0.1509] | 0.9294 [0.9294, 0.9294] | 0.0894 [0.0894, 0.0894] |
| multilingual | 24573 | 1.7097 [1.4440, 1.9126] | 0.4935 [0.4415, 0.5412] | 1.3697 [1.0515, 1.6590] |
| science | 24573 | 0.4466 [0.3384, 0.6436] | 0.7401 [0.7089, 0.7707] | 0.3136 [0.2241, 0.4637] |
| vision | 5363 | 0.3663 [0.3278, 0.4002] | 0.8646 [0.8556, 0.8723] | -0.2403 [-0.3219, -0.1661] |

**CUDA correctness gates (layer by layer against the torch reference, [`cuda/tests/gate_v41_golden.cu`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/cuda/tests/gate_v41_golden.cu)).** Source checkpoint: 33 of 41 teacher-forced next-token disagreements were near-ties (80.5%), FAIL as registered (rule: at least 90%); passed under an owner amendment made before its diagnostic was read, for chained bf16 drift ([`ops/decisions/2026-10-24-golden-src-drift-amendment.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/decisions/2026-10-24-golden-src-drift-amendment.md)). This build: 41 of 46 teacher-forced next-token disagreements were near-ties (89.1%), FAIL as registered (rule: at least 90%); the owner waived it and released the build on the serving gates (Engram row store, decode determinism, 1M context, server) ([`ops/decisions/2026-10-25-golden-a192-drift-amendment.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/decisions/2026-10-25-golden-a192-drift-amendment.md), which also records the per-row diagnostics). Every hidden-state layer check and the agreement and final-cosine rules passed in both runs.

**Release gate:** PASS under the gate as amended 2026-10-24 (FAIL as originally registered; see the evaluation). The repository is public only if the pre-registered gate for this arm passed and the build fits the budget above.

---

## How to run it

The server is [`patrickbdevaney/deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server):

1. Download this repository. It includes `tokenizer.json` and the NVFP4 dense overlay (`overlay_nvfp4/`).
2. Build the Engram row files from the two base-model Engram shards (below), with [`ops/s09_engram_rowfile.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_engram_rowfile.py) `build`, then `verify --full`, from the pipeline repo. The row files are about the size of the two shards, and they must sit on a local NVMe drive, because they are read with O_DIRECT for every token. The llama.cpp GGUF uses the same files.
3. Build and start:

```bash
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor --local-dir ~/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor
ln -s /nvme/engram_rows ~/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor/engram_rows     # or set ENGRAM_DIR
git clone https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server && cd deepseek-v4.1-flash-cuda-server
scripts/build_v41.sh v41_server          # CUDA 13, sm_110a, liburing
./start.sh                               # http://127.0.0.1:8000, 32K context (SEQMAX=...), NVFP4 overlay on
```

The server provides OpenAI-style `/v1/chat/completions` (streaming, tool calls in the model's DSML encoding, thinking blocks), `/v1/completions`, `/health`, `/metrics` and a built-in web UI. `start.sh` refuses to start beside another model-loading process, because Thor's memory is unified.

---

## Speculative decoding

The checkpoint's own DSpark/MTP head (`mtp-*.safetensors`) is wired in the server (`--spec K`), but it is off by default. It measured about 1.6x on math-like text and about 0.98x on chat, so it is not a net win for general use yet. In llama.cpp, the GGUF's DSpark drafter gives 1.33–1.61x on chat.

---

## Engram tables (required, not included)

- `model-00047-of-00048.safetensors` (101.5 GB, sha256 `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5`)
- `model-00048-of-00048.safetensors` (101.5 GB, sha256 `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd`)

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00047-of-00048.safetensors model-00048-of-00048.safetensors --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir .
```

---

## 🔎 License & Provenance Ledger

| Claim / Artifact | Primary Verification Source | Digest / Revision |
|---|---|---|
| **License** | [`LICENSE`](./LICENSE) *(MIT, copied verbatim from base model)* | MIT |
| **Upstream Teacher** | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| **Master Pruned Checkpoint** | [`patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) | Keep-set `1ace9b33134a2ca7..` |
| **Per-Shard Manifest & Checksums** | [`thor.json`](./thor.json) & [`reap_keepset.json`](./reap_keepset.json) *(in this repo)* | `thor-v1` (`106.9 GiB`) |
| **GPTQ Error-Feedback Quantizer** | [`ops/s07_quant.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s07_quant.py) & [`cuda/FORMAT_PROPOSAL.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/cuda/FORMAT_PROPOSAL.md) | `2.828 bpw` routed experts |
| **CUDA Golden Correctness Gates** | [`cuda/tests/gate_v41_golden.cu`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/cuda/tests/gate_v41_golden.cu), [`ops/decisions/2026-10-24-golden-src-drift-amendment.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/decisions/2026-10-24-golden-src-drift-amendment.md), [`ops/decisions/2026-10-25-golden-a192-drift-amendment.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/decisions/2026-10-25-golden-a192-drift-amendment.md) | Commit [`36bd04c`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/commit/36bd04c) |
| **Pre-Registered Gates & Eval Results** | [`ops/s10_gates.json`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s10_gates.json), [`DECISIONS.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/DECISIONS.md), [`RESULTS.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/RESULTS.md) | `f95b26721124c6a343f41b004b8740bfe6eb22033d77dcc260dceda039082e0a` |
