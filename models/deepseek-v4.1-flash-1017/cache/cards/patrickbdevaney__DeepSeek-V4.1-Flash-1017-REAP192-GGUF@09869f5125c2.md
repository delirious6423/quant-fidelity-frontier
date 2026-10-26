---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - zh
tags:
  - gguf
  - llama-cpp
  - deepseek
  - deepseek-v4.1-flash
  - moe
  - reap
  - expert-pruning
  - hope
  - iquant
  - iq2_xs
  - iq3_xxs
  - imatrix
  - engram
  - mhc
  - edge-ai
  - jetson-thor
extra_gated_ReadMe_metadata:
  keep_set_sha256: 1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec
---

<div align="center">

# DeepSeek-V4.1-Flash-1017 — REAP-192 (GGUF `L3` Split Shards)

[![Base Model](https://img.shields.io/badge/Base_Model-DeepSeek--V4.1--Flash--1017-blue)](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
[![Master Repo](https://img.shields.io/badge/Master_Checkpoint-REAP192--MXFP4-blueviolet)](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4)
[![Pipeline Repo](https://img.shields.io/badge/GitHub-deepseek--v4.1--flash--reap--thor-black?logo=github)](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor)
[![Format](https://img.shields.io/badge/Format-3--Shard_GGUF_L3_(85.55_GiB)-green)](#why-l3-is-3-sharded-gguf-files--how-to-load-them-in-llamacpp)
[![Thor Fit](https://img.shields.io/badge/128GB_Thor-87.3_GiB_Used_(~30.4_GiB_Free)-76b900)](#memory-kv-context-consumption--headroom-on-a-128-gb-device)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**85.55 GiB 3-shard GGUF (`L3`: 2.46 bpw imatrix I-quant routed experts + `Q8_0`/`Q6_K`/`BF16`/`F32` dense path) of 50%-pruned `deepseek-ai/DeepSeek-V4.1-Flash-1017` (192 of 384 routed experts/layer) — running 100% GPU-offloaded (`-ngl 999`, 11.86 tok/s) with ~30.4 GiB free headroom on a single 128 GB NVIDIA Jetson AGX Thor.**

| Sibling Artifact | Format | Size | Target Hardware / Runtime |
|---|---|---:|---|
| **[`DeepSeek-V4.1-Flash-1017-REAP192-MXFP4`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4)** | Native MXFP4 + FP8 (`safetensors`) | 146.4 GiB | Bit-exact 192-expert master checkpoint |
| **[`DeepSeek-V4.1-Flash-1017-REAP192-Thor`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor)** | `thor-v1` (2.828 bpw K-quant + FP8/BF16) | 106.9 GiB | Single 128 GB NVIDIA Jetson AGX Thor (`v41_server`, 8.09 tok/s, 1M ctx) |
| **[`DeepSeek-V4.1-Flash-1017-REAP192-GGUF`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF)** *(this repo)* | 3-shard GGUF `L3` (`IQ2_XXS`–`IQ3_XXS` + `Q8_0`/`Q6_K`) | **85.55 GiB** | `llama.cpp` (`PR #28696` + 6 patches, `-ngl 999`, **11.86 tok/s** on 128 GB Thor) |
| **[`patrickbdevaney/deepseek-v4.1-flash-reap-thor`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor)** | Pipeline, `llama.cpp` patches & CUDA source | — | Full build pipeline (`ops/s01`..`s12`), `llamacpp/patches/`, and gates |

</div>

---

This is [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`1bae0e1088993b9b82de08f59de5be5a34e42170`) pruned to **192 of 384 routed experts per decoder layer** (keep-set `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec`), quantized once from the lossless GGUF of the [MXFP4 master](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) into the **`L3` community GGUF rung (`85.55 GiB` across 3 split shards, `2.62 bpw` overall / `2.46 bpw` routed experts)**. Every file and SHA256 digest in this repository is recorded in [`gguf_manifest.json`](./gguf_manifest.json).

---

## ✨ Highlights

- **Why `L3` (`IQ2_XXS`–`IQ3_XXS` + `Q8_0`/`Q6_K`) Instead of Uniform `Q4_K_M`:** Upstream `DeepSeek-V4.1-Flash-1017` ships its routed experts natively in **4-bit OCP MXFP4 (`E2M1` + `E8M0`, 4.25 bpw)**. Re-quantizing an FP4 floating-point grid into a 4-bit affine grid (`Q4_K_M`, `IQ4_XS`) costs *more* bytes (`~150+ GiB`, exceeding 128 GB RAM) while losing information across incompatible grids. Instead, [`ops/s09_gguf_ladder.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_gguf_ladder.py) keeps the entire dense/attention path at **`Q8_0` / `Q6_K` / `BF16` / `F32`** and allocates imatrix-guided I-quants (**`IQ3_XXS`** on protected layers `0, 1, 2, 8, 14, 20, 37–39`; **`IQ2_XS` / `IQ2_S` / `IQ2_XXS`** on remaining layers) so all 40 decoder layers fit **100% in GPU memory (`-ngl 999`)** on a 128 GB device.
- **30+ GiB of Free Headroom on a 128 GB Box (`87.30 GiB` Total Resident at 8K Context):** On a 128 GB NVIDIA Jetson AGX Thor (`117.66 GiB` usable UMA memory), loading all 41/41 layers (`-ngl 999`) + an 8,192-token KV cache (`360.03 MiB`) + compute buffers (`1,429.10 MiB`) consumes **`87.30 GiB` total**, leaving **`~30.36 GiB` of free unified memory** while streaming the external 188.8 GiB Engram tables from NVMe via `O_DIRECT` (`0 GiB` RAM).
- **Drop-In 3-Shard Loading in `llama.cpp`:** Split at 45 GB per file via `llama-gguf-split` Solely to stay under Hugging Face's 50 GB per-file upload limit. Pass **only shard `00001`** (`-m DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf`) to `llama-server` or `llama-cli`—`llama.cpp` automatically discovers and maps shards `00002` and `00003`.
- **11.86 tok/s Decode & 19.54 tok/s Prompt Eval on Jetson AGX Thor:** Verified end-to-end with `llama-server` (`-ngl 999`, `O_DIRECT` NVMe Engram row store) and evaluated across **180,224 held-out confirmation tokens** (`kl32 = 0.5076 [0.3224, 0.7337]`, `top-1 = 0.7725 [0.7116, 0.8287]`).

---

## 📋 Model Summary

| Property | Value |
|---|---|
| **Base Model** | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (`1bae0e1088993b9b82de08f59de5be5a34e42170`) |
| **Master Pruned Checkpoint** | [`patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) (`reap_keepset.json` `1ace9b33134a2ca7`) |
| **Architecture (`general.architecture`)** | `deepseek41` (40 layers, `n_embd = 5120`, 192 routed experts/layer, top-6 active + 1 shared expert, 4-stream mHC, Engram on L1 & L14) |
| **GGUF Ladder Level** | **`L3`** (`2.4594 bpw` routed experts; `2.62 bpw` overall across `280.05B` params; `1,044` tensors across 3 split files) |
| **Routed Expert Quantization** | Protected layers (`0, 1, 2, 8, 14, 20, 37, 38, 39`): **`IQ3_XXS`** (`27` tensors); Other layers: **`IQ2_XS`** (`79`), **`IQ2_XXS`** (`12`), **`IQ2_S`** (`2`) |
| **Dense & Attention Quantization** | Attention/compressor/indexer/`engram_wkv`: **`Q8_0`** (`229`); Shared expert/`token_embd`/`output`: **`Q6_K`** (`122`); Router: **`BF16`** (`44`); mHC/norms/sinks: **`F32`** (`529`) |
| **Total GGUF Size (3 Shards)** | **`85.55 GiB`** (`91,862,148,960` bytes: `41.59 GiB` + `41.51 GiB` + `2.46 GiB`; see [`gguf_manifest.json`](./gguf_manifest.json)) |
| **128 GB Thor Memory & Speed** | **`87.30 GiB` total resident** at `-c 8192` (`~30.36 GiB` free headroom); **`11.86 tok/s` decode**, **`19.54 tok/s` prompt eval** |
| **License** | MIT ([`LICENSE`](./LICENSE)) |

---

## Why `L3` Is 3 Sharded GGUF Files & How to Load Them in `llama.cpp`

### 1. Why `L3` (I-Quant Mix) Instead of a Uniform `Q4_K_M` or `IQ4_XS`?
If you are used to downloading a single `Q4_K_M.gguf` or `IQ4_XS.gguf`, `DeepSeek-V4.1-Flash-1017` is different for two architectural reasons ([`ops/s09_gguf_ladder.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_gguf_ladder.py)):
1. **The upstream base model is already native 4-bit FP4 (`MXFP4`, 4.25 bpw):** Re-quantizing an OCP MXFP4 (`E2M1` + `E8M0`) floating-point grid into a 4-bit affine integer grid (`Q4_K_M` at ~4.8 bpw or `IQ4_XS` at ~4.25 bpw) spends *more* bytes (`~145–155 GiB` — too large for a 128 GB device!) while still suffering cross-grid rounding loss.
2. **To run 100% GPU-offloaded (`-ngl 999`) on a 128 GB unified-memory device, the 271.8B routed-expert parameters must sit at `~2.46–2.85 bpw` while keeping the dense path near-lossless:**
   [`ops/s09_gguf_ladder.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_gguf_ladder.py) defines a 4-rung ladder (`L0` lossless `MXFP4` ~145 GiB; `L1` `3.40 bpw` ~118 GiB; `L2` `2.85 bpw` ~101 GiB; **`L3` `2.46 bpw` `85.55 GiB`**) quantized directly from the lossless source GGUF using our keep-set-matched activation importance matrix (`v41-1017-reap192.imatrix.dat`, `240` entries):
   - **Never-compromised dense path (`~10 GiB`):** All MLA attention, KV compressor, DSA lightning-indexer, and `engram_wkv` weights stay at **`Q8_0`** (`229` tensors); shared experts, token embeddings, and `output.weight` stay at **`Q6_K`** (`122` tensors); router gates (`ffn_gate_inp`) stay at **`BF16`** (`44` tensors); and all 4-stream hyper-connection mixers (`hc_*`), RMSNorms, attention sinks, and biases stay at **`F32`** (`529` tensors).
   - **Sensitivity-ranked routed experts (`2.4594 bpw` average across 120 expert tensors):** Protected architectural layers — **Layers `0, 1`, Engram layers `1, 14`, KV-source layers `2, 8, 14, 20`, and DSpark tap layers `37, 38, 39`** — are locked at **`IQ3_XXS` (`3.0625 bpw`, 27 tensors)**. The remaining MoE layers are promoted one step at a time by layer saliency (`down_proj` weighted $1.1\times$ above `gate`/`up`) into **`IQ2_S` (`2.50 bpw`, 2 tensors)**, **`IQ2_XS` (`2.3125 bpw`, 79 tensors)**, and **`IQ2_XXS` (`2.0625 bpw`, 12 tensors)**.

### 2. Why Are There 3 `.gguf` Shards and What Is `gguf_manifest.json`?
Hugging Face Hub enforces a **50 GB per-file upload limit**. Because the complete `L3` model is **85.55 GiB** (`91.86 GB`), it was split with standard `llama-gguf-split --split --split-max-size 45G` into three shards:

| Shard File | Size (GiB) | Exact Bytes | SHA256 (from [`gguf_manifest.json`](./gguf_manifest.json)) |
|---|---:|---:|---|
| **[`DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf`](./DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf)** | **41.59 GiB** | `44,657,469,792` | `8bb8c4a376c37b3cdafdd7777f36f24d54a3b4e4a04df313e2c84606bf894cb4` |
| **[`DeepSeek-V4.1-Flash-1017-REAP192-L3-00002-of-00003.gguf`](./DeepSeek-V4.1-Flash-1017-REAP192-L3-00002-of-00003.gguf)** | **41.51 GiB** | `44,568,604,928` | `dbc8627d152faa3f27e2fc149d8fa12869bdeaeebc239b54ff7e97f65c7b4d1f` |
| **[`DeepSeek-V4.1-Flash-1017-REAP192-L3-00003-of-00003.gguf`](./DeepSeek-V4.1-Flash-1017-REAP192-L3-00003-of-00003.gguf)** | **2.46 GiB** | `2,636,074,240` | `ac75352fc05d5295cc55ff2ddf2abc28d2a461b34372d67d9a393b179b03aff5` |
| **Total (`L3`, `split.count = 3`, `1,044` tensors)** | **85.55 GiB** | **`91,862,148,960`** | Provenance & eval recorded in **[`gguf_manifest.json`](./gguf_manifest.json)** |

- **How `llama.cpp` loads the 3 shards:** Place all three `.gguf` files in the same directory and pass **only the first file (`-00001-of-00003.gguf`)** to `-m`. You do **not** need to concatenate/merge the files, and you do **not** pass `gguf_manifest.json` to `llama.cpp`. When `llama-server` or `llama-cli` opens `DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf`, it reads `split.count = 3` from the GGUF header and automatically loads `-00002-of-00003.gguf` and `-00003-of-00003.gguf`.
- **What [`gguf_manifest.json`](./gguf_manifest.json) is:** A machine-readable audit file generated by [`ops/s09_gguf_ladder.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_gguf_ladder.py) that records the SHA256 and byte size of each split, the dense tensor precision table, the keep-set SHA256, the exact `llama.cpp` PR and 6 patch filenames, and the 180,224-token confirmation evaluation metrics.

---

## Memory, KV Context Consumption & Headroom on a 128 GB Device

### Why the KV Cache Is Only `360 MiB` at 8K Context (`720 MiB` at 16K Context)
Unlike standard dense or GQA models whose KV cache grows by gigabytes per thousand tokens, `DeepSeek-V4.1-Flash-1017` uses **Multi-head Latent Attention (MLA)** combined with **Cross-Layer KV Sharing** ([`0002-deepseek41-reader-layers-reuse-their-index-source-s-.patch`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/tree/main/llamacpp/patches)):
- Only **4 KV-source layers (`layers 2, 8, 14, 20`)** allocate compressed CSA latent KV caches (`ratio = 2`, `32.00 MiB` at 8K) and lightning-indexer KV caches (`8.00 MiB` at 8K).
- The **30 reader layers (`3–7, 9–13, 15–19, 21–39`)** reuse their source layer's compressed KV cache and top-k indices at **0 MiB of additional KV memory**.
- All 40 layers maintain a compact 128-token Sliding-Window Attention (`SWA`, `n_embd_head_k_all = 512`) buffer (`320.00 MiB` at `-c 8192`).
- Total KV cache consumption is therefore just **`45 KiB per token`** (**`360.03 MiB` at `-c 8192`**, or **`720.03 MiB` at `-c 16384`**).

### Measured Memory Budget on a 128 GB NVIDIA Jetson AGX Thor (`-ngl 999`, All 41/41 Layers on GPU)
Every figure below is taken directly from the logged `llama.cpp` allocation trace on a **128 GB NVIDIA Jetson AGX Thor** (`120,479 MiB` = **`117.66 GiB` usable unified memory** reported by `ggml_backend_cuda_get_available_uma_memory` after OS/kernel reserve):

| Memory Component | `-c 8192` (Default Eval/Serve) | `-c 16384` (Fork Context Ceiling) | Allocation Details (from `llama.cpp` loader log) |
|---|---:|---:|---|
| **Model Weights (`-ngl 999`, 41/41 layers)** | **`85.55 GiB`** (`87,601.0 MiB`) | **`85.55 GiB`** (`87,601.0 MiB`) | `87,083.04 MiB` on `CUDA0` + `517.82 MiB` on `CUDA_Host` + `0.16 MiB` CPU |
| **SWA KV Cache (40 layers, `F16`)** | **`0.31 GiB`** (`320.0 MiB`) | **`0.63 GiB`** (`640.0 MiB`) | `n_embd_head_k_all = 512` across layers `0..39` |
| **DSV4 CSA Compressed KV (4 source layers, `F16`)** | **`0.03 GiB`** (`32.0 MiB`) | **`0.06 GiB`** (`64.0 MiB`) | Allocated on source layers `2, 8, 14, 20`; reused by layers `3–7, 9–13, 15–19, 21–39` |
| **DSV4 Lightning-Indexer KV + Comp State** | **`0.01 GiB`** (`8.03 MiB`) | **`0.02 GiB`** (`16.03 MiB`) | `8.00 MiB` indexer K (`n_embd_head_k_all = 128`) + `0.03 MiB` CSA/HCA/LID state |
| **Compute & Output Buffers (`-b 2048 -ub 512`)** | **`1.40 GiB`** (`1,429.6 MiB`) | **`~1.45 GiB`** (`~1,485 MiB`) | `1,291.01 MiB` `CUDA0` compute + `138.09 MiB` `CUDA_Host` compute + `0.49 MiB` output |
| **Engram Row Store (`engram_L1` & `engram_L14`)** | **`0.00 GiB` RAM** | **`0.00 GiB` RAM** | `188.8 GiB` on local NVMe, read per token via 16-thread `O_DIRECT` (`1.13–1.30 ms p50`) |
| **Total Active Memory Consumption** | **`87.30 GiB`** (`89,390.6 MiB`) | **`87.71 GiB`** (`89,810.6 MiB`) | **100% of decoder layers (`41/41`) offloaded to GPU** |
| **Free Headroom on 128 GB Thor (`117.66 GiB` UMA)** | **`+30.36 GiB` free** | **`+29.95 GiB` free** | **`+40.70 GiB` below physical 128 GiB** (`31,135,980 KB` free UMA after weight load) |

- **Measured Throughput on 128 GB Jetson AGX Thor (`llama-server`, `-ngl 999`, `n_threads = 14`, `O_DIRECT` NVMe Engram):**
  - **Prompt evaluation:** **`19.54 tok/s`** (`51.18 ms/token`)
  - **Token generation (decode):** **`11.86 tok/s`** (`84.32 ms/token`, greedy `temperature = 0`)

---

## What Is in This Repository

### Quantization Levels

Every level is quantized once from the lossless GGUF of the MXFP4 master ([`ops/s09_gguf_ladder.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_gguf_ladder.py)). No level is made from another level. The keep-set fingerprint (`1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec`) is checked against the master before conversion, against the imatrix before quantization, and on the first split of every level before upload.

| level | expert bpw | size (GiB) | kl32 (95% CI) | top-1 agreement | dNLL | GPU layers in the eval |
|---|---|---|---|---|---|---|
| L3 | 2.46 | 85.6 | 0.5076 [0.3224, 0.7337] | 0.7725 [0.7116, 0.8287] | 0.3399 [0.1758, 0.5434] | all |

Every level was scored fully offloaded to the GPU.

### Non-Routed-Expert Tensors (Fixed Across All Levels)

| tensors | precision |
|---|---|
| attention, compressor, indexer, engram_wkv | Q8_0 |
| shared expert, token_embd, output | Q6_K |
| router (ffn_gate_inp) | BF16 (as stored) |
| hyper-connection | F32 |
| norms, sinks, biases | F32 |
| engram tables | not in the GGUF (O_DIRECT row files, [`llamacpp/ENGRAM_ROWFILE.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/ENGRAM_ROWFILE.md)) |

No heal gain is applied: the files carry no `deepseek41.reap.routed_gain` key, so the routed experts' summed output is used exactly as converted from the unhealed MXFP4 master.

---

## The Keep-Set

- **Calibration corpus ([`ops/corpus_spec.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/corpus_spec.py), [`ops/s02_calib_corpus.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s02_calib_corpus.py)):** `4,975,198` tokens in 8k sequences, plus a long-context pool of 32k–128k rows, token-balanced across general, multilingual, agentic, code, vision, math, science, finance, and long-context text (corpus spec `c1803e70af7db162`). Executed layer-by-layer through the teacher with Engram switched on ([`ops/s03_saliency.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s03_saliency.py)). Only human-written or open-model text was used; no Claude-written text.
- **Ranking & specialist protection ([`ops/s04_select_arbitrate.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s04_select_arbitrate.py)):** Candidate keep-sets were ranked by **HOPE** (pairwise co-activation with the REAP-sum diagonal; [arXiv:2609.18916](https://arxiv.org/abs/2609.18916)), REAP-sum ([arXiv:2510.13999](https://arxiv.org/abs/2510.13999)), routing mass, frequency, and random anchors.
  - Each domain pool — and image tokens ranked separately on their own routing (`gate.bias_vl`) — protected its top experts at the measured protect-fraction knee (`0.06`).
  - Super-Experts and position-sensitive / attention-sink experts were never pruned ([`ops/drift_guard.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/drift_guard.py)).
- **Held-out KL arbitration:** Arbitrated by held-out teacher top-32 KL on the screening split (`winner: hope_sum_pf0`, keep-set sha256 `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec`). The full per-layer list of surviving expert IDs is stored in the GGUF header (`deepseek41.reap.keep_ids`, `7,680` ints = `40 × 192`) and in [`reap_keepset.json`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4/blob/main/reap_keepset.json) on the master repo.

---

## How It Was Evaluated

### 1. GGUF Confirmation Evaluation (`s09`, 180,224 Tokens in `llama.cpp`)
The `L3` GGUF table above (`kl32 = 0.5076 [0.3224, 0.7337]`, `top-1 = 0.7725 [0.7116, 0.8287]`, `dNLL = 0.3399 [0.1758, 0.5434]`, recorded in [`gguf_manifest.json`](./gguf_manifest.json)) was measured by [`llamacpp/tools/v41-teacher-kld.cpp`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/tools/v41-teacher-kld.cpp) and [`ops/s09_teacher_eval.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_teacher_eval.py) on the **22 text-only confirmation rows** (`180,224` tokens, each `<= 8,192` tokens) from the pre-registered confirmation split, against the unpruned teacher (`deepseek-ai/DeepSeek-V4.1-Flash-1017` at `1bae0e1088993b9b82de08f59de5be5a34e42170`) with a tail bucket and 95% row-bootstrap confidence intervals.
- **GGUF release gate status ([`gguf_manifest.json`](./gguf_manifest.json)):** FAIL on the pre-registered floor (`top-1 0.7725 < 0.78`); `L3` released by owner decision (`llama-server` smoke passed; [`ops/s09_owner_release.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_owner_release.py)).

### 2. Family-Wide Confirmation Evaluation (`s10/final/results.json`, 185,565 Tokens Across All 26 Confirmation Documents)
Every number below comes from `s10/final/results.json` of the build pipeline ([`ops/s10_eval.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s10_eval.py), summarized in [`RESULTS.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/RESULTS.md)). It is a paired, teacher-forced comparison against the unpruned `deepseek-ai/DeepSeek-V4.1-Flash-1017` at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, run in its own MXFP4/FP8 numerics. The comparison uses **185,565 scored tokens** from **26 held-out documents** in the confirmation split (the 22 text rows above plus the 4 vision caption rows). That split was never used to choose the model. The intervals are ninety-five percent document-cluster bootstrap intervals.

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

---

## How to Run It in `llama.cpp`

### Step 1: Build the Patched `llama.cpp` Fork (`PR #28696` + 6 Patches)
Upstream `llama.cpp` [PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696) (`2076412598e99127ea2554467b7530856895d3`) adds the `deepseek41` graph, and [`llamacpp/patches/`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/tree/main/llamacpp/patches) (`fork_head` `92031e6a4add77f9d8b09702296b64ea484a2132`, automated by [`llamacpp/setup.sh`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/setup.sh) and [`llamacpp/build.sh`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/build.sh)) applies six required patches:
- `0001-deepseek41-keep-hyper-connection-tensors-F32-when-qu.patch` — keeps `*.hc_*` and `output_hc_*` in `F32` (80 Sinkhorn iterations per token).
- `0002-deepseek41-reader-layers-reuse-their-index-source-s-.patch` — reader layers reuse their KV-source layer's top-k indices (`2, 8, 14, 20, 24, 28, 32, 36`).
- `0003-deepseek41-Engram-O_DIRECT-row-store-input-LLAMA_ENG.patch` — reads `engram_L1.v41rows` and `engram_L14.v41rows` per token via 16-thread aligned `O_DIRECT` `pread` when `LLAMA_ENGRAM_ROWFILE_DIR` is set.
- `0004-deepseek41-REAP-heal-gain-on-the-routed-expert-sum-d.patch` — optional `deepseek41.reap.routed_gain` support.
- `0005-deepseek41-export-the-V4.1-DSpark-drafter-as-a-dflas.patch` — exports the V4.1 DSpark drafter (`--spec-type draft-dspark`).
- `0006-deepseek41-clamp-n_ctx_seq-to-16384-until-the-candid.patch` — clamps `n_ctx_seq` to `16,384` until the two-level (`2048×8`) candidate pass is ported.

```bash
git clone https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor.git
cd deepseek-v4.1-flash-reap-thor
bash llamacpp/setup.sh
CUDA_ARCH=110a bash llamacpp/build.sh   # Set CUDA_ARCH=89 / 90 / 100 / 120 for other GPUs
```

### Step 2: Download the Two Base Engram Shards & Build the NVMe Row Store
The two Engram tables (`layers.1.engram.embed` and `layers.14.engram.embed`, 101.5 GB each) are unchanged by REAP pruning and are **not** inside the GGUF. Download shards 47 and 48 from the base model and convert them once into `.v41rows` files on a **local NVMe drive** ([`llamacpp/ENGRAM_ROWFILE.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/ENGRAM_ROWFILE.md), [`ops/s09_engram_rowfile.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_engram_rowfile.py)):

- `model-00047-of-00048.safetensors` (101.5 GB, sha256 `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5`)
- `model-00048-of-00048.safetensors` (101.5 GB, sha256 `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd`)

```bash
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 \
  model-00047-of-00048.safetensors model-00048-of-00048.safetensors \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir /path/to/base-engram

python3 ops/s09_engram_rowfile.py build --src /path/to/base-engram --out /nvme/engram_rows
python3 ops/s09_engram_rowfile.py verify --full --out /nvme/engram_rows
```

### Step 3: Download the 3 GGUF Shards & Launch `llama-server` (Pass Shard `00001` Only)
Download all three `.gguf` shards into one directory and point `-m` at `DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf`:

```bash
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF --local-dir /nvme/v41-reap192-gguf

LLAMA_ENGRAM_ROWFILE_DIR=/nvme/engram_rows \
  ./llamacpp/llama.cpp/build-thor/bin/llama-server \
  -m /nvme/v41-reap192-gguf/DeepSeek-V4.1-Flash-1017-REAP192-L3-00001-of-00003.gguf \
  -c 8192 -np 1 -ngl 999 --jinja \
  --host 127.0.0.1 --port 18089
```

Both `eos_token_id` and `eot_token_id` are already set to `1` (`<｜end▁of▁sentence｜>`) in the GGUF header, and the DeepSeek-V4.1 Jinja chat template is embedded in `tokenizer.chat_template`.

---

## Known Limitations

- **Text-only in `llama.cpp`:** Vision (`mmproj`, `gate.bias_vl`) is not yet ported to the `deepseek41` `llama.cpp` graph.
- **16,384 context ceiling in `llama.cpp`:** Practical context is clamped to **16,384 tokens** (`0006-deepseek41-clamp-n_ctx_seq-to-16384-until-the-candid.patch`) until the two-level (`2048×8`) DSA candidate pass is ported.
- **Multilingual degradation:** As documented in the evaluation section above, 50% expert pruning degrades multilingual top-1 agreement (`0.5247` on the MXFP4 master); prefer the unpruned model for non-English/Chinese multilingual workloads.

---

## 🔎 Provenance & Verification Ledger

Every claim, metric, and weight tensor in this repository is traceable to a primary artifact or source file:

| Claim / Artifact | Primary Verification Source | Digest / Revision |
|---|---|---|
| **Upstream Base Model** | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) | Commit `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| **GGUF Shards, Precision Map & `s09` KLD Eval** | [`gguf_manifest.json`](./gguf_manifest.json) *(in this repo)* | Shards: `8bb8c4a3..`, `dbc8627d..`, `ac75352f..` |
| **Keep-Set Indices (192 / 384 Experts)** | `deepseek41.reap.keep_ids` *(in GGUF header)* & [`reap_keepset.json`](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4/blob/main/reap_keepset.json) | SHA256 `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec` |
| **Calibration Corpus Specification** | [`ops/corpus_spec.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/corpus_spec.py) & [`ops/s02_calib_corpus.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s02_calib_corpus.py) | Spec hash `c1803e70af7db162` (`4,975,198` tokens) |
| **Pre-Registered Release Gates & Amendments** | [`ops/s10_gates.json`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s10_gates.json), [`DECISIONS.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/DECISIONS.md), [`RESULTS.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/RESULTS.md) | Gates SHA256 `f95b26721124c6a343f41b004b8740bfe6eb22033d77dcc260dceda039082e0a` |
| **GGUF Ladder Planner & Converter** | [`ops/s09_gguf_ladder.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_gguf_ladder.py), [`llamacpp/convert_v41_reap.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/convert_v41_reap.py), [`llamacpp/README.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/README.md) | Pipeline commit [`36bd04c`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/commit/36bd04c) |
| **Patched `llama.cpp` Runtime** | [Upstream `llama.cpp` PR #28696](https://github.com/ggml-org/llama.cpp/pull/28696) + [`llamacpp/patches/0001..0006`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/tree/main/llamacpp/patches) | PR head `20764125`, patched tree `92031e6a` |
| **Pruning Algorithms** | [REAP (arXiv:2510.13999)](https://arxiv.org/abs/2510.13999) · [HOPE (arXiv:2609.18916)](https://arxiv.org/abs/2609.18916) | `winner: hope_sum_pf0` |
| **License** | [`LICENSE`](./LICENSE) *(MIT, copied verbatim from base model)* | MIT |
