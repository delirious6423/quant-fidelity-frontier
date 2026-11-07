---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - deepseek
  - deepseek-v4.1-flash
  - moe
  - reap
  - hope
  - expert-pruning
  - k-quant
  - nvfp4
  - engram
  - jetson-thor
  - cuda
---

# DeepSeek-V4.1-Flash-1017-REAP192-Thor

This is [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) pruned to 192 of 384 routed experts per layer and requantized to 2.83-bit K-quant experts. It is stored in the `thor-v1` layout for one 128 GB NVIDIA Jetson AGX Thor. The Engram tables stay on NVMe and are read per token, so the 106.9 GiB build fits in unified memory.

It is served by a dedicated pure-CUDA server, [`deepseek-v4.1-flash-cuda-server`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server). The author measured it on 2026-10-30 at **25.3 tok/s** plain decode and 25.1–40.2 tok/s with the model's own DSpark speculation.

> **Release status.** This build **fails its release gate as pre-registered** on two clauses. Its top-1 agreement is 0.7756, against a floor of 0.78. It also does not beat a random keep-set on kl32. It was published on 2026-10-26 under two owner amendments made on 2026-10-24, *after* the result: multilingual was left out of the beat-random test, and the 0.78 top-1 floor was waived for this build. The CUDA layer-by-layer correctness gate also failed as registered (89.1% of disagreements were near-ties, against the 90% rule), and the owner waived it. See [Release gates](#release-gates).

## Model overview

| | |
|---|---|
| Base model | [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) at `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| Made from | the [REAP192-MXFP4](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) keep-set (`reap_keepset.json`, sha256 `1ace9b33134a2ca74174952593682e94edbf3c3348b4ac505ff59597a05d49ec`) |
| Architecture | DeepSeek-V4.1-Flash: 40 decoder layers, hidden 5120, Engram on layers 1 and 14, 4-stream hyper-connections |
| Routed experts | **192 of 384 per layer**, 6 active, plus 1 shared expert |
| Speculative-decoding layers | the 3 DSpark/MTP layers of the base, 128 MXFP4 experts each, **unchanged** (`mtp-0..2.safetensors`) |
| Vision | the base's 32-layer ViT and aligner, BF16, unchanged (`vision.safetensors`) |
| Routed experts, precision | ggml K-quant blocks with GPTQ error-feedback rounding, **2.828 bpw** average, 96,082,329,600 bytes (89.5 GiB). Q3_K for both projection groups in protected layers 0, 1, 2, 8, 14, 20, 37, 38, 39, plus Q3_K `w2` in layers 28, 29, 36; Q2_K elsewhere |
| Dense path, precision | the base checkpoint's own tensors: FP8 E4M3 with E8M0 scales for attention, indexer, compressor and shared expert; BF16 router, embedding and head; F32 norms and hyper-connections |
| Optional NVFP4 dense overlay | `overlay_nvfp4/L00..L39`, 3,668,904,096 bytes (3.42 GiB): an NVFP4 copy of the dense attention and shared-expert weights, served by default |
| Size on disk | build 114,754,314,496 bytes (**106.9 GiB**), plus the 3.42 GiB overlay |
| Context | 1,048,576 in the base config. The server's default ceiling is 32,768 (`SEQMAX`). Quality not evaluated past 8K tokens |
| Heal gain | none (every per-layer gain in `thor.json` is 1.0; the heal was rejected) |
| Licence | MIT, inherited from the base model |

### Files

| file(s) | contents |
|---|---|
| `globals.safetensors` | token embedding, output head, final norm, image marker embeddings (BF16) |
| `layer-00.safetensors` … `layer-39.safetensors` | one decoder layer each: FP8 attention, compressor and indexer, FP8 shared expert, BF16 router sliced to the kept ids, F32 hyper-connections, and K-quant routed experts |
| `mtp-0/1/2.safetensors` | DSpark/MTP layers, copied from the base (128 experts each, MXFP4) |
| `vision.safetensors` | ViT and aligner (BF16) |
| `overlay_nvfp4/L00..L39.safetensors` | NVFP4 dense overlay (optional) |
| `thor.json` | the layout manifest: per-file bytes and sha256, per-tensor dtype, quant type and shape, per-layer expert allocation, Engram table pointers |
| `reap_keepset.json` | kept expert ids per layer, in base numbering |
| `tokenizer.json`, `tokenizer_config.json` | from the base model |

The layout is described in [`cuda/FORMAT_PROPOSAL.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/cuda/FORMAT_PROPOSAL.md). No other runtime reads it.

## Which version should I use

| repository | format | size | runtime | measured decode on Thor (2026-10-30) |
|---|---|---:|---|---|
| **REAP192-Thor** (this repo) | `thor-v1`: K-quant experts, FP8 dense, NVFP4 overlay | 106.9 GiB (+3.4) | [CUDA server](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server), Jetson AGX Thor (sm_110a) only | **25.3 tok/s plain; 25.1–40.2 with DSpark K=3** |
| [REAP192-GGUF](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF) | GGUF L3 + mmproj + DSpark | 85.6 GiB | [llama.cpp branch](https://github.com/patrickbdevaney/llama.cpp/tree/deepseek-v4.1-reap-thor) | 12.3–12.5 tok/s; 17.3–20.7 with DSpark |
| [REAP192-MXFP4](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4) | bit-exact master | 151.6 GiB | none tested; does not fit 128 GB | — |

On a Jetson AGX Thor this build is the recommended one. On other hardware, use the GGUF.

## Quickstart

These steps follow the server's [README](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server). They were run on a Jetson AGX Thor with CUDA 13.0. The server targets `sm_110a` only.

```bash
# 1. this repository (weights, NVFP4 overlay, tokenizer)
hf download patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor --local-dir ~/models/DeepSeek-V4.1-Flash-1017-REAP192-Thor

# 2. Engram row files on local NVMe (see below), at $CKPT/engram_rows or ENGRAM_DIR

# 3. build and start (CUDA 13, sm_110a, liburing)
git clone https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server && cd deepseek-v4.1-flash-cuda-server
scripts/build_v41.sh v41_server
./start.sh               # http://127.0.0.1:8000, 32K context, NVFP4 overlay on, DSpark K=3 on
```

The server provides an OpenAI-style `/v1/chat/completions` endpoint (streaming, tool calls in the model's DSML encoding, thinking blocks, base64 image parts), `/v1/completions`, `/v1/models`, `/health`, `/metrics` and a web UI. The settings `OVERLAY=0` (FP8 dense instead of NVFP4), `SPEC=0` (no speculation) and `SEQMAX` are environment variables. `start.sh` refuses to start beside another model-loading process, because on Thor's unified memory two such processes can hard-reboot the box.

### Engram row files (required, not included)

The two Engram tables are not pruned and are not in this repository. They are read per token with `O_DIRECT` from two `.v41rows` files: 202,758,040,592 bytes (188.8 GiB) together, which must be on a local NVMe drive. You build them once from the base model's shards 47 and 48 with [`ops/s09_engram_rowfile.py`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/s09_engram_rowfile.py) (`build --dst DIR`, then `verify --dst DIR --full`, with `V41_SRC` pointing at a directory holding the base `config.json`, `model.safetensors.index.json` and the two shards). The format is described in [`llamacpp/ENGRAM_ROWFILE.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/llamacpp/ENGRAM_ROWFILE.md). The [GGUF card](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-GGUF#engram-row-files-required-not-included) has the full command sequence. The same files serve both runtimes.

| base file | bytes | sha256 |
|---|---:|---|
| `model-00047-of-00048.safetensors` | 101,535,150,936 | `76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5` |
| `model-00048-of-00048.safetensors` | 101,537,926,640 | `9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd` |

## Evaluation

**Method.** This is a paired, teacher-forced comparison against the unpruned base model on the **confirmation split: 185,565 tokens from 26 held-out documents**, never used for any selection or tuning decision. Intervals are 95% document-cluster bootstrap intervals. The metric definitions are on the [master card](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4#evaluation). Source: `s10/final/results.json` of the build pipeline.

| arm | what it is | kl32 (95% CI) | top-1 (95% CI) | dNLL (95% CI) | confident-flip rate |
|---|---|---|---|---|---|
| `mxfp4_A` | the bit-exact MXFP4 master, same keep-set | 0.4405 [0.2639, 0.6465] | 0.7980 [0.7431, 0.8481] | 0.2718 [0.1219, 0.4585] | 0.0524 |
| **`thorA`** | **this build, FP8 dense path** (`OVERLAY=0`) | **0.5194 [0.3261, 0.7423]** | **0.7756 [0.7179, 0.8284]** | **0.3091 [0.1426, 0.5119]** | **0.0619** |
| **`thorA_overlay`** | **this build with the NVFP4 overlay (server default)** | **0.5376 [0.3420, 0.7623]** | **0.7681 [0.7102, 0.8217]** | **0.3209 [0.1542, 0.5240]** | **0.0640** |
| `b128` | 128 experts, bit-exact MXFP4, at similar expert bytes (rejected) | 0.8026 [0.5627, 1.0801] | 0.7092 [0.6443, 0.7701] | 0.5469 [0.3271, 0.8010] | 0.0995 |
| `random0` | a random 192-expert keep-set | 0.5419 [0.4351, 0.6554] | 0.7507 [0.7088, 0.7925] | 0.2070 [0.0726, 0.3210] | 0.0587 |

**Paired costs:**
- Requantizing the experts (`thorA` vs `mxfp4_A`) costs kl32 +0.0789 [+0.0571, +0.1024] (gate limit 0.25) and 2.2 points of top-1.
- The NVFP4 overlay (`thorA_overlay` vs `thorA`) costs kl32 +0.0182 [+0.0137, +0.0233] (gate limit: upper bound ≤ 0.03) and 0.21 points of confident flips (limit 0.5). It passed its pre-registered gate.
- At similar expert bytes, 192 requantized experts beat 128 bit-exact ones: kl32 −0.2831 [−0.3479, −0.2245], top-1 +6.6 points.

**By domain.** Each domain has only 1 to 5 documents; math is a single document.

| domain | tokens | `thorA` kl32 | `thorA` top-1 | `thorA_overlay` kl32 | `thorA_overlay` top-1 |
|---|---:|---|---|---|---|
| agentic | 40,955 | 0.2265 [0.1998, 0.2526] | 0.8456 [0.8369, 0.8558] | 0.2417 [0.2151, 0.2683] | 0.8350 [0.8283, 0.8429] |
| code | 24,573 | 0.0607 [0.0403, 0.0937] | 0.9687 [0.9569, 0.9759] | 0.0657 [0.0441, 0.1003] | 0.9670 [0.9567, 0.9733] |
| finance | 24,573 | 0.4582 [0.2725, 0.7006] | 0.7440 [0.6859, 0.7810] | 0.4744 [0.2960, 0.7220] | 0.7389 [0.6866, 0.7707] |
| general | 32,764 | 0.5545 [0.4610, 0.6687] | 0.7523 [0.7078, 0.7882] | 0.5784 [0.4726, 0.6896] | 0.7431 [0.6997, 0.7804] |
| math | 8,191 | 0.1509 | 0.9294 | 0.1599 | 0.9270 |
| **multilingual** | 24,573 | **1.7097 [1.4440, 1.9126]** | **0.4935 [0.4415, 0.5412]** | **1.7349 [1.4744, 1.9132]** | **0.4858 [0.4357, 0.5303]** |
| science | 24,573 | 0.4466 [0.3384, 0.6436] | 0.7401 [0.7089, 0.7707] | 0.4709 [0.3695, 0.6581] | 0.7293 [0.7020, 0.7563] |
| vision | 5,363 | 0.3663 [0.3278, 0.4002] | 0.8646 [0.8556, 0.8723] | 0.3958 [0.3580, 0.4309] | 0.8600 [0.8494, 0.8688] |

**What these numbers are, and are not.** `thorA` and `thorA_overlay` are this build's weights, scored by the pipeline's teacher-forced replay harness (`ops/s10_eval.py`), not by the CUDA server. The server was instead checked layer by layer against a torch reference (below). There are no generative benchmarks, no long-context quality past 8K tokens, no tool-calling accuracy and no image-task benchmarks.

### Release gates

| gate | result as registered | resolution |
|---|---|---|
| top-1 ≥ 0.78 (`thor_quant` applies the pruned floor) | **FAIL**: 0.7756 | waived for this build by the owner, 2026-10-24, **after the result** |
| must beat `random0` on kl32 | **FAIL**: interval upper bound +0.1311 | passes once multilingual is excluded (owner amendment, 2026-10-24, **after the result**) |
| requantization cost: d_kl32 vs master ≤ 0.25 | PASS: +0.0789 | — |
| kl32 ≤ 0.8; every domain's top-1 ≥ 0.65 except multilingual (≥ 0.40, set before any result) | PASS | — |
| NVFP4 overlay: d_kl32 upper bound ≤ 0.03, confident flips +≤ 0.005 | PASS: +0.0233, +0.0021 | overlay adopted |
| CUDA golden, source checkpoint: ≥ 90% of disagreements are near-ties | **FAIL**: 33 of 41 (80.5%) | passed under a drift amendment written before its diagnostic was read ([`2026-10-24-golden-src-drift-amendment.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/decisions/2026-10-24-golden-src-drift-amendment.md)). That diagnostic ran on a gate harness with a stream race, fixed later |
| CUDA golden, this build | **FAIL**: 41 of 46 (89.1%) | **waived by the owner** 2026-10-25 ([`2026-10-25-golden-a192-drift-amendment.md`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/blob/main/ops/decisions/2026-10-25-golden-a192-drift-amendment.md)). The routing near-tie hypothesis for the deciding row is untested |
| serving gates: Engram row store, decode determinism, 1M-token context allocation, server smoke | PASS | — |

In the CUDA golden runs, every hidden-state layer check, the agreement rule and the final-cosine rule passed. The gates were registered 2026-10-22 (sha256 `169c0bba…`); after both amendments the file's sha256 is `f95b2672…`.

## Hardware & memory

Everything was run on one **NVIDIA Jetson AGX Thor** (128 GB unified memory, CUDA 13.0, Linux 6.8.12-tegra).

**Fit (estimated).** The build occupies 106.9 GiB. The planning figures add 2.0 GiB for the main 1M-token context state (not measured) and 6.0 GiB of workspace, for 114.9 GiB of a 117.0 GiB budget. The Engram tables take no memory.

**Release gate measurements** (`s08`, 2026-10-25, the first working engine, FP8 dense, no speculation):

| run | result |
|---|---|
| decode, 16-token prompt, 3 runs | 8.09 tok/s, deterministic |
| decode after an 8,192-token prompt | 7.14 tok/s |
| 1,048,576-token context allocated, 65,536-token prompt | prefill 30.4 min, then 6.59 tok/s decode |

**Current server** (measured by the author 2026-10-30, `./start.sh` defaults: 32K context, NVFP4 overlay, greedy, one stream, as reported in the [server README](https://github.com/patrickbdevaney/deepseek-v4.1-flash-cuda-server)):

| | result |
|---|---|
| decode, 256-token chat answers (code / explanation / math), DSpark K=3 (default) | 33.1 / 25.1 / 40.2 tok/s |
| decode, `SPEC=0` | 25.3 tok/s |
| sampled decode, T = 0.7 (with / without speculation) | 27.1 / 22.7 / 34.9; 25.0–25.2 tok/s |
| prefill, 1.8K / 7.2K-token prompt | 398 / 739 tok/s |
| unified memory still free while serving | 9 GiB of 122 (17 GiB with `SPEC=0`) |

The server README also reports these results:
- Speculation averages 1.345× over plain decoding on 27 held-out prompts. Greedy output was token-identical, with 0 divergences.
- Plain decode is about 74% of a 247 GB/s measured bandwidth roofline, at about 7.6 GB read per token.
- Image input through the CUDA vision tower was added on 2026-10-31.

These are the author's runs, not this card's audit; the pipeline's [`docs/zenith/`](https://github.com/patrickbdevaney/deepseek-v4.1-flash-reap-thor/tree/main/docs/zenith) holds the history.

## How it was made

1. **Keep-set.** `hope_sum_pf0`, 192 of 384 experts per layer, chosen on a held-out screen split. It is the same keep-set as the [MXFP4 master](https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-MXFP4#how-it-was-made), which documents the calibration corpus and the selection.
2. **Requantization** (`ops/s07_quant.py`). Each kept expert was quantized once, from the MXFP4 source, into ggml Q2_K/Q3_K blocks with GPTQ error-feedback rounding. The per-layer type mix favours the protected layers: the first three layers, the Engram, KV-source and DSpark-tap layers. The heal gain was rejected, so no gain is folded in.
3. **Dense path.** Copied from the base checkpoint unchanged. The NVFP4 overlay is a separate, optional copy of the dense attention and shared-expert weights.
4. **Arm choice.** The pre-registered A/B decision at matched expert bytes picked 192 requantized experts over 128 bit-exact ones (paired kl32, CI excludes zero). A model confidence set at α = 0.1 kept only `thorA` (FP8 dense). The overlay arm is measurably worse and ships because its pre-registered speed-lever gate passed.

## Limitations

- **The release gate failed as registered** (see above). The default served configuration (NVFP4 overlay on) scores top-1 0.7681.
- **Multilingual quality is much lower than in the unpruned model** (top-1 0.49). For non-English use, prefer the unpruned model.
- **Tool calling.** The calibration renderer dropped tool-call turns. The release server smoke test returned a malformed tool-call argument string. Tool-call accuracy is not evaluated.
- **Long context** quality is untested past 8K tokens. Allocating a 1M-token context and serving a 64K prompt worked in the release gate.
- **Runtime lock-in.** The `thor-v1` layout is read only by the CUDA server above, which targets Jetson AGX Thor (sm_110a).
- This repository has no `config.json` or `LICENSE` file. The architecture lives in `thor.json`, and the licence is the base model's MIT licence.

## Changelog / corrections

- 2026-10-26: published (s12 card generated from s10/s08 artifacts).
- 2026-10-30: tokenizer files and the NVFP4 overlay added. Measured CUDA-server numbers added.
- 2026-11-06: card rewritten and verified against repo files and build artifacts; a 2026-10-26 automated rewrite contained errors. Corrections: there is no `dense.safetensors`; the dense path lives in `layer-NN.safetensors` and `globals.safetensors`. There is no `config.json` or `LICENSE` in this repository. The MXFP4 master is 151.6 GiB, not 146.4 GiB. The overlay is 3.42 GiB, not 3.5. Speculation is on by default in the server; an earlier section said it was off. The card no longer says "text only": the server accepts images since 2026-10-31. The heal gain is 1.0, so nothing was folded into `w2`. The keep-set used no per-domain protection (protect fraction 0). The default served configuration (`thorA_overlay`) is now reported beside `thorA`.

## License & attribution

MIT, the same as the base model. The weights are derived from [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by [DeepSeek-AI](https://huggingface.co/deepseek-ai). Its licence is at [`LICENSE`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/main/LICENSE).

## Citation

```bibtex
@misc{devaney2026dsv41reap192thor,
  title        = {DeepSeek-V4.1-Flash-1017-REAP192-Thor: a 192-of-384 expert-pruned DeepSeek-V4.1-Flash-1017 for one Jetson AGX Thor},
  author       = {Patrick Devaney},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/patrickbdevaney/DeepSeek-V4.1-Flash-1017-REAP192-Thor}}
}
```
