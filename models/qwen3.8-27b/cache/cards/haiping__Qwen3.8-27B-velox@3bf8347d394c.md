---
license: apache-2.0
base_model:
  - Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - nvfp4
  - iq3_xxs
  - iq2_s
  - velox
  - dflash2
  - dspine
  - speculative-decoding
---

# Qwen3.8-27B for velox: low-bit decode + NVFP4 prefill + DFlash2

These are single-file velox `.vi` v1 artifacts. Each holds the text model, the vision tower and the
DFlash2 drafter. The MTP head is not included, because these files are served with DFlash2.

Each file pairs two kinds of weight encoding:

- **IQ3_XXS and Q4/Q5:** decode and DFlash2 verify are limited by memory bandwidth, so these
  low-bit encodings make them faster.
- **NVFP4:** prefill is limited by compute, so it runs on FP4 tensor cores.

The files can be read only by [velox](https://github.com/Oceania2018/velox-inferer). NInfer cannot
read `.vi` files.

## Files

The IQ3_XXS files need a 32 GB card. For 16 GB cards, see the IQ2_S hybrid v1 below.

| File | Size | sha256 | Status |
|---|---:|---|---|
| `qwen3_8_27b_iq3_xss_nvfp4_hybrid_v3.vi` | 24.09 GB | `86e3782da7afaa9ce1b72eb95110118596fcca5b692a7a7b86384e248d759805` | **Recommended** (RTX 5090 production since 2026-11-01) |
| `qwen3_8_27b_iq3_xss_nvfp4_hybrid_v2.vi` | 24.38 GB | `152ee4304efbc6b9531b92480f9b6f1f270e6b06a53fb18121de72eb4bb36bca` | Previous version, kept for comparison |
| `qwen3_8_27b_iq2_s_nvfp4_hybrid_v1.vi` | 12.46 GB | `6bd53ba74c40c33b38afab6630d5733b9eccf6653279dca9b2a787b1866e9282` | For 16 GB cards (RTX 5060 Ti), 1 lane; see [below](#16-gb-cards-iq2_s--nvfp4-hybrid-v1) |

### What each version contains

**v2**

- Starts from the IQ3_XXS DFlash2 package.
- The MLP `gate`/`up` weights stay IQ3_XXS, for decode. Each also carries an NVFP4 `@prefill`
  encoding, which velox uses for chunks longer than 16 tokens.
- MLP `down` and the GDN/attention `output` projections are replaced with the NVFP4 or FP8 weights
  of the NVFP4 release.
- The GDN and attention input projections stay Q4/Q5 with int8 activations.
- The MTP head is dropped. The payload keeps only the objects the model references.

**v3**

- Everything in v2.
- The GDN `query/key/value/z` and attention `query/gate/key/value` projections are re-encoded as
  NVFP4 with FP4 activations. The source is the FP8 weights of the NVFP4 release.
- Each layer's activation divisor comes from the bound of its input RMSNorm,
  `sqrt(dim) * max|1 + w|`, so activations cannot saturate.
- With this change, every large prefill projection runs on FP4 tensor cores. Decode reads fewer
  bytes than v2, because NVFP4 is smaller than Q5.

v3 weight formats (`velox inspect`):

| Format / layout | Objects | GB |
|---|---:|---:|
| `nvfp4` / `block_scale_k16_m128x4_v1` | 193 | 12.95 |
| `iq3_xxs` / `iq3_xxs_planar_v1` | 64 | 4.37 |
| `fp8_e4m3fn_row_bf16` / `row_scale_v1` | 80 | 4.15 |
| `q8_g32_fp16` / `row_split_k128_v1` | 8 | 1.57 |
| `q4_g64_fp16` / `row_split_k128_v1` | 55 | 0.49 |
| `bf16` / `contiguous_le_v1` | 620 | 0.45 |
| `q5_g64_fp16` / `row_split_k128_v1` | 54 | 0.11 |
| others (q6, int32, fp32, resources) | | 0.01 |

## Benchmarks

There are two sets of numbers. The first uses public data only and anyone with an RTX 5090 can
rerun it. The second replays a private agent session, which is the production workload.

### Public datasets (reproducible)

All inputs are pinned:

- **ShareGPT V3:** 200 prompts at 1 and at 6 concurrent clients.
- **Random 4K:** 50 prompts of 4096 tokens in, up to 128 out, 1 client.
- **ScreenSpot-v2:** all 1272 samples.

The serving tests use vLLM's `benchmark_serving.py` v0.6.6.post1. Each variant runs on its own
fresh server with the flags listed below. The serving numbers are the mean of three rounds'
medians; between rounds, these medians differ by at most ±3% (±6% for v2's TPOT at 6 clients).
The scripts, raw results and caveats are in the velox repo, in
[`bench/2026-11-01-rtx5090-32g-public`](https://github.com/Oceania2018/velox-inferer/tree/main/bench/2026-11-01-rtx5090-32g-public).

| Test | Metric | NVFP4 baseline | v2 | v3 |
|---|---|---:|---:|---:|
| ShareGPT, 1 client | TTFT p50 | 58.2 ms | 61.2 ms (+5.2%) | **53.3 ms (−8.4%)** |
| | TPOT p50 | 5.5 ms | 4.8 ms (−12.7%) | **4.5 ms (−17.9%)** |
| | End-to-end p50 | 814 ms | 722 ms (−11.3%) | **677 ms (−16.8%)** |
| ShareGPT, 6 clients | TTFT p50 | 118 ms | 124 ms (+5.3%) | **112 ms (−5.0%)** |
| | TPOT p50 | 12.2 ms | 14.3 ms (+17.4%) | **11.9 ms (−2.8%)** |
| | Output tokens/s | 445 | 334 (−24.8%) | **503 (+13.2%)** |
| Random 4K, 1 client | TTFT p50 (prefill) | 672 ms | 726 ms (+8.0%) | **640 ms (−4.7%)** |
| | TPOT p50 | 6.3 ms | 5.7 ms (−9.1%) | **5.3 ms (−15.9%)** |
| ScreenSpot-v2 | Click accuracy | **96.1%** | 95.8% | 95.6% |

- **v3:** faster than the baseline on every median, and 0.5 points lower on ScreenSpot-v2
  (6 fewer hits out of 1272).
- **v2:** decodes faster on one client, but its prefill is slower, and at 6 clients it is slower
  throughout.

Notes:

- **Output tokens/s:** velox ignores `ignore_eos`, so the variants generate different lengths.
  Compare TTFT and TPOT instead. Output tokens/s also reflects length, and a few
  connection stalls on the Windows test host make it noisy.
- **ScreenSpot prompt:** the prompt is ours, not an official one. These scores compare the
  three files with each other; they are not leaderboard entries.

### Public datasets on RTX PRO 6000 (2026-11-01)

The same client, data, seeds and v3 file as above, on an RTX PRO 6000 Blackwell Server Edition
(96 GB, 188 SMs, ECC on). Four differences from the RTX 5090 run:

- The server ran with **10 lanes** (this card's production setting). ShareGPT also ran at **10
  clients**.
- The server ran natively, not in Docker, from branch `rtx-pro-6000` commit `0bbda78`.
- The production server shared the card during the run.
- There were **2 rounds**, not 3. The two rounds' medians differ by at most ±2%.

The RTX 5090 column repeats the v3 results above (3 rounds, 6 lanes).

| Test | Metric | RTX 5090 | RTX PRO 6000 |
|---|---|---:|---:|
| ShareGPT, 1 client | TTFT p50 | 53.3 ms | **38.9 ms (−27%)** |
| | TPOT p50 | **4.5 ms** | 5.1 ms (+13%) |
| ShareGPT, 6 clients | TTFT p50 | 112 ms | **98 ms (−13%)** |
| | TPOT p50 | **11.9 ms** | 12.5 ms (+5%) |
| | Output tokens/s | **503** | 478 (−5%) |
| ShareGPT, 10 clients | TTFT p50 | — | 128 ms |
| | TPOT p50 | — | 16.0 ms |
| | Output tokens/s | — (6 lanes) | **605** |
| Random 4K, 1 client | TTFT p50 (prefill) | 640 ms | **449 ms (−30%)** |
| | TPOT p50 | **5.3 ms** | 5.8 ms (+9%) |
| ScreenSpot-v2 | Click accuracy | 95.6% | 95.6% |

- **Prefill and first token:** faster on the PRO 6000, which has 10% more SMs.
- **Decode with 1–6 clients:** slower per token on the PRO 6000. Decode is limited by memory
  bandwidth. Both cards list 1792 GB/s, but with ECC on the PRO 6000 streams about 1536 GB/s.
- **10 clients:** the PRO 6000 serves 605 output tokens/s. That is 20% above the RTX 5090's
  6-client result. The 5090 was not measured at 10 clients; with 6 lanes, more clients would
  queue rather than add throughput.
- **ScreenSpot-v2:** both cards produce the same answers: 1216 of 1272 hits, identical in every
  category.

Single round of the NVFP4 baseline on the same PRO 6000 server, for reference: TPOT p50 6.1 ms at
1 client and 17.1 ms at 10 clients; 559 output tokens/s at 10 clients; TTFT p50 481 ms on
Random 4K.

### Private agent session

**Workload:** a recorded 142-step computer-use agent session, replayed through an
OpenAI-compatible gateway with `velox replay` at temperature 0. Every step sends a screenshot,
receives a tool call, and reuses the session's prefix cache.

**Baseline:** the same velox server with
[`haiping/Qwen3.8-27B-nvfp4-dflash2-NInfer`](https://huggingface.co/haiping/Qwen3.8-27B-nvfp4-dflash2-NInfer).
That is the official Qwen3.8-27B NVFP4 artifact, with its DFlash2 MLP re-encoded to NVFP4. Its
projections are NVFP4 and FP8.

**Statistic:** each value is the mean of the per-run medians. TTFT and prefill: lower is better.
Decode: higher is better.

| Weights | Runs | TTFT p50 | Prefill p50 | Decode p50 |
|---|---:|---:|---:|---:|
| NVFP4 baseline | 1 | 693 ms | 264 ms | 267 tok/s |
| **v3** | 3 | **587 ms (-15.4%)** | **237 ms (-10.4%)** | **330 tok/s (+23.3%)** |
| v2 | 2 | 647 ms (-6.7%) | 273 ms (+3.4%) | 309 tok/s (+15.6%) |

The three v3 runs differed by less than 1 percentage point. A rerun of the unchanged baseline
differed from the first baseline run by about ±3%.

**Tool-call quality.** All runs use temperature 0, so these numbers are deterministic for each
artifact:

| Weights | Valid calls | Same tool | Same arguments | Clicks within 50 px | Click distance p50 |
|---|---:|---:|---:|---:|---:|
| NVFP4 baseline | 142 | 109 | 79 | 58 | 9.2 px |
| v3 | 142 | 108 | 81 | 56 | 11.7 px |
| v2 | 142 | 108 | 79 | 58 | 10.3 px |

v3 trades a small amount of click precision for its speed: 2 fewer clicks land within 50 px, out
of 142 steps. The likely cause is that its input projections use 4-bit weights and activations.
v2 matches the baseline's quality.

### Hardware and software

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 32 GB (GB202, sm_120a, 170 SMs), driver 617.42 |
| Host | Windows 11 + Docker Desktop (WSL2) |
| Toolchain | CUDA 13.1, Rust stable |
| velox | branch [`rtx-5090-32g`](https://github.com/Oceania2018/velox-inferer/tree/rtx-5090-32g), commit `5b627e3` or later |
| Server flags | `--max-concurrency 6 --max-context 49152 --kv-format nvfp4 --prefill-chunk 2048 --host-context-mib 24576 --host-state-dtype bf16 --device-state-dtype f32 --ctx-checkpoints 1 --vram-budget-mib 30720` |
| Device memory | 27.1 GiB after warm-up, before the attention pool, with 6 lanes |
| RTX PRO 6000 | RTX PRO 6000 Blackwell Server Edition 96 GB (GB202, sm_120a, 188 SMs, ECC on), Linux. Branch [`rtx-pro-6000`](https://github.com/Oceania2018/velox-inferer/tree/rtx-pro-6000) commit `0bbda78` or later, same flags with `--max-concurrency 10 --vram-budget-mib 90112 --disk-context-mib 0` |

The public benchmark ran on the same machine (Intel Core Ultra 9 285K, 63.4 GB RAM, WSL VM capped
at 48 GB), with `--disk-context-mib 0` added.

The tuned kernels live on velox's hardware branches. This README states numbers only for
`rtx-5090-32g` and `rtx-pro-6000`. On other cards, use the branch that matches the card and
measure there.

## Usage

```bash
velox serve --artifact qwen3_8_27b_iq3_xss_nvfp4_hybrid_v3.vi --kv-format nvfp4 --port 8010
```

## How v3 is built

1. Re-encode the mixer projections of the NVFP4 release:
   `velox_requant_mixer_nvfp4.py qwen3_8_27b_nvfp4_draftmlp-nvfp4.ninfer mixer.ninfer`.
2. Swap them into v2 with velox:

   ```bash
   velox compose --base qwen3_8_27b_iq3_xss_nvfp4_hybrid_v2.vi --alt mixer.ninfer \
     --match "$M" --replace "$M" --suffix mixer --output qwen3_8_27b_iq3_xss_nvfp4_hybrid_v3.vi
   ```

   `$M` is the comma-separated list of the eight projections:
   `text/layers/*/gdn/{query,key,value,z}` and `text/layers/*/attention/{query,gate,key,value}`.

## 16 GB cards: IQ2_S + NVFP4 hybrid v1

`qwen3_8_27b_iq2_s_nvfp4_hybrid_v1.vi` is built for the RTX 5060 Ti 16 GB.

### Contents

- **Base:** the GSQ IQ2_S package `qwen3_8_27b_gsq_iq2s_p_dflash2.vi` from
  [haiping/Qwen3.8-27B-GSQ-IQ2_S-velox](https://huggingface.co/haiping/Qwen3.8-27B-GSQ-IQ2_S-velox).
  Its text weights are ISTA-DASLab's GSQ-RCO IQ2_S GGUF, with GGML blocks kept as stored. It also
  carries the vision tower and the DFlash2 drafter. Every one of its weights is unchanged.
- **Added:** NVFP4 `@prefill` encodings, with FP4 activations, for the dense-MLP `gate` and
  `up` matrices of 11 layers (22 matrices, 1.10 GB). They come from
  [neroued/Qwen3.8-27B-nvfp4-NInfer](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).
  velox uses them only for chunks longer than 16 tokens. Decode and DFlash2 verify still read the
  IQ2 weights.
- **Matrices chosen:** we ranked each layer's MLP matrices by the prefill time an NVFP4 copy saves
  per byte, using `velox bench-linear` at 1024 tokens on the 5060 Ti. IQ2_S, IQ2_XS and IQ1_M
  matrices come first, because their GGML prefill kernels are the slowest. The selection was
  filled up to the memory a 1-lane, 48K-context server can spare on 16 GB.

### Why only 1.10 GB

NVFP4 is slower than the IQ2 kernels at decode widths, so the IQ2 weights have to stay. The NVFP4
copies are added on top of them. `velox serve` gives the attention pool whatever is left of
`--vram-budget-mib`. Every extra byte of weights therefore costs context: about 57K tokens per GB
on this card. Total device memory stays at the budget either way.

With 1.10 GB of copies, one lane still fits a 48K context: 14.04 GiB in use after warm-up, plus a
50,960-token pool, within a 15,872 MiB budget. Two lanes no longer fit at 48K.

### Measured (RTX 5060 Ti 16 GB, 2026-11-01)

The baseline is the IQ2_S base file above, on the same build. All runs use `velox generate`
with DFlash2 (7 drafts), an NVFP4 KV cache and 1024-token prefill chunks.

| | IQ2_S base | IQ2_S + NVFP4 hybrid v1 |
|---|---:|---:|
| Prefill, 6,774-token prompt, warm (two interleaved pairs) | 6.34 s / 6.07 s | 6.14 s / 5.82 s (−3 to −4%) |
| Decode | | Unchanged: the same IQ2 weights |
| DFlash2 rounds for 256 tokens, summed over 6 prompts | 423 | 429 |

The two files differ only above 16 tokens, so greedy outputs can diverge after prefill. On a
single prompt this can shift DFlash2 acceptance either way. Summed over the six prompts,
acceptance matched within noise. These are single-stream measurements; no agent replay or
public-dataset benchmark has been run on this file yet.

### Running

The server needs velox branch
[`rtx-5060-ti`](https://github.com/Oceania2018/velox-inferer/tree/rtx-5060-ti) with main's
`@prefill` support, which is commits `c998296` and `a3fd04c` or later. Older builds load the file
but ignore the NVFP4 copies.

```bash
velox serve --artifact qwen3_8_27b_iq2_s_nvfp4_hybrid_v1.vi --model-id qwen3.8-27b \
  --chat-template qwen3_8.jinja --max-context 49152 --max-concurrency 1 --kv-format nvfp4 \
  --prefill-chunk 1024 --host-context-mib 24576 --host-state-dtype bf16 --ctx-checkpoints 1 \
  --vram-budget-mib 15872
```

### How it is built

```bash
velox compose --base qwen3_8_27b_gsq_iq2s_p_dflash2.vi --alt qwen3_8_27b_nvfp4.ninfer \
  --match "$M" --suffix prefill --output qwen3_8_27b_iq2_s_nvfp4_hybrid_v1.vi
```

`$M` is the comma-separated list of `text/layers/<n>/mlp/gate` and `text/layers/<n>/mlp/up`
for layers 14, 33, 13, 2, 23, 37, 47, 6, 22, 35 and 38 (22 matrices). `down` keeps its IQ2 encoding.

## DSpine drafter

The DSpine drafter is published separately at [haiping/Qwen3.8-27B-DSpine-b16-nvfp4](https://huggingface.co/haiping/Qwen3.8-27B-DSpine-b16-nvfp4). It replaces the built-in
DFlash2 drafter with `--drafter dspine`. On the RTX 5090 agent replay with the v3 file, it raises
tokens per round from 4.86 to 8.02, decode by 33.8%, and cuts end-to-end latency by 15%.

## License

Apache-2.0, the same license as Qwen3.8-27B.
