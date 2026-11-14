---
license: mit
base_model: deepseek-ai/DeepSeek-V4.1-Flash-1017
base_model_relation: quantized
library_name: vllm
pipeline_tag: text-generation
tags:
- moe
- expert-offload
- 2-bit
- vector-quantization
- tandem
---

# DeepSeek-V4.1-Flash-1017 — Tandem VQ2 (lite package)

A smaller copy of DeepSeek-V4.1-Flash-1017 for **Tandem-vLLM**, an expert-offload
build of vLLM that runs this MoE model on one or two GPUs, with the
experts in host RAM and the CPU computing the experts the GPU does not hold.

**This is not an official DeepSeek release.** It is derived from DeepSeek's
MIT-licensed checkpoint (`LICENSE` and `DEEPSEEK_README.md` are DeepSeek's), with the routed experts
re-encoded. Quality is below the official model; see the numbers below.

| | official checkpoint | this package |
|---|---|---|
| disk | 476 GiB | **293 GiB** |
| routed experts | FP4 | 2-bit vector codes (2.10 bits/weight) + a ranked FP4 hot set |
| Engram tables | FP8 (2 × 96 GiB) | 4-bit e2m1 (2 × 48.6 GiB) |

## What is inside

| part | size | |
|---|---|---|
| `vq/` | 133 GiB | all routed experts of layers 0–39 as 2-bit vector codes ("VQ"): 512 sign-symmetric 8-weight patterns plus super-block scales, rounding optimized end to end against the official model's outputs |
| `fp4hot/` | 45 GiB | the 64 most-routed experts of every layer in their original FP4, ranked by a calibration routing count (`ranking.tsv`); the engine serves the top K of them in FP4 |
| `engram-e2m1-*.safetensors` | 97 GiB | the two Engram tables, requantized to 4-bit (e2m1 + UE8M0 per 32 values) |
| `model-0000*-of-00004.safetensors` | 17.5 GiB | everything else, unchanged: attention, shared experts, router, embeddings, the MTP layers (used as a speculative drafter) |

"VQ2" names this code family (2-bit vector codes); a different format would ship as a new repo.

## Quality

Measured on one RTX PRO 6000 (96 GB) + EPYC 9534, greedy unless noted, thinking on with low effort.
"Same-token KL" is the mean KL divergence of the next-token distribution from the official
checkpoint, teacher-forced on the official model's own outputs (lower is better).

| configuration | same-token KL | top-1 agreement | GPQA-diamond (198 × 3 runs) |
|---|---|---|---|
| this package, FP4 top 52 per layer | 0.109 | 0.904 | 77.4% |
| this package, VQ only | 0.151 | 0.877 | 74.1% |

GPQA: one greedy run plus two at temperature 0.6, 8000 max tokens; the FP4 hot set's
+3.4 points is significant (95% CI [+0.8, +5.9], p = 0.016). The 4-bit Engram tables
cost nothing measurable (KL 0.1092 → 0.1090 with the FP4 hot set). For scale: with all
experts in FP4 the same engine was at the official model's own noise floor (KL 0.007,
measured on the earlier V4.1-Flash).

Against the official checkpoint (stock vLLM 0.32, 4× H200), same client and settings (greedy,
thinking low, GPQA 8000 max tokens):

| test | official checkpoint | FP4 top 52 | VQ only |
|---|---|---|---|
| GSM8K-200 | 192 | 194 | 192 |
| GPQA-diamond 198 (one greedy run) | 161 (81.3%) | 152 (76.8%) | 145 (73.2%) |
| passkey retrieval, 8k / 32k / 94k tokens × 3 depths | 9/9 | 9/9 | 9/9 |

The 2-bit experts cost GPQA accuracy (−4.5 points with the FP4 hot set, −8 VQ only; both significant
paired against the official run); easier tasks and long-context retrieval do not show it.

## Speed and memory

Tandem-vLLM is a serving engine (continuous batching). System decode speed, VQ only, drafter off,
RTX PRO 6000 + EPYC 9534: 65 / 182 / 284 / 414 tok/s at 1 / 4 / 8 / 32 streams. On an RTX 5090 32 GB +
i5-14600KF with 128 GB DDR4 (`BASE_GB=10 RAM_GB=90 ENGRAM=files`): 16 / 18 / 20 tok/s at 1 / 2 / 4 streams.

Single user, one RTX PRO 6000 + EPYC 9534 (DDR5-4800), GSM8K-200 generation speed (thinking low):

| configuration | no drafter | DSpark k=3 (MTP layers in the package) |
|---|---|---|
| VQ only (65.5 GiB expert cache; 59.5 with the drafter) | 75.6 tok/s | 159.4 tok/s |
| FP4 top 52 (28 GiB VQ cache + 37 GiB FP4; 22 + 37 with the drafter) | 67.1 tok/s | 121.6 tok/s |

Accuracy on that set was 192–195/200 in every configuration. Speed depends mostly on the CPU and its
memory bandwidth (it computes the experts the GPU does not hold).

Other GPUs (Tandem-vLLM 0.2.0), GSM8K-50, one stream, decode tok/s:

| GPUs | CPU | tok/s |
|---|---|---|
| 2× RTX PRO 6000 (`GPU=0,1`) | EPYC 9534 | 93.1 |
| H200 (`BASE_GB=100`) | Threadripper PRO 7965WX, 24 cores | 60.7 |
| RTX 6000 Ada 48 GB (`BASE_GB=24`) | EPYC 7F32, 8 cores | 25.5 |
| RTX A6000 48 GB (`BASE_GB=24`) | EPYC 7F32, 8 cores | 22.1 |
| RTX A6000 + A6000 as helper GPU | EPYC 7F32, 8 cores | 26.7 |
| 2× 24 GB (A6000 limited to 24 GB), main + helper GPU | EPYC 7F32, 8 cores | 21.7 |
| 2× 16 GB (H200 limited to 16 GB, `GPU=0,1 BASE_GB=3.5`), Tandem-vLLM 0.2.1 | Threadripper PRO 7965WX, 24 cores | 36.6 |

Host RAM: the VQ experts (133 GiB) and the Engram tables (97 GiB) are pinned, plus the
FP4 rows in use. Measured engine RSS: 233 GiB VQ only, 270 GiB with FP4 top 52;
plan on 256 GB of RAM (VQ only) or 288 GB (FP4 top 52). With 128 GB, `RAM_GB=90 ENGRAM=files` pins only
the most-routed experts and reads the rest and the Engram rows from the package files (slower). The engine is exact with respect to
this package: an expert missing from the GPU cache is computed from host memory (by
the GPU directly or by the CPU), never skipped.

## How to run

Needs the Tandem-vLLM build: <https://github.com/ewhacc/tandem> (installs into its own venv on top of
the stock vLLM 0.31.0 wheel). GPUs: Blackwell (RTX 50xx, RTX PRO 6000), Hopper (H100/H200), Ada (RTX 40xx,
RTX 6000 Ada) and Ampere (RTX 30xx, RTX A6000), Tandem-vLLM 0.2.0 or later; AVX-512 or AVX2 x86-64 CPU.
Recommended: 96 GB GPU + 256 GB RAM. Smaller: 48 GB, 32 GB (RTX 5090, slower), two 24 GB cards (main + helper GPU), or two 16 GB cards (tensor parallel, 0.2.1).

```bash
git clone https://github.com/ewhacc/tandem && cd tandem && tools/install_tandem.sh
hf download ewhacc/DeepSeek-V4.1-Flash-1017-Tandem-VQ2 --local-dir DeepSeek-V4.1-Flash-1017-Tandem-VQ2
~/.venvs/tandem/bin/tandem-serve MODEL=$PWD/DeepSeek-V4.1-Flash-1017-Tandem-VQ2 BASE_GB=65.5           # VQ only
~/.venvs/tandem/bin/tandem-serve MODEL=$PWD/DeepSeek-V4.1-Flash-1017-Tandem-VQ2 FP4_TOP=52 BASE_GB=22 SPEC_N=3
~/.venvs/tandem/bin/tandem-serve MODEL=$PWD/DeepSeek-V4.1-Flash-1017-Tandem-VQ2 BASE_GB=24 KV_GB=2 STEP_T=8   # 48 GB card
~/.venvs/tandem/bin/tandem-serve MODEL=$PWD/DeepSeek-V4.1-Flash-1017-Tandem-VQ2 BASE_GB=7 KV_GB=2 STEP_T=8 CHUNK=2048 \
  RAM_GB=90 ENGRAM=files HELPER_GPU=1 HELPER_GB=20                                                       # 2x 24 GB, 128 GB RAM
~/.venvs/tandem/bin/tandem-serve MODEL=$PWD/DeepSeek-V4.1-Flash-1017-Tandem-VQ2 GPU=0,1 BASE_GB=3.5 KV_GB=1 STEP_T=8 CHUNK=1024  # 2x 16 GB, 256 GB RAM
```

- `FP4_TOP=K`: experts per layer served in FP4 from the hot set, 0–64. Each one costs
  about 18 MiB per layer of GPU memory (K = 52: 37 GiB); 0 = VQ only.
- `BASE_GB`: GPU cache for the VQ experts. Give it what is left of the card after the
  FP4 pool, KV cache and activations (and 6 GiB for the drafter with `SPEC_N`).
- `SPEC_N=k`: DSpark speculative decoding with the package's MTP layers.
- `GPU=0,1`: two GPUs, tensor parallel (needs a 16+ core CPU; two 16 GB cards work this way). `HELPER_GPU=1 HELPER_GB=20`: the second GPU
  caches and computes hot experts the main GPU does not hold (for 8-core CPUs or 24 GB cards).
- Download the whole repository: the weights loader alone does not fetch `vq/` or `fp4hot/`.
- The engine reads the package layout from the `"tandem"` block of `config.json`.

### Your own hot set

The shipped ranking comes from a general calibration mix. `tools/tandem_package.py hotlist`
(in the Tandem repo) builds a top-K list from a routing profile of your own workload,
restricted to the 64 experts per layer this package stores.

## Credits and license

Model by DeepSeek-AI ([DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)),
released under the MIT License, which this derived package keeps (`LICENSE`).
Package and engine: the Tandem project.
