---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
quantized_by: community
---

# Qwen3.8-27B-MixQ4A6E8-v0 (experimental rc1)

> **Experimental candidate — not optimized, recommended, or generally
> validated.** This immutable candidate exists to make a measured mixed-quant
> experiment reproducible. No quality-equivalence, speed-superiority, memory,
> or maximum-context claim is made beyond the sealed evidence cited below.

This community release is not affiliated with or endorsed by Qwen or Alibaba
Cloud.

## Artifact

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `Qwen3.8-27B-MixQ4A6E8-v0.gguf` | 19,021,793,888 | `5cd87bedac2cec27d2b16203ef35b4fa861cf3a6e409a28a7133b37237e49509` |

Verify the download with `sha256sum -c SHA256SUMS`. `MANIFEST.json` is the
canonical machine-readable provenance record.

## Recipe and immutable provenance

- Upstream model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0).
- BF16 input: `ggml-org/Qwen3.8-27B-GGUF@0669b98607d47046c7c2b3f801011d54a08cfccf`, SHA256 `5a3eedc837bcbd1365cdbf5b71e698df3122e76586ba872f07ce3ed4a9bfa97e`.
- Quantizer: `llama.cpp` b10442, commit `9b0a2ce859d3884252705e9ac7c93c98616bb238`.
- External imatrix: Bartowski `Qwen3.8-27B-imatrix.gguf@f0eec4a4bb4975114a030d048952d83c0a53c034`, SHA256 `aaa933d4b9ce23e1f65c548ad34f16956d8af44a51b5c15bf4f393ba59508cd8`.
- Quantization: Q4_K_M fallback; Q6_K for matched `.*attn.*` and `.*ssm.*` tensors; Q8_0 token embedding and output. Tensor rules SHA256 `4000c7ce7f467c4972df085e0ce115bb1f79cc252be8dbc2dd28b14eea2599e4`.

The external importance matrix was not created by this project. See
`ATTRIBUTION.md`. Build evidence (dry-run validation, quantize log, commands,
environment): `results/build-mixq4a6e8-v0-20260818T231742Z`.

## Sealed Q4_K_M comparison (quality lane)

Full WikiText-2 test set (602 chunks, ctx/batch/ubatch 512, pinned corpus,
BF16-relative KL against freshly generated BF16 base logits), pinned b10442
Vulkan lane, identical protocol across artifacts. Lower is better.

| Artifact | BF16-relative KL | Full PPL | ΔPPL vs BF16 (6.648031) |
| --- | ---: | ---: | ---: |
| MixQ4A6E8-v0 | 0.007835 ± 0.00005 | 6.6830 ± 0.0415 | +0.0350 |
| Q4_K_M | 0.009584 ± 0.00008 | 6.6921 ± 0.0416 | +0.0441 |

Within this sealed two-artifact comparison, the candidate's BF16-relative KL is
18.2492% lower than Q4_K_M (0.007835 versus 0.009584); the gap is ~35× the
combined standard error. Its full PPL is 0.0091 lower (6.6830 versus 6.6921);
given the ±0.04 uncertainties this supports the KL claim but only a weak PPL
direction. Evidence:
`results/ppl-full-bf16ngl48-mixq4a6e8-v0-vulkan-b10442-20260819T063633Z`.

## Sealed Q4_K_M comparison (speed lanes)

Identical `llama-bench` protocol on both lanes: pp512/pp2048/tg128, 5
repetitions, pinned b10442 CUDA build, medians in tok/s. Higher is better.

**Single RTX 3090 24 GiB** (one GPU, tensor split 1):

| Artifact | `pp512` | `pp2048` | `tg128` |
| --- | ---: | ---: | ---: |
| Q4_K_M | 1,399.42 | 1,396.11 | 41.53 |
| MixQ4A6E8-v0 | 1,366.28 | 1,363.71 | 39.50 |

Candidate deltas: −2.3681% (`pp512`), −2.3206% (`pp2048`), −4.8880% (`tg128`).
Evidence: `results/llama-bench-mixq4a6e8-v0-single-gpu-cuda-b10442-20260818T233255Z`.

**Dual RTX 3090** (layer split 1/1):

| Artifact | `pp512` | `pp2048` | `tg128` |
| --- | ---: | ---: | ---: |
| Q4_K_M | 1,406.90 | 2,103.70 | 42.47 |
| MixQ4A6E8-v0 | 1,368.37 | 2,011.01 | 40.49 |

Candidate deltas: −2.7386% (`pp512`), −4.4062% (`pp2048`), −4.6733% (`tg128`).
Evidence: `results/llama-bench-frontier-q4toq8-cuda-b10442-20260819T094256Z`.

Net measured trade within these sealed comparisons: the candidate is 7.0291%
larger than Q4_K_M, 18.2492% lower in BF16-relative KL, and 2.3–4.9% slower
across the three workloads. Whether that trade serves a given use case is left
to the reader; no recommendation is made.

## Measured Q4–Q8 frontier (CUDA lane, dual RTX 3090)

Six artifacts, identical sealed protocol; KL/PPL from the pinned Vulkan
perplexity lane (Q8_0 not KL/PPL-measured here). Medians.

| Artifact | Bytes | BF16-relative KL ↓ | Full PPL ↓ | `pp512` ↑ | `pp2048` ↑ | `tg128` ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 17,772,537,440 | 0.009584 | 6.6921 | 1,406.90 | 2,103.70 | 42.47 |
| MixQ4A6E8-v0 (this artifact) | 19,021,793,888 | 0.007835 | 6.6830 | 1,368.37 | 2,011.01 | 40.49 |
| Q5_K_M | 19,231,099,520 | 0.003949 | 6.6715 | 1,353.92 | 1,980.47 | 39.49 |
| MixQ5A6E8-v0 (sibling) | 20,804,373,120 | 0.003083 | 6.6674 | 1,336.72 | 1,963.61 | 37.43 |
| Q6_K | 22,082,529,920 | 0.002099 | 6.6553 | 1,251.84 | 1,816.35 | 34.56 |
| Q8_0 | 29,116,388,960 | not measured | not measured | 1,515.64 | 2,186.57 | 28.58 |

Within this sealed set no artifact is simultaneously higher-fidelity (KL) and
faster (`tg128`) than another: every point is nondominated. `MixQ4A6E8-v0`
occupies the measured intermediate position between Q4_K_M and Q5_K_M —
1.0883% smaller and +1.0672%/+1.5420%/+2.5225% faster (`pp512`/`pp2048`/
`tg128`) than Q5_K_M at worse fidelity (KL 0.007835 versus 0.003949).

This does not establish a broad optimization, best, recommendation, VRAM
benefit, Q8/BF16 equivalence, or industry-benchmark quality claim.

## Validation scope: what is NOT covered

- **MTP** (multi-token prediction draft heads): not validated for this
  artifact; do not assume the sibling's MTP parity results transfer.
- **Multimodal projector**: not validated for this artifact.
- **Text integration smoke suite** (chat template, thinking controls, tool
  calling): passed for this artifact (exact tokenizer/architecture metadata
  parity plus five deterministic integration tests; sealed run
  `text-validation-mixq4a6e8-v0-vulkan-b10442-20260819T131943Z`). It is a
  reproducibility smoke, not a quality benchmark.
- Downstream task-quality suites: not run. Perplexity/KL are distributional
  proxies, not task benchmarks.
- Downstream task-quality suites: not run. Perplexity/KL are distributional
  proxies, not task benchmarks.

## Compatibility and limitations

- Evidence was collected on a dedicated validation host with 2× NVIDIA RTX 3090
  24 GiB GPUs: pinned b10442 CUDA sm_86 build (single- and dual-GPU speed
  lanes) and pinned b10442 Vulkan tools (quality lane). Other runtimes and
  hardware are unverified.
- At 19,021,793,888 bytes the artifact fits a single 24 GiB card with
  context headroom; exact maximum context per runtime is unmeasured and not
  claimed.
- The tensor-selection rules remain experimental even though this exact
  artifact was built and experimentally checked.
- Review `MANIFEST.json`, `NOTICE`, and `ATTRIBUTION.md` before redistribution.

## Sibling release

`MixQ5A6E8-v0` (Q5_K_M fallback, same attention/SSM and embedding rules):
`Alogotron/Qwen3.8-27B-MixQ5A6E8-GGUF`.
