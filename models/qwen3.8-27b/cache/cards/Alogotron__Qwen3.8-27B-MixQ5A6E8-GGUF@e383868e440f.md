---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: gguf
quantized_by: community
---

# Qwen3.8-27B-MixQ5A6E8-v0 (experimental rc1)

> **Experimental candidate — not optimized, recommended, or generally
> validated.** This immutable candidate exists to make a measured mixed-quant
> experiment reproducible. No quality-equivalence, speed-superiority, memory,
> or maximum-context claim is made beyond the sealed evidence cited below.

This community release is not affiliated with or endorsed by Qwen or Alibaba
Cloud.

## Artifact

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `Qwen3.8-27B-MixQ5A6E8-v0.gguf` | 20,804,373,120 | `eae1e5ba33e6b3ebeda4a22dc540ae644f8114eb85df63848d80842cf78dd348` |

Verify the download with `sha256sum -c SHA256SUMS`. `MANIFEST.json` is the
canonical machine-readable provenance record.

## Recipe and immutable provenance

- Upstream model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (Apache-2.0).
- BF16 input: `ggml-org/Qwen3.8-27B-GGUF@0669b98607d47046c7c2b3f801011d54a08cfccf`, SHA256 `5a3eedc837bcbd1365cdbf5b71e698df3122e76586ba872f07ce3ed4a9bfa97e`.
- Quantizer: `llama.cpp` b10442, commit `9b0a2ce859d3884252705e9ac7c93c98616bb238`.
- External imatrix: Bartowski `Qwen3.8-27B-imatrix.gguf@f0eec4a4bb4975114a030d048952d83c0a53c034`, SHA256 `aaa933d4b9ce23e1f65c548ad34f16956d8af44a51b5c15bf4f393ba59508cd8`.
- Quantization: Q5_K_M fallback; Q6_K for matched `.*attn.*` and `.*ssm.*` tensors; Q8_0 token embedding and output. Tensor rules SHA256 `cc2d2d8f797b3d6fb7895e0ec46a0dc1e88b5b9623d37d501385237183ae1033`.

The external importance matrix was not created by this project. See
`ATTRIBUTION.md`. Exact normalized dry-run/build argv and pins are retained in
`MANIFEST.json`; the reproducible entry points are:

```bash
bash scripts/build_candidate.sh --mode dry-run
bash scripts/build_candidate.sh --mode build
```

## Sealed Q5_K_M/Q6_K comparison (Vulkan lane)

Within this sealed three-artifact comparison only, `MixQ5A6E8-v0` is a measured
intermediate, nondominated point across artifact size, BF16-relative KL, and the
three named `llama-bench` workloads. Lower is better for size and KL; higher is
better for throughput.

| Artifact | Bytes | BF16-relative KL | `pp512` median tok/s | `pp2048` median tok/s | `tg128` median tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q5_K_M | 19,231,099,520 | 0.003949 | 1,141.60 | 1,668.81 | 39.0804 |
| MixQ5A6E8-v0 | 20,804,373,120 | 0.003083 | 1,139.07 | 1,707.78 | 36.5086 |
| Q6_K | 22,082,529,920 | 0.002099 | 1,116.58 | 1,600.91 | 33.5946 |

Compared with Q6_K, the candidate is 5.7881% smaller and its medians are
+2.0142% (`pp512`), +6.6756% (`pp2048`), and +8.6740% (`tg128`), while fidelity
is worse: full PPL is 6.6674 versus 6.6553 and BF16-relative KL is 0.003083
versus 0.002099. Compared with Q5_K_M, it is 8.1809% larger, has lower
BF16-relative KL (0.003083 versus 0.003949), and has mixed throughput:
-0.2216% (`pp512`), +2.3352% (`pp2048`), and -6.5808% (`tg128`). Its full PPL
of 6.6674 versus 6.6715 is too close to support a PPL-superiority claim.

Evidence for every value and the scoped classification above:
`results/llama-bench-mixq5a6e8-v0-vulkan-b10442-20260815T2111Z` and
`results/ppl-full-bf16ngl48-mixq5a6e8-v0-vulkan-b10442-20260815T2210Z`.

## CUDA lane (sm_86, dual RTX 3090)

The identical three-model matrix was rerun with the same pinned b10442 tree
compiled for CUDA sm_86 and the same protocol (pp512/pp2048/tg128, 5
repetitions, layer split 1/1). Median tok/s:

| Artifact | `pp512` | `pp2048` | `tg128` |
| --- | ---: | ---: | ---: |
| Q5_K_M | 1,353.92 | 1,980.47 | 39.49 |
| MixQ5A6E8-v0 | 1,336.72 | 1,963.61 | 37.43 |
| Q6_K | 1,251.84 | 1,816.35 | 34.56 |

CUDA prompt-processing medians exceed Vulkan by +13% to +20% for all three
artifacts; `tg128` medians differ by +1–3%. The candidate's Vulkan-only
`pp2048` advantage over Q5_K_M (+2.34%) did not reproduce under CUDA (−0.85%
here), while its margin over Q6_K widened. The scoped classification above is
unchanged: a measured intermediate, nondominated point — on either backend.
Evidence: `results/llama-bench-frontier-q4toq8-cuda-b10442-20260819T094256Z`
and `results/llama-bench-mixq5a6e8-v0-cuda-b10442-20260818T154432Z` (an earlier
3-model CUDA session; values reproduced within ~1–2%).

## Measured Q4–Q8 frontier (CUDA lane)

Six artifacts, identical sealed protocol (CUDA b10442, dual RTX 3090, layer
split 1/1, pp512/pp2048/tg128, 5 repetitions, medians). BF16-relative KL and
full WikiText-2 PPL are from the pinned Vulkan perplexity lane (identical
protocol across artifacts; Q8_0 was not KL/PPL-measured here).

| Artifact | Bytes | BF16-relative KL ↓ | Full PPL ↓ | `pp512` tok/s ↑ | `pp2048` tok/s ↑ | `tg128` tok/s ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 17,772,537,440 | 0.009584 | 6.6921 | 1,406.90 | 2,103.70 | 42.47 |
| MixQ4A6E8-v0 (sibling candidate) | 19,021,793,888 | 0.007835 | 6.6830 | 1,368.37 | 2,011.01 | 40.49 |
| Q5_K_M | 19,231,099,520 | 0.003949 | 6.6715 | 1,353.92 | 1,980.47 | 39.49 |
| MixQ5A6E8-v0 (this artifact) | 20,804,373,120 | 0.003083 | 6.6674 | 1,336.72 | 1,963.61 | 37.43 |
| Q6_K | 22,082,529,920 | 0.002099 | 6.6553 | 1,251.84 | 1,816.35 | 34.56 |
| Q8_0 | 29,116,388,960 | not measured | not measured | 1,515.64 | 2,186.57 | 28.58 |

BF16 reference PPL is 6.648031. Within this sealed six-artifact set, the ladder
is a monotone quality↔throughput trade-off: no artifact is simultaneously
higher-fidelity (KL) and faster (`tg128`) than another, so every measured point
is nondominated. `MixQ5A6E8-v0` occupies the measured intermediate position
between Q5_K_M and Q6_K. Caveat: Q6_K and Q8_0 each had one outlier `pp2048`
repetition (stddev 91/134 tok/s); medians are reported throughout.

This does not establish a broad optimization, best, recommendation, VRAM
benefit, Q8/BF16 equivalence, or industry-benchmark quality claim. Prompt
processing is compute-bound on this host (Q8_0 leads `pp*` despite its size);
token generation is bandwidth-bound (ordering follows artifact size).

The fixed structural and text integration smoke suite passed; it is not an
industry quality benchmark. Evidence:
`results/text-validation-mixq5a6e8-v0-vulkan-b10442-20260816T0810Z`.

## MTP: separate, unbundled, experimental

MTP is disabled by default. The Q4_0 and Q8_0 MTP draft models are separate
upstream companion files and are **not included in this upload**. In the sealed
8K test both MTP modes first diverged from target-only at generated token 117;
exact token-ID parity passed at 32K. Therefore MTP is not lossless or
token-identical and must not be used for exact, schema-constrained,
regression, or reproducibility-sensitive work. Evidence:
`results/mtp-parity-v021-mixq5a6e8-v0-vulkan-b10442-20260816T1045Z`.

## Multimodal: failed experimental gate

The matching Q8_0 projector is a separate upstream companion and is **not
included in this upload**. In sealed run
`results/multimodal-v02-mixq5a6e8-v0-vulkan-b10442-20260816T1145Z`, the
candidate and Q5_K_M reference both passed projector CLI load, server startup,
telemetry capture, object/color/count, chart/table, and the text-only regression
with the projector loaded. Both returned `LEFT_OF` and `ABOVE` for the spatial
fixture but failed its exact verbose-label scorer, and both read `LIMA-47` as
`LINA-47`. The full projector integration gate therefore failed. These matching
fixture outcomes show no candidate-specific regression on this narrow suite;
they do not validate general multimodal quality or support any quality or
optimization claim.

## Compatibility and limitations

- Evidence was collected on a dedicated dual-GPU validation host with 2× NVIDIA
  RTX 3090 24 GiB GPUs, pinned llama.cpp b10442 Vulkan tools, and a pinned
  b10442 CUDA sm_86 build; other runtimes and hardware are unverified.
- The tensor-selection rules remain experimental even though this exact
  artifact was built and experimentally checked.
- Published companion artifacts, if desired, must be fetched separately from
  their pinned upstream revision and verified by hash.
- Review `MANIFEST.json`, `NOTICE`, and `ATTRIBUTION.md` before redistribution.

## Sibling candidate

`MixQ4A6E8-v0` (Q4_K_M fallback, same attention/SSM and embedding rules) is
built and measured on the same evidence base; see
`Alogotron/Qwen3.8-27B-MixQ4A6E8-GGUF` *(link activates when published)*.
