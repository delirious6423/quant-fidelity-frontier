---
base_model: Qwen/Qwen3.8-27B
license: apache-2.0
tags:
  - gguf
  - llama.cpp
  - quantized
  - qwen
pipeline_tag: image-text-to-text
---

<div align="center">
  <img src="assets/banner.svg" alt="Qwen3.8 STQ — Quantization Frontier" width="100%" />
  <p><strong>One base model. One pinned toolchain. Four measured GGUF tiers.</strong></p>
  <p>
    <a href="https://github.com/Topabaem05/mix-stq">Research</a> ·
    <a href="https://huggingface.co/datasets/topabaem/mix-stq-artifacts">Evidence</a> ·
    <a href="https://huggingface.co/Qwen/Qwen3.8-27B">Base model</a>
  </p>
</div>

## Model family

This repository publishes controlled GGUF quantizations of `Qwen/Qwen3.8-27B`.
Every tier was produced from the same BF16 GGUF with the same pinned `llama.cpp`
build and importance matrix. This is a measurement study, not a new codec claim.

| tier | size | physical bpw | Top-1 | WikiText-2 PPL | best fit |
|---|---:|---:|---:|---:|---|
| **IQ3_XXS** | 11.19 GB | 3.3273 | 86.00% | 7.3318 | smallest practical text deployment |
| **IQ4_XS** | 15.08 GB | 4.4862 | 85.63% | 6.9023 | stronger perplexity at low memory |
| **Q4_K_M** | 16.55 GB | 4.9219 | 86.50% | 6.8611 | balanced default |
| **Q5_K_M** | 19.23 GB | 5.7201 | 86.88% | 6.8498 | closest measured quality to BF16 |

Point `llama.cpp` at the first shard; the remaining shards are discovered automatically.

## Measured quality

<div align="center">
  <img src="assets/top1-frontier.svg" alt="Top-1 accuracy by quantization tier" width="100%" />
</div>

BF16 baseline: **696/800 = 87.00%**. Two independent paid runs reproduced the
same file hash and item-level correctness vector. Results use a preregistered
800-item stratified MMLU+ARC letter-logprob harness, seed 22, 10,000 paired
bootstrap samples, exact two-sided McNemar, and a +2.00 pp non-inferiority margin.

| arm | correct | accuracy | loss vs BF16 | 95% CI of loss | McNemar p | verdict |
|---|---:|---:|---:|---|---:|---|
| IQ3_XXS | 688/800 | **86.00%** | 1.000 pp | [−0.375, 2.375] pp | 0.215 | undetermined |
| IQ4_XS | 685/800 | **85.63%** | 1.375 pp | [0.500, 2.375] pp | 0.007 | significant loss on this set |
| Q4_K_M | 692/800 | **86.50%** | 0.500 pp | [−0.375, 1.375] pp | 0.388 | non-inferior |
| Q5_K_M | 695/800 | **86.88%** | 0.125 pp | [−0.750, 1.000] pp | 1.000 | non-inferior |

Perplexity is monotone with bit width even though Top-1 is not. The IQ4/IQ3
ordering is therefore treated as a property of this 800-item sample, not a broad
quality claim.

## BF16 token agreement · unmatched protocols

<div align="center">
  <img src="assets/top1-pacific-vs-unsloth.svg" alt="Pacific and Unsloth BF16 next-token Top-1 agreement by quantization tier; corpora and protocols are not matched" width="100%" />
</div>

| nominal tier | Pacific agreement | agreements | Wilson 95% CI | Unsloth chart estimate | direct delta |
|---|---:|---:|---:|---:|---|
| IQ3_XXS | **72.000%** | 576/800 | [68.789, 75.001]% | ≈89.2% | not calculated |
| IQ4_XS | **87.625%** | 701/800 | [85.162, 89.729]% | ≈94.0% | not calculated |
| Q4_K_M | **88.500%** | 708/800 | [86.103, 90.529]% | ≈95.8% | not calculated |
| Q5_K_M | **90.750%** | 726/800 | [88.543, 92.568]% | ≈97.2% | not calculated |

Pacific agreement is measured from the completed 800-item runs: for each fixed
MMLU+ARC prompt, the candidate's raw pre-sampling Top-1 token ID is compared with
the BF16 Top-1 token ID. All five evidence files share the same ordered item
fingerprint and pinned `llama.cpp` commit. This is one next-token position per
prompt; it is not task accuracy or generation quality.

The
[Unsloth provider chart](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/blob/4ca720788d1e01f1bff70c033e0d0028fd02e502/README.md)
also reports BF16 next-token Top-1 agreement, but on a different provider
KL-divergence corpus and protocol. Its values above were digitized from the
published PNG and rounded to 0.1 pp because no numeric result table is published.
The green bars are estimates, and the cross-provider differences are not valid
quality deltas. A valid provider comparison still requires running both GGUFs
against the same 800 item IDs and scoring code.

The figure is generated with
[Microsoft Flint Chart 0.5.1](https://github.com/microsoft/flint-chart/tree/34ef4516554b323a740a426bd1a1e6ba31ee8245)
using its Swiss theme. Reproducibility assets:
[measured result](benchmarks/token-agreement-800.json) ·
[calculation script](tools/compute-token-agreement.py) ·
[Flint input](assets/top1-pacific-vs-unsloth.flint.json) ·
[compiled Vega-Lite spec](assets/top1-pacific-vs-unsloth.vegalite.json) ·
[render script](tools/render-flint-top1-comparison.mjs).

## LiveCodeBench v6 · preliminary

> **Execution grading is pending.** The value below is a hard upper bound, not a
> measured pass@1 score. No extraction failure was manually repaired.

| stage | result | status |
|---|---:|---|
| Raw generation | **1,055 / 1,055** | complete; zero empty raw responses |
| Official code extraction | **918 / 1,055** | 87.01% extraction coverage |
| Extraction failures | **137 / 1,055** | retained as automatic failure candidates |
| Measured pass@1 | **pending** | sandboxed execution has not run |
| Hard pass@1 ceiling | **≤87.01%** | best possible before execution grading |
| BF16 model-card reference | **90.3%** | upstream, unmatched protocol |

The best-case cross-protocol gap to the upstream BF16 reference is **≤−3.29 pp**.
The actual gap can only be larger. A matched BF16 run is required before claiming
quantization retention.

<details>
<summary><strong>LiveCodeBench v36 reproducibility profile</strong></summary>

| setting | value |
|---|---|
| Dataset | `release_v6`, 1,055 tasks |
| Task-ID SHA-256 | `6108435b7b39af8ffaf11dc6c206adb6fa2ae92d6f01ea58ae0134ba345fd47a` |
| LiveCodeBench commit | `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24` |
| Runtime | Tesla V100-SXM2-16GB, NVIDIA driver 580.173.02 |
| llama.cpp | `580e88d8b7dece7099d9b62323521d0254ff3615`, build 10716 |
| Sampling | n=1, temperature 1.0, top-p 0.95, top-k 20, min-p 0.0 |
| Penalties | presence 0.0, repetition 1.0 |
| Thinking | enabled; 8,192-token reasoning budget |
| Limits | 16,384 output tokens, 32,768 context, 1,200 s timeout |
| Server | full GPU offload, FP16 KV, batch 512, micro-batch 128, parallel 1 |
| API | local OpenAI-compatible endpoint, Jinja chat template |

The initial 15-item pilot produced two reasoning-only truncations under an
8,192-token total limit. A 16,384-token limit with unrestricted reasoning reproduced
the failure. v36 separated an 8,192-token reasoning budget from final-answer capacity;
its pilot passed 15/15. Thirteen normally terminated pilot responses were retained,
the two truncated responses were regenerated, and 1,040 remaining prompts ran under v36.

[Machine-readable audit →](benchmarks/livecodebench-v6-v36-preliminary.json)

</details>

## Throughput

Measured on one A100 80GB with full GPU offload, prompt 512 / generation 128, five runs.

| arm | prompt tok/s | generation tok/s | peak VRAM |
|---|---:|---:|---:|
| BF16 | 2,423.5 | 26.9 | 50.1 GB |
| IQ3_XXS | 1,232.9 | 50.9 | **11.3 GB** |
| IQ4_XS | 1,338.5 | **57.1** | 14.9 GB |
| Q4_K_M | 1,288.1 | 48.2 | 16.3 GB |
| Q5_K_M | 1,248.0 | 44.2 | 18.7 GB |

## Pinned build

| component | revision |
|---|---|
| Base model | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| llama.cpp | `580e88d8b7dece7099d9b62323521d0254ff3615` |
| Conversion | `convert_hf_to_gguf.py --outtype bf16 --no-mtp` |
| Importance matrix | shared; 20 realized chunks from the pinned 96-record corpus |

The reproducible BF16 text GGUF SHA-256 is
`03ab7ad49486af2f111ed8d7616a0f485f9c5032bd6bc2419b84bb3b90f3930f`.
The BF16 text model and full run evidence are hosted in the
[artifact dataset](https://huggingface.co/datasets/topabaem/mix-stq-artifacts).

## Usage

```bash
llama-cli \
  -m Q4_K_M/qwen38-27b-q4_k_m-00001-of-00003.gguf \
  -p "Explain why low-bit perplexity and task accuracy can disagree."
```

For vision input, add:

```bash
--mmproj mmproj/qwen38-27b-mmproj-bf16.gguf
```

Merge split shards with `llama-gguf-split --merge` if a single file is required.

## Scope

- MMLU+ARC Top-1, WikiText-2 PPL, and A100 throughput are measured results.
- BF16 next-token agreement is measured at 800 fixed prompt positions from the
  preserved pre-sampling token IDs; it is not a generation-quality score.
- LiveCodeBench `87.01%` is an ungraded ceiling, not pass@1.
- The older v0.26 MLP-only IQ3 result is not pooled with this full-model GGUF study.
- This repository is a research measurement, not a universal deployment recommendation.

## License

Apache-2.0, inherited from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
