---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - auto-round
  - int4
  - w4a16
  - g128
  - intel-xpu
---

# Qwen3.8-27B-AutoRound-W4A16-G128

**27B model. INT4 weights. One 32 GB B65.**

An **Intel Arc Pro B65/B70-targeted AutoRound quantization** for vLLM XPU serving, distributed in **safetensors** with GPTQ-compatible packing. Quantized directly from the [original Qwen3.8-27B BF16 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), with pinned source and creation provenance below.

**Tested on B65 using patched vLLM XPU. B70 is a deployment target, not a tested result.**

## At a glance: one B65

| Measured result | Workload |
|---|---|
| **63.16 tok/s decode** | One ~4K request · MTP4 |
| **2.08× no-MTP decode rate** | Same short-context workload · 63.16 vs 30.31 tok/s |
| **~3 s prefill · ~1.37K tok/s** | Fresh ~4K prompt · MTP4 · no cache hits |
| **125.97 tok/s aggregate decode** | Four concurrent ~32K requests · MTP3 |
| **262,144-token request budget** | 261,616 prompt + 512 output + 16 reserve · MTP3 |

**All headline rates are for one 32 GB Intel Arc Pro B65.** Single-request decode rates exclude prefill; concurrent throughput is the aggregate over the streams' common decode interval. The full-context workload took **13.3–13.6 minutes to prefill** without cache hits. Full measurements and timing definitions appear below.

Independent quantization and qualification by **[Ryan Weddle](https://huggingface.co/weddle)**. Original model by **[Qwen](https://huggingface.co/Qwen/Qwen3.8-27B)**; tooling by **[Intel AutoRound](https://github.com/intel/auto-round)**. Not an official Qwen or Intel release.

## Artifact

| Property | Value |
|---|---|
| Quantization | Symmetric INT4, group size 128, W4A16 |
| Coverage | 400 language-model matrices |
| Export format | `auto_round:auto_gptq` |
| Scale dtype | FP16 |
| Unquantized tensors | 799 BF16 tensors, including vision and all 15 MTP tensors |

> **HF sidebar:** the “6B” count reflects packed storage, not the model's logical 27B weights. INT4 weights are packed into `I32` tensors; `BF16` holds preserved weights and `F16` holds quantization scales.

Embeddings, the output head, vision, MTP, normalization, and selected recurrent-attention projections remain unquantized. AutoRound identifies the quantization method; GPTQ-compatible packing identifies the export format.

**MTP is stored in BF16.** Preserve `model_extra_tensors.safetensors` and its associated configuration. The INT4 MTP/draft-head treatment tested during inference is a separate runtime operation.

## Creation provenance

Quantized directly from the [pinned BF16 source](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0).

- **Method:** AutoRound `0.16.0.dev136+gce90add6`, BF16 calibration activations.
- **Calibration:** [NeelNanda/pile-10k](https://huggingface.co/datasets/NeelNanda/pile-10k), 128 samples × 2,048 tokens.
- **Tuning:** 200 iterations, seed 42, batch size 8, gradient accumulation 1.
- **Execution:** Torch compilation and low-memory modes enabled; NVIDIA A100 80GB.
- **Software:** PyTorch `2.6.0+cu124`, Transformers `5.8.0`, Triton `3.2.0`.

<details>
<summary>Exact revisions and calibration checksum</summary>

- **Model:** [1d4bf0f2ff60](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
- **AutoRound:** [ce90add61e93](https://github.com/intel/auto-round/commit/ce90add61e937f55b04bc51e8464c3a6332c37d5)
- **Dataset:** [127bfedcd504](https://huggingface.co/datasets/NeelNanda/pile-10k/tree/127bfedcd5047750df5ccf3a12979a47bfa0bafa)

Full revisions are encoded in the links. Recorded calibration-token file SHA-256:

```text
338b5b962c754e137501d39859632e7e3175ae1f0c35ed57c9e2c4d46080e86d
```

</details>

The export metadata abbreviates AutoRound's version to `0.16.0`; the development version and commit above identify the creation software. The preserved transfer receipt records successful SHA-256 verification of all 76 export files.

## B65 serving qualification

### Hardware and runtime

**All throughput rates below are for one Intel Arc Pro B65 with 32 GB GDDR6**, not a combined multi-GPU rate. [Intel's B65 datasheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b65-gpu.pdf) specifies **608 GB/s memory bandwidth and 197 peak dense INT8 TOPS**. TOPS is a vendor XMX compute rating, not measured W4A16 throughput.

Tested with **patched vLLM XPU and LMCache**, FP16 target compute, native W4A16 kernels, FP8 KV, and PIECEWISE compilation. MTP-enabled runs used separate runtime INT4 MTP/draft-head handling.

### Decode throughput

| Workload on one B65 | Without MTP | MTP3 | MTP4 |
|---|---:|---:|---:|
| ~4,094-token prompt, one request | 30.31 | 59.28 | **63.16** |
| 261,616-token prompt, one request | Not measured | **21.28** | 18.72 |
| Four concurrent 32,240-token prompts | Not measured | **125.97** | 122.35 |

Rates are **tokens/second**. Single-request rates measure streamed decode, excluding prefill; the four-request row is **per-GPU aggregate common-overlap decode**, not per-request throughput.

Short-context results used 2,048 generated tokens in the September 8, 2026 run. Long-context results used 512 generated tokens and geometric means over two counterbalanced September 9 rounds. Different workloads and runs make this a serving summary, not a controlled context-scaling experiment.

**Recommended for these tested workloads:** MTP4 at short context; MTP3 at long context and four-request concurrency. Only MTP3 and MTP4 were compared at the longer lengths; neither is a universal optimum.

### Uncached prefill

**One request on one B65, with zero local or external cache hits.**

| Prompt tokens | MTP | Time (s) | Rate (tok/s) |
|---:|---|---:|---:|
| 4,094 | Off | 2.913 | 1,405.3 |
| 4,094 | MTP3 | 2.993 | 1,367.7 |
| 4,094 | MTP4 | 2.994 | 1,367.2 |
| 261,616 | MTP3 | 810.45–814.47 | 321.2–322.8 |
| 261,616 | MTP4 | 797.72–802.17 | 326.1–328.0 |

Time is vLLM's reported request-prefill duration; rate is prompt tokens divided by that duration, not isolated kernel throughput. Short-context rows are one measured request per configuration after warmup. Long-context rows show ranges over two runs, not averages. Long-context prefill without MTP was not measured.

A fresh ~4K prompt took about **3 seconds** to prefill; a fresh near-262K prompt took **13.3–13.6 minutes**. These are uncached prompt-processing times, separate from decode throughput and populated-prefix cache restoration.

### Context capacity

- **Observed warmed KV allocation:** 275,324 token slots on one B65 with the MTP3 profile. This was reported after compiled-artifact reload, not guaranteed on a cold start.
- **Configured per-request limit:** 262,144 tokens. The full-context workload budget was 261,616 prompt + 512 output + 16 reserve.
- **Four-request workload:** four 32,768-token budgets, totaling **131,072 tokens per GPU**, including output allowances and reserves.
- **Two-replica full-context test:** simultaneous full-context requests on two independent B65s, totaling **524,288 tokens of request budgets**. This is not pooled GPU memory or a 524K single-request context.

The tests also passed eight-request concurrency across two replicas, persisted hybrid attention/recurrent-state restoration, mixed local-prefix/disk restoration, and cancellation followed by recovery.

Accepted restoration outputs matched cold-reference token arrays exactly, with zero reported scheduler preemptions. Coverage combines successful checks across runs and runtime corrections, not a complete rerun on one final runtime revision.

## Limits

- **Runtime-specific:** compatibility with unmodified upstream vLLM or other loaders is unverified.
- **Not a zero-quality-loss claim:** serving and cache correctness do not establish broad equivalence to the BF16 source. Upstream benchmark scores are not results for this quantization.
- **Text-tested:** vision tensors are retained, but image/video quality was not evaluated in this serving campaign. The multimodal task tag describes the architecture, not evaluation coverage.
- **Workload-specific:** the tested concurrency does not establish capacity for four maximum-length requests per GPU.

## License

The pinned upstream model card identifies **Apache-2.0**. Retain the applicable upstream license and notices when redistributing. This independent derivative credits Qwen and documents its quantization changes above.
