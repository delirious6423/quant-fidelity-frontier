---
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
license: apache-2.0
library_name: llama.cpp
pipeline_tag: image-text-to-text
tags:
- gguf
- opentq
- turboquant
- qwen
- qwen3.8
- multimodal
- vision
- mtp
- speculative-decoding
- dynamic-quantization
- llama.cpp
---

![Qwen3.8 OpenTQ — dynamic tensor allocation](assets/opentq-qwen38-hero.png)

# Qwen3.8-27B OpenTQ GGUF

[![Base model](https://img.shields.io/badge/base-Qwen3.8--27B-201d19?labelColor=fcfaf6)](https://huggingface.co/Qwen/Qwen3.8-27B)
[![Multimodal](https://img.shields.io/badge/image%20input-verified-2c7a64?labelColor=fcfaf6)](#verified-multimodal-path)
[![MTP](https://img.shields.io/badge/MTP-fast%20inference-verified-2f62b7?labelColor=fcfaf6)](#mtp-fast-inference)
[![Quality](https://img.shields.io/badge/paired%20BF16-Q4%20%2B%20Q5%20pass-c37832?labelColor=fcfaf6)](#paired-bf16-quality)

A **multimodal llama.cpp release** of Qwen3.8-27B with role-aware OpenTQ quantization and a measured MTP fast-inference path. The package follows llama.cpp's native component model:

1. an OpenTQ language trunk (`Q4_K_M` or `Q5_K_M`);
2. a vision projector (`mmproj-BF16` recommended, `mmproj-F16` alternative);
3. an MTP draft only where the measured per-trunk compatibility gate passes (`Qwen3.8-27B-MTP-Q8_0.gguf` for Q4).

The trunk is not advertised as self-contained vision. Pairing it with `mmproj` is what makes the runtime multimodal; pairing the same trunk with the MTP draft enables verified speculative decoding.

## Choose the language trunk

| Download | Size | Paired BF16 signal | Measured CUDA | Use case |
|---|---:|---:|---:|---|
| [Q4_K_M](https://huggingface.co/zlaabsi/Qwen3.8-27B-OTQ-GGUF/resolve/main/Qwen3.8-27B-OTQ-DYN-Q4_K_M.gguf) | 16.22 GiB | 159/232, Δ 0.00 pp | 1,279.93 pp · 50.15 tg tok/s | **Recommended default** |
| [Q5_K_M](https://huggingface.co/zlaabsi/Qwen3.8-27B-OTQ-GGUF/resolve/main/Qwen3.8-27B-OTQ-DYN-Q5_K_M.gguf) | 19.61 GiB | 158/232, Δ -0.43 pp | 1,232.90 pp · 44.20 tg tok/s | More precision |

Q3 is intentionally absent: v2 reached 154/232, Δ -2.16 pp, and failed the unchanged -1.00 pp release limit.

## Add vision and MTP

| Component | Size | Coverage | Status |
|---|---:|---:|---|
| [`mmproj-BF16.gguf`](https://huggingface.co/zlaabsi/Qwen3.8-27B-OTQ-GGUF/resolve/main/mmproj-BF16.gguf) | 0.87 GiB | 333 source vision tensors → 334 GGUF tensors | **Recommended · image verified** |
| [`mmproj-F16.gguf`](https://huggingface.co/zlaabsi/Qwen3.8-27B-OTQ-GGUF/resolve/main/mmproj-F16.gguf) | 0.86 GiB | 333 source vision tensors → 334 GGUF tensors | Image verified with Q4 |
| [`Qwen3.8-27B-MTP-Q8_0.gguf`](https://huggingface.co/zlaabsi/Qwen3.8-27B-OTQ-GGUF/resolve/main/Qwen3.8-27B-MTP-Q8_0.gguf) | 2.95 GiB | 15 source MTP tensors + 3 required trunk tensors | **Q4 selected by measured search** |

Every component was converted from [`Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0) with stock [`llama.cpp@7e4c0a96880dae4fc4268ad441f8a6446bd5460a`](https://github.com/ggml-org/llama.cpp/commit/7e4c0a96880dae4fc4268ad441f8a6446bd5460a). Size and SHA-256 are recorded in [`release-manifest.json`](release-manifest.json).

## Quick start

Download one trunk and the two recommended sidecars:

```bash
hf download zlaabsi/Qwen3.8-27B-OTQ-GGUF \
  Qwen3.8-27B-OTQ-DYN-Q4_K_M.gguf \
  mmproj-BF16.gguf \
  Qwen3.8-27B-MTP-Q8_0.gguf \
  --local-dir models/qwen38-opentq
```

Multimodal server with MTP:

```bash
llama-server \
  -m models/qwen38-opentq/Qwen3.8-27B-OTQ-DYN-Q4_K_M.gguf \
  --mmproj models/qwen38-opentq/mmproj-BF16.gguf \
  --model-draft models/qwen38-opentq/Qwen3.8-27B-MTP-Q8_0.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --n-gpu-layers all --n-gpu-layers-draft all \
  --flash-attn on --ctx-size 8192 \
  --host 127.0.0.1 --port 8080
```

Text-only use remains possible by omitting `--mmproj`, `--model-draft` and the speculative-decoding flags.

## MTP fast inference

![MTP fast inference benchmark](assets/charts/mtp-fast-inference.png)

The frozen A100 search compared BF16, Q8_0 and Q4_K_M MTP drafts at `n_max` 2, 4 and 6. A configuration was eligible only when it accepted draft tokens and reproduced the ordinary greedy output exactly.

| Trunk | Ordinary decode | Selected MTP | Speedup | Output gate |
|---|---:|---:|---:|---|
| Q4_K_M | 49.05 tok/s | 62.36 tok/s | **+27.15%** · acceptance 60.2% | Exact parity |
| Q5_K_M | 43.55 tok/s | Per-trunk search | blocked: no exact-parity positive-speedup configuration | Not published as MTP-compatible |

These numbers are device- and prompt-specific measurements, not a universal speed claim. The complete search matrix is preserved in [`evidence/multimodal-mtp-cuda.json`](evidence/multimodal-mtp-cuda.json).

## Verified multimodal path

The supplied release hero was used as the frozen image fixture. Every case had to transcribe `Qwen3.8 OpenTQ` and produce a non-empty visual description. The two Q4 cases additionally had to exercise MTP and accept at least one draft token; Q5 was tested without MTP because its independent exact-parity search did not pass.

| Language trunk | Projector | Image understanding | MTP used |
|---|---|---|---|
| Q4_K_M | BF16 | PASS | PASS |
| Q4_K_M | F16 | PASS | PASS |
| Q5_K_M | BF16 | PASS | Not claimed |

This establishes **image input** for the published package. Video input was not evaluated and is not claimed.

## Paired BF16 quality

![Paired BF16 vs GGUF quality signal](assets/charts/paired-quality-signal.png)

The same 232 frozen prompts and deterministic `qwen3-no-think` policy were run against BF16 and every candidate. The aggregate paired delta is the release gate; the 13 smaller benchmark slices remain diagnostic.

| Variant | BF16 → GGUF | Delta and bootstrap 95% interval | Gate |
|---|---:|---:|---|
| Q4_K_M | 159 → 159 / 232 | 0.00 pp, [-2.16, +2.16] | PASS · limit -0.75 pp |
| Q5_K_M | 159 → 158 / 232 | -0.43 pp, [-2.16, +1.29] | PASS · limit -0.50 pp |

![Per-benchmark paired deltas](assets/charts/paired-benchmark-deltas.png)

<details>
<summary>Exact per-benchmark counts</summary>

| Benchmark | BF16 | Q4, delta | Q5, delta |
|---|---:|---:|---:|
| mmlu, n=16 | 14 | 13, -6.25 pp | 14, +0.00 pp |
| mmlu_pro, n=24 | 14 | 16, +8.33 pp | 15, +4.17 pp |
| arc, n=16 | 16 | 16, +0.00 pp | 16, +0.00 pp |
| hellaswag, n=16 | 16 | 15, -6.25 pp | 16, +0.00 pp |
| gsm8k, n=16 | 5 | 5, +0.00 pp | 4, -6.25 pp |
| math, n=16 | 6 | 6, +0.00 pp | 6, +0.00 pp |
| bbh, n=24 | 19 | 19, +0.00 pp | 18, -4.17 pp |
| gpqa, n=24 | 0 | 0, +0.00 pp | 0, +0.00 pp |
| truthfulqa, n=16 | 15 | 15, +0.00 pp | 15, +0.00 pp |
| winogrande, n=16 | 14 | 14, +0.00 pp | 14, +0.00 pp |
| drop, n=16 | 12 | 12, +0.00 pp | 12, +0.00 pp |
| piqa, n=16 | 14 | 14, +0.00 pp | 14, +0.00 pp |
| commonsenseqa, n=16 | 14 | 14, +0.00 pp | 14, +0.00 pp |

</details>

Machine-readable evidence: [`evidence/paired-quality-report.json`](evidence/paired-quality-report.json).

## What OpenTQ changes

OpenTQ does not force every matrix into one nominal type. It allocates precision by tensor role, records the exact 851-tensor map, then checks the emitted GGUF tensor-by-tensor.

![Dynamic GGUF tensor-type allocation](assets/charts/tensor-type-allocation.png)

![Where OpenTQ spends precision](assets/charts/precision-spend.png)

The precision plot is **count-weighted and categorical**; it is not presented as storage-weighted model bpw. Norms and small state/conv tensors requested as F16 remain F32 where stock llama.cpp preserves them.

## Measured llama.cpp CUDA throughput

![Measured llama.cpp CUDA throughput](assets/charts/cuda-throughput.png)

One NVIDIA A100-SXM4-80GB, full offload, FlashAttention, five `llama-bench` repetitions. `pp8192` measures 8K-token prompt processing; `tg128` measures 128-token generation. No cross-hardware extrapolation is made.

## Reproducibility

| Item | Frozen identity |
|---|---|
| Source | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| Language trunk conversion | `llama.cpp@1692f9e50bb20fd96b963af38a282daf78feea64` |
| Multimodal/MTP conversion and validated runtime | `llama.cpp@7e4c0a96880dae4fc4268ad441f8a6446bd5460a` |
| Language tensor coverage | 851 / 851, zero unmatched or type mismatches |
| Vision source coverage | 333 / 333 |
| MTP source coverage | 15 / 15 |
| Paired quality | 232 prompts across 13 benchmark subsets |

Evidence index:

- [`release-manifest.json`](release-manifest.json) — downloadable artifacts, hashes, revisions and claims.
- [`evidence/component-build.json`](evidence/component-build.json) — vision/MTP conversion inventory.
- [`evidence/multimodal-mtp-cuda.json`](evidence/multimodal-mtp-cuda.json) — MTP matrix and joint image + MTP tests.
- [`evidence/paired-quality-report.json`](evidence/paired-quality-report.json) — paired outcomes and intervals.
- [`assets/charts/chart-manifest.json`](assets/charts/chart-manifest.json) — figure hashes and source bindings.

## Limits

- Q3 remains blocked and is not published.
- Image input is verified; video input is not evaluated.
- CUDA figures describe the measured A100 setup only.
- The paired suite is a frozen release-regression signal, not a complete capability evaluation.
- Metal/MLX and NVFP4 are deliberately outside this llama.cpp-first release phase.

## License

Apache-2.0, following the source model. See [`LICENSE`](LICENSE).
