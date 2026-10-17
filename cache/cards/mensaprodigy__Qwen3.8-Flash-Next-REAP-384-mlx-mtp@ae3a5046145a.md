---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
library_name: mlx
tags:
  - mlx
  - omlx
  - qwen4-exp
  - mixture-of-experts
  - mtp
  - apple-silicon
---

# Qwen3.8-Flash-Next REAP-384 oQ4e + Native MTP

An Apple Silicon–oriented oMLX checkpoint built from **Qwen3.8-Flash-Next** with the target MoE expert bank REAP-pruned from 512 to 384 experts per target layer, while retaining the original native 512-expert MTP predictor. The combined checkpoint was converted with **oMLX oQ4e mixed-precision quantization**. It is intended for oMLX's Qwen4-Exp runtime, with native Lightning MTP for decode-time acceleration.[1][3][4]

## Architecture at a glance

| Component | Published configuration |
|---|---|
| Runtime architecture | `qwen4_exp` / `Qwen4ExpForConditionalGeneration` |
| Target transformer | 48 target layers |
| Target MoE pool | 384 available experts per layer |
| Target routing | top-10 (`num_experts_per_tok=10`) |
| Native MTP | 1 MTP layer, 512 available experts |
| MTP routing | top-10 (`mtp_num_experts_per_tok=10`) |
| Quantization | oMLX oQ4e mixed precision; primary 4-bit affine, group size 64 |
| Final package | Flattened repository root; `model.safetensors.index.json` present |

The published `config.json` records `num_experts=384`, `num_experts_per_tok=10`, `mtp_num_experts=512`, `mtp_num_experts_per_tok=10`, and `mtp_num_hidden_layers=1` under `text_config`. The final quantized tensor index contains native `mtp.*` tensors. The asymmetric expert counts are intentional.

## Why REAP-384 + full native MTP?

The target model has a routed expert bank across 48 layers. REAP-384 reduces the target pool from 512 to 384 experts per layer: **128 fewer target experts × 48 layers = 6,144 removed target-expert instances**.

This does **not** reduce routing top-k:

- **Target:** 384 available experts; 10 routed per token.
- **MTP:** 512 available experts; 10 routed per token.

The MTP predictor contains only one MoE layer, so this release retains its complete original 512-expert pool. The design goal is to capture most of REAP-384's storage and memory benefit, retain more target capacity than REAP-288, and preserve the original native speculative predictor.

## Provenance and build lineage

```text
Qwen/Qwen3.8-Flash-Next BF16
        |
        +-- extract original native 512-expert MTP
        |
        v
Sh0wie REAP-384 BF16 target
        |
        +-- embed native Qwen MTP
        |
        v
REAP-384 BF16 + 512-expert native MTP
        |
        v
oMLX oQ4e conversion
        |
        v
REAP-384 oQ4e + embedded native MTP
```

The target side descends from `sh0wie/Qwen3.8-Flash-Next-REAP-384-bf16`. The MTP predictor was taken directly from the official Qwen BF16 checkpoint before quantization. The original native MTP export comprised 31 `mtp.*` tensors distributed across the official safetensor shards; after oQ quantization, the final index exposes the corresponding quantized MTP tensors, including weights, scales, and biases.

This is **not** a generic MLX Q4 conversion. It is an oMLX oQ4e mixed-precision checkpoint. The primary configuration is 4-bit affine quantization with group size 64; the published `quantization_config` also contains 392 per-tensor overrides. PLE/N-gram settings intentionally differ from the main-model defaults where the config specifies them—for example, published PLE/N-gram shard entries use 4-bit/group-32 settings. Inspect `config.json` for the authoritative per-tensor configuration.

## oMLX compatibility

This model requires Qwen4-Exp support for **independent target and MTP expert counts**: `num_experts=384` and `mtp_num_experts=512`.

Use an oMLX build from `main` that includes both:

1. [PR #3659: `fix: honor Qwen4-Exp MTP expert counts`](https://github.com/jundot/omlx/pull/3659), which adds independent MTP expert-count fields and regression coverage for target 384 / MTP 512.[3]
2. The subsequent [MTP expert-weight stacking fix](https://github.com/jundot/omlx/commit/eee176f6296b3a0546cbdb372bfbf3701b17408f), which applies `mtp_num_experts` when stacking MTP expert weights.[4]

The latest official release verified for this card is `v0.6.4`; it predates both required changes. The `v0.7.0.dev2` development tag also predates the stacking fix. Do **not** assume either tag can load this asymmetric configuration. Until an official release contains both changes, use a current oMLX `main` build that includes them.

## Recommended oMLX configuration

For native decode-time MTP:

```text
Lightning MTP:             ON
External decode draft:     NONE
DFlash:                    OFF (unless independently validated)
```

### Embedded MTP: no separate draft model required

The native MTP predictor is embedded in this checkpoint. The intended package includes:

- REAP-384 target weights
- PLE/N-gram weights
- vision components
- native 512-expert MTP weights

Users should not download a separate MTP/draft checkpoint for Lightning MTP.

**SpecPrefill is different from native MTP.** A tokenizer-compatible small Qwen draft may optionally accelerate prompt/prefill work where supported; the embedded MTP accelerates decode/token generation.

```text
Prompt
  |
  v
Optional small Qwen SpecPrefill draft
  |
  v
REAP-384 target
  |
  v
Embedded 512-expert native MTP
  |
  v
Lightning MTP decode
```

These mechanisms perform different jobs and can be complementary when validated in the selected runtime.

## REAP-lineage evaluation (not this checkpoint)

This exact oQ4e checkpoint with embedded native MTP has **not** yet received an independently run broad standardized evaluation suite. The figures below are reported by Sh0wie for the related REAP pruning lineage; they are **not** direct measurements of this release.[2]

| Related REAP lineage build | Expert pool | Reported HumanEval pass@1 |
|---|---:|---:|
| Full Flash-Next Q4 | 512 | 93.9% |
| REAP-384 Q4 | 384 | 92.1% |
| REAP-288 Q4 | 288 | 91.5% |
| REAP-256 Q4 | 256 | 88.4% |

Sh0wie reports a common harness and machine, 164 HumanEval problems, unit-test verification, and one run per build; that card also cautions that single-run results have no confidence intervals.[2] Treat these as lineage context, not a substitute for evaluating this oQ4e + embedded-MTP release on your workload.

## Observed local service telemetry (not a benchmark)

A cumulative oMLX REAP service counter for this exact model recorded 39 requests, 1,253,641 prompt tokens, and 21,233 completion tokens. Derived rates were:

- **269.20 uncached prefill tokens/s**
- **33.62 generated tokens/s**
- **2,491.89 prompt tokens/s including cached-prompt accounting**

These are operational counters rather than a controlled reproducible benchmark. Context length, cache-hit rate, PLE/N-gram residency, SSD mmap/offload, Lightning MTP settings, concurrency, and oMLX version can materially change results.

## Why 384 instead of 288?

Both REAP-384 and REAP-288 retain top-10 routing. REAP-288 removes 43.75% of the original 512 target experts and is attractive where memory capacity is the dominant constraint. REAP-384 removes 25% and is intended as a more capacity-oriented point on the pruning curve for systems with greater unified memory, including higher-memory Apple Silicon configurations. Neither is universally better: choose according to memory budget, workload, and independent evaluation.

## Hardware

Primary target: Apple Silicon running oMLX. High unified-memory configurations are recommended. Do not infer a fixed RAM minimum from the model card: actual memory use varies with context length, KV cache configuration, PLE/N-gram residency, SSD mmap/offload, Lightning MTP settings, SpecPrefill, concurrency, and oMLX version.

## Vision

The Qwen vision configuration and vision-tower tensors remain in the checkpoint. Post-REAP multimodal quality has not been independently characterized for this release. Users with vision-critical workloads should validate their own image and multimodal tasks before deployment.

## Limitations

- **REAP calibration:** Expert pruning reflects a particular calibration/workload distribution and may affect uncommon routing patterns differently.
- **Rare knowledge:** Rare proper nouns, niche knowledge, and low-frequency concepts can be more sensitive to pruning.
- **Quantization:** oQ4e mixed precision can introduce small deviations from BF16 behavior.
- **Evaluation scope:** The REAP-lineage table above is not a direct benchmark of this checkpoint.
- **Runtime compatibility:** This package targets oMLX's Qwen4-Exp implementation. Compatibility with MLX-Serve-specific runtimes or transformed-checkpoint loaders is not implied.

## Credits and acknowledgements

- **Qwen Team:** original Qwen3.8-Flash-Next architecture, weights, tokenizer, PLE/N-gram and multimodal components, and native MTP.[1]
- **Sh0wie:** REAP-384 expert pruning, expert-selection/calibration work, BF16 REAP-384 source checkpoint, and published REAP-lineage evaluation.[2]
- **Jundot and oMLX contributors:** oMLX, oQ quantization, Apple Silicon inference, Qwen3.8 Flash-Next runtime support, Lightning MTP, SSD-backed PLE, and the independent target/MTP expert-count implementation. PR #3659 was authored by `mensaprodigy`; the required follow-up stacking fix was authored by `jundot`.[3][4]

## License

The weights are derived from Qwen/Qwen3.8-Flash-Next and are distributed under the **Qwen Community License 1.0**, not Apache-2.0. See [LICENSE](LICENSE) for the upstream license text.[5]

This card is not legal advice. Review the license before commercial use, especially for Model-as-a-Service or AI Work Assistant offerings and the license's scale-related requirements.

## Sources
## Sources

[1] https://huggingface.co/Qwen/Qwen3.8-Flash-Next — Qwen3.8-Flash-Next
[2] https://huggingface.co/sh0wie/Qwen3.8-Flash-Next-REAP-384-MLX-4bit/raw/main/README.md — Sh0wie REAP-384 model card
[3] https://github.com/jundot/omlx/pull/3659 — oMLX PR #3659
[4] https://github.com/jundot/omlx/commit/eee176f6296b3a0546cbdb372bfbf3701b17408f — oMLX MTP stacking fix
[5] https://huggingface.co/Qwen/Qwen3.8-Flash-Next/raw/main/LICENSE — Qwen Community License 1.0

