---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - mlx
  - omlx
  - oq5e
  - imatrix
  - mtp
  - apple-silicon
  - wls-refit
---

# Qwen3.8-Flash-Next — oQ5e WLSRefit MTP | oMLX 0.7.3 Rebuild

> **MODEL UPDATE — November 1, 2026**
>
> This repository has been updated in place with a newly quantized checkpoint created using **oMLX 0.7.3**, replacing the previous oMLX 0.7.1.dev1 build.
>
> **What's new:**
> - Rebuilt with the updated oMLX 0.7.3 quantization pipeline.
> - Incorporates the revised oQe calibration implementation, which follows the model-specific forward pass.
> - Retains importance-aware oQ5e quantization, Weighted Least-Squares (WLS) Refit, iMatrix guidance and MTP support.
> - Preserves non-quantized weights in BF16 where applicable.
>
> oMLX 0.7.3 also introduces precision-allocation improvements for certain model architectures. Their reported quality gains should **not** be assumed to apply directly to this large MoE checkpoint.
>
> **No model-specific accuracy improvement is claimed without controlled testing.**
>
> This is an update to the **existing repository**, not a separate model release. The repository URL remains unchanged.

## About this model

A quality-focused, importance-aware **oQ5e quantization** of the original **Qwen3.8-Flash-Next**, created for inference on Apple Silicon using compatible oMLX runtimes.

The current checkpoint was quantized with **oMLX 0.7.3**.

Compared with oQ4e, oQ5e prioritizes higher weight precision and model fidelity at the cost of increased memory requirements.

The goal is to preserve the original model's capabilities while retaining support for accelerated MTP decoding.

## Quantization features

- **oQ5e:** Importance-aware quantization with a higher precision budget than oQ4e.
- **WLS Refit:** Weighted Least-Squares refinement designed to reduce quantization error.
- **iMatrix:** Importance-guided treatment of model weights.
- **MTP:** Multi-Token Prediction weights preserved for compatible accelerated decoding.
- **BF16:** Non-quantized tensors retained in BF16 where applicable.
- **Apple Silicon:** Intended for use with the oMLX inference engine.

The earlier WLS Refit implementation was associated with approximately 11–15% lower KL divergence in the developer's quantizer evaluations. This is a historical, architecture-dependent result, not a measured quality improvement of the current checkpoint.

For technical details, see the [oMLX 0.7.3 release notes](https://github.com/jundot/omlx/releases/tag/v0.7.3).

## Historical reasoning evaluation — previous builds

**Important:** The following observations relate to earlier local quantizations and tests conducted before the oMLX 0.7.3 rebuild.

They do not constitute benchmark results for the current oQ5e checkpoint.

### Earlier testing with GPT-6

Previous local testing with GPT-6 found that an oQ4e conversion using the newer WLS Refit approach produced substantially more coherent results than an earlier oQ4e conversion of the same base model.

The demanding 60-question **Silbergrat Reasoning Challenge** demonstrated:

- Improved language stability.
- Strong logical coherence and evidence evaluation.
- More careful handling of uncertainty.
- Fewer serious reasoning inconsistencies.

The earlier oQ5e WLSRefit checkpoint used the same quantization approach with higher precision, aiming to preserve more of the original model's capabilities.

However, that oQ5e build was not independently benchmarked against the other variants.

The observed improvements in the historical oQ4e comparison could not be attributed exclusively to WLS Refit because the runtime version also changed.

These findings are qualitative observations rather than standardized benchmark results.

### Current build evaluation

The new **oMLX 0.7.3 oQ5e rebuild** has not yet been independently evaluated against the previous checkpoint.

Its higher weight precision compared with oQ4e is a technical characteristic, not proof of superior reasoning performance in every task.

No improvement in reasoning accuracy, reliability or language quality is claimed at this time.

## Compatibility

Requires a compatible [oMLX](https://github.com/jundot/omlx) runtime.

This is an **oMLX/oQ checkpoint**, not a GGUF model.

Lightning MTP acceleration depends on runtime and configuration support.

Due to the model's higher memory requirements, sufficient unified memory is essential, especially when using large context windows.

## Credits and License

- **Original model:** [Qwen Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Quantization software:** [oMLX](https://github.com/jundot/omlx)
- **Quantized and published by:** GBP-DE
- **License:** Qwen Community License 1.0. Original license terms apply.
