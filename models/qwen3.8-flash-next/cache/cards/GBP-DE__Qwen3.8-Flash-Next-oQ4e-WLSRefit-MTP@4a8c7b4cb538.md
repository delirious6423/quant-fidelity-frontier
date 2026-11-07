---
license: other
license_name: qwen-community-1.0
license_link: https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
- mlx
- omlx
- oq4e
- imatrix
- mtp
- apple-silicon
- wls-refit
---

# Qwen3.8-Flash-Next — oQ4e WLSRefit MTP | oMLX 0.7.3 Rebuild

> **MODEL UPDATE — November 1, 2026**
>
> This repository has been updated in place with a newly quantized checkpoint created using **oMLX 0.7.3**, replacing the previous oMLX 0.7.1.dev1 build.
>
> **What's new:**
> - Rebuilt with the updated oMLX 0.7.3 quantization pipeline.
> - Incorporates the revised oQe calibration implementation, which follows the model-specific forward pass.
> - Retains importance-aware oQ4e quantization, Weighted Least-Squares (WLS) Refit, iMatrix guidance and MTP support.
> - Preserves non-quantized weights in BF16 where applicable.
>
> oMLX 0.7.3 also introduces precision-allocation improvements for certain model architectures. Their reported quality gains should **not** be assumed to apply directly to this large MoE checkpoint.
>
> **No model-specific accuracy improvement is claimed without controlled testing.**
>
> This is an update to the **existing repository**, not a separate model release. The repository URL remains unchanged.

## About this model

An enhanced, importance-aware **oQ4e quantization** of the original **Qwen3.8-Flash-Next**, designed for inference on Apple Silicon using compatible oMLX runtimes.

The current checkpoint was quantized with **oMLX 0.7.3**.

Its objective is to combine efficient memory usage with high fidelity to the original model, while retaining MTP acceleration support.

## Quantization features

- **oQ4e:** Importance-aware quantization with variable precision.
- **WLS Refit:** Weighted Least-Squares refinement designed to reduce quantization error.
- **iMatrix:** Importance-guided treatment of model weights.
- **MTP:** Multi-Token Prediction weights preserved for compatible accelerated decoding.
- **BF16:** Non-quantized tensors retained in BF16 where applicable.
- **Apple Silicon:** Intended for use with the oMLX inference engine.

The earlier WLS Refit implementation was associated with approximately 11–15% lower KL divergence in the developer's quantizer evaluations. This is a historical, architecture-dependent result, not a measured quality improvement of the current checkpoint.

For details about the updated quantization pipeline, see the [oMLX 0.7.3 release notes](https://github.com/jundot/omlx/releases/tag/v0.7.3).

## Historical testing — previous oMLX 0.7.1.dev1 build

**Important:** The following observations relate to the previous oQ4e WLSRefit MTP checkpoint, created with oMLX 0.7.1.dev1.

They are preserved for transparency and historical reference. They must not be interpreted as benchmark results for the new oMLX 0.7.3 rebuild.

### Tested and reviewed with GPT-6

The previous checkpoint was evaluated using demanding reasoning and analytical challenges, with its outputs reviewed by GPT-6.

Compared with an earlier oQ4e conversion of the same base model, the previous WLSRefit build showed a noticeable improvement in practical output quality during local testing.

In the 60-question **Silbergrat Reasoning Challenge**, it demonstrated:

- Strong logical coherence and evidence evaluation.
- Substantially fewer serious reasoning inconsistencies.
- Much better language stability than the earlier conversion.
- Careful handling of uncertainty and competing explanations.

A separate **XAGUSD quantitative trading research challenge** also demonstrated strong analytical capabilities, producing testable hypotheses without inventing profitability claims.

**Historical assessment:** The previous WLSRefit build showed promising improvements in practical reasoning quality, particularly for complex, long-form analytical tasks.

These observations were qualitative, not standardized benchmarks. The runtime version also changed, so the observed differences could not be attributed exclusively to WLS Refit.

### Current build evaluation

The oMLX 0.7.3 rebuild has not yet been independently evaluated against the previous checkpoint.

No improvement in reasoning accuracy, reliability or language quality is claimed at this time.

## Compatibility

Requires a compatible [oMLX](https://github.com/jundot/omlx) runtime.

This is an **oMLX/oQ checkpoint**, not a GGUF model.

Lightning MTP acceleration depends on runtime and configuration support.

## Credits and License

- **Original model:** [Qwen Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Quantization software:** [oMLX](https://github.com/jundot/omlx)
- **Quantized and published by:** GBP-DE
- **License:** Qwen Community License 1.0. Original license terms apply.