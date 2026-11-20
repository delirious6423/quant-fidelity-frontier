
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

# Qwen3.8-Flash-Next — oQ4e WLSRefit MTP | oMLX 0.8.0 Rebuild

| 🚀 **MODEL UPDATE — November 19, 2026** |
| :--- |
| **Fresh 4-bit rebuild with oMLX 0.8.0.** This repository has been updated in place, replacing the November 1 build made with oMLX 0.7.3. |
| **Why update?** Jun Kim's refined **oQe quantization** is designed to reduce quantization error at the same nominal precision and model size. Re-quantization is needed to benefit from these improvements; updating oMLX alone does not change existing model weights. |
| **Retained features:** oQ4e, WLS Refit, iMatrix-guided quantization, MTP weights and BF16 for unquantized tensors where applicable. |
| **Initial local tests:** OBSIDIAN-40 and SILBERGRAT produced generally strong reasoning results. Generation speed improved slightly in these runs, but total execution time increased. A consistent accuracy gain over the previous checkpoint has **not** been established. |
| **Already downloaded an earlier build?** Download the model files again. The repository URL remains unchanged. See also the [5-bit WLSRefit MTP build](https://huggingface.co/GBP-DE/Qwen3.8-Flash-Next-oQ5e-WLSRefit-MTP) and the [oMLX 0.8.0 release notes](https://github.com/jundot/omlx/releases/tag/v0.8.0). |

## About this model

An enhanced, importance-aware **oQ4e quantization** of the original **Qwen3.8-Flash-Next**, intended for inference on Apple Silicon using compatible oMLX runtimes.

The current checkpoint was rebuilt with **oMLX 0.8.0**. It aims to combine efficient memory use with high fidelity to the original model while retaining MTP acceleration support.

## Quantization features

- **oQ4e:** Importance-aware quantization with variable precision.
- **WLS Refit:** Weighted Least-Squares refinement designed to reduce quantization error.
- **iMatrix:** Importance-guided treatment of model weights.
- **MTP:** Multi-Token Prediction weights preserved for compatible accelerated decoding.
- **BF16:** Unquantized tensors retained in BF16 where applicable.
- **Apple Silicon:** Intended for use with [oMLX](https://github.com/jundot/omlx).

The earlier WLS Refit implementation was associated with approximately 11–15% lower KL divergence in the developer's quantizer evaluations. This is a **historical, architecture-dependent result**, not a measured quality gain for this checkpoint.

For the new oQe quantization method, see the [oMLX 0.8.0 release notes](https://github.com/jundot/omlx/releases/tag/v0.8.0). For the earlier rebuild, see the [oMLX 0.7.3 release notes](https://github.com/jundot/omlx/releases/tag/v0.7.3).

## Update history

- **November 19, 2026 — oMLX 0.8.0:** Fresh quantization using the refined oQe method. Same model family and repository URL. No confirmed model-specific accuracy improvement claimed.
- **November 1, 2026 — oMLX 0.7.3:** Replaced the previous oMLX 0.7.1.dev1 build. Used revised model-specific oQe calibration with WLS Refit, iMatrix and MTP support. Improvements reported for other model architectures were not assumed to apply to this large MoE checkpoint.

## Local testing — November 19, 2026 rebuild

Two demanding reasoning exercises were run locally on an M5 Max with 128 GB unified memory using oMLX 0.8.0:

| Test | Prefill | Generation | Thinking time | Total time |
| :--- | ---: | ---: | ---: | ---: |
| OBSIDIAN-40 | 745.1 tok/s | 80.1 tok/s | 382.2 s | 461.0 s |
| SILBERGRAT | 1,897.9 tok/s | 71.5 tok/s | 475.2 s | 625.5 s |

The outputs were generally strong but not error-free. A small numerical inconsistency appeared in OBSIDIAN-40. SILBERGRAT was run with a lightly corrected test prompt. These are individual, non-standardized runs, not controlled evidence that quantization increased reasoning accuracy.

For comparison, the previous oMLX 0.7.3-quantized checkpoint, measured under oMLX 0.7.4.dev1, generated 75.3 tok/s (OBSIDIAN-40) and 68.9 tok/s (SILBERGRAT). Different runtime versions and prompt revisions limit direct attribution of these differences to quantization alone.

## Historical testing — previous oMLX 0.7.1.dev1 build

**Important:** The following qualitative observations concern the earlier oQ4e WLSRefit MTP checkpoint created with oMLX 0.7.1.dev1. They are retained for transparency and must not be read as results for the 0.7.3 or 0.8.0 rebuilds.

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

## Compatibility

Requires a compatible [oMLX](https://github.com/jundot/omlx) runtime. This checkpoint was built with **oMLX 0.8.0**.

This is an **oMLX/oQ checkpoint**, not a GGUF model. Lightning MTP acceleration depends on runtime and configuration support.

## Credits and License

- **Original model:** [Qwen Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Quantization software:** [oMLX](https://github.com/jundot/omlx), developed by Jun Kim and contributors.
- **Quantized and published by:** GBP-DE
- **License:** [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). Original license terms apply.
