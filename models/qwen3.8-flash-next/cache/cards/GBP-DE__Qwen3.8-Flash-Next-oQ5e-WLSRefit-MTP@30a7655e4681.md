
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
  - oq5e
  - imatrix
  - mtp
  - apple-silicon
  - wls-refit
---

# Qwen3.8-Flash-Next — oQ5e WLSRefit MTP | oMLX 0.8.0 Rebuild

| 🚀 **MODEL UPDATE — November 19, 2026** |
| :--- |
| **Fresh 5-bit rebuild with oMLX 0.8.0.** This repository has been updated in place, replacing the November 1 build made with oMLX 0.7.3. |
| **Why update?** Jun Kim's refined **oQe quantization** is designed to reduce quantization error at the same nominal precision and model size. Re-quantization is needed to benefit from these improvements; updating oMLX alone does not change existing model weights. |
| **Retained features:** oQ5e, WLS Refit, iMatrix-guided quantization, MTP weights and BF16 for unquantized tensors where applicable. |
| **Initial local tests:** OBSIDIAN-40 and SILBERGRAT produced generally strong reasoning results. Generation speed improved in the SILBERGRAT run but was nearly unchanged in OBSIDIAN-40. Total execution times increased. A consistent accuracy gain over the previous checkpoint has **not** been established. |
| **Already downloaded an earlier build?** Download the updated model files again. The repository URL remains unchanged. See also the [4-bit WLSRefit MTP build](https://huggingface.co/GBP-DE/Qwen3.8-Flash-Next-oQ4e-WLSRefit-MTP) and the [oMLX 0.8.0 release notes](https://github.com/jundot/omlx/releases/tag/v0.8.0). |

## About this model

A quality-focused, importance-aware **oQ5e quantization** of the original **Qwen3.8-Flash-Next**, intended for inference on Apple Silicon using compatible oMLX runtimes.

The current checkpoint was rebuilt with **oMLX 0.8.0**.

Compared with oQ4e, oQ5e prioritizes higher weight precision and model fidelity at the cost of increased memory requirements.

Its objective is to preserve the original model's capabilities while retaining support for accelerated MTP decoding.

## Quantization features

- **oQ5e:** Importance-aware quantization with a higher precision budget than oQ4e.
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
| OBSIDIAN-40 | 1,532.4 tok/s | 67.1 tok/s | 531.3 s | 621.4 s |
| SILBERGRAT | 485.4 tok/s | 70.7 tok/s | 358.5 s | 540.2 s |

The outputs demonstrated strong reasoning and analytical capabilities, but were not error-free. A minor timestamp error occurred in SILBERGRAT. The test prompt had also received a small correction since earlier runs.

These are individual, non-standardized tests and do not establish a consistent accuracy improvement over previous quantizations.

For comparison, the previous oMLX 0.7.3-quantized checkpoint, tested under oMLX 0.7.4.dev1, generated 67.6 tok/s (OBSIDIAN-40) and 60.8 tok/s (SILBERGRAT).

Different runtime versions, prompt revisions and generation lengths limit direct attribution of performance differences to quantization alone.

## Historical reasoning evaluation — previous builds

**Important:** The following qualitative observations concern earlier local quantizations and tests conducted before the oMLX 0.7.3 rebuild.

They are retained for transparency and must not be interpreted as benchmark results for the newer 0.7.3 or 0.8.0 checkpoints.

### Earlier testing with GPT-6

Previous local testing with GPT-6 found that an oQ4e conversion using the newer WLS Refit approach produced substantially more coherent results than an earlier oQ4e conversion of the same base model.

The demanding 60-question **Silbergrat Reasoning Challenge** demonstrated:

- Improved language stability.
- Strong logical coherence and evidence evaluation.
- More careful handling of uncertainty.
- Fewer serious reasoning inconsistencies.

The earlier oQ5e WLSRefit checkpoint used the same quantization approach with higher precision, aiming to preserve more of the original model's capabilities.

However, that earlier oQ5e build was not independently benchmarked against the other variants at the time of those observations.

The observed improvements in the historical oQ4e comparison could not be attributed exclusively to WLS Refit because the runtime version also changed.

These findings are qualitative observations rather than standardized benchmark results.

## Compatibility

Requires a compatible [oMLX](https://github.com/jundot/omlx) runtime. This checkpoint was built with **oMLX 0.8.0**.

This is an **oMLX/oQ checkpoint**, not a GGUF model.

Lightning MTP acceleration depends on runtime and configuration support.

Due to the model's higher memory requirements, sufficient unified memory is essential, especially when using large context windows. SSD-backed offloading and memory configuration may affect practical usability.

## Credits and License

- **Original model:** [Qwen Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Quantization software:** [oMLX](https://github.com/jundot/omlx), developed by Jun Kim and contributors.
- **Quantized and published by:** GBP-DE
- **License:** [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). Original license terms apply.
