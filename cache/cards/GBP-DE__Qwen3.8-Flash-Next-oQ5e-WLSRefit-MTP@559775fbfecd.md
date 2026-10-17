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

# Qwen3.8-Flash-Next — oQ5e WLSRefit MTP

A quality-focused **oQ5e quantization** of the original
Qwen3.8-Flash-Next, created using **oMLX 0.7.1.dev1**
and its improved **Weighted Least-Squares (WLS) Refit**.

## Why this version?

- **Improved oQe quantization:** The new WLS Refit aims to
  reduce quantization error. The oMLX developer reports
  approximately 11–15% lower KL divergence in quantizer
  evaluations.
- **Higher precision:** oQ5e prioritizes model fidelity
  over memory savings compared with oQ4e.
- **iMatrix-guided quantization** to preserve sensitive weights.
- **MTP preserved** for compatible accelerated decoding.
- **BF16 non-quantized weights.**
- **Designed for Apple Silicon** using compatible oMLX runtimes.

## Reasoning Quality

Earlier local testing with GPT-6 found that an oQ4e conversion
using the new WLS Refit produced substantially more coherent
results than a previously tested oQ4e conversion.

The demanding 60-question **Silbergrat Reasoning Challenge**
demonstrated improved language stability, strong evidence
evaluation, and careful handling of uncertainty.

This oQ5e release uses the same newer quantization approach
with higher precision. It is intended to preserve more of
the original model's capabilities, although its practical
quality has not yet been independently benchmarked.

The observed improvements in the oQ4e comparison cannot
be attributed exclusively to WLS Refit because the
runtime version also changed.

## Compatibility

Requires a compatible [oMLX](https://github.com/jundot/omlx)
runtime. This is an oMLX/oQ checkpoint, not a GGUF model.

Due to its higher memory requirements, sufficient unified
memory is essential, especially for longer context windows.

## Credits and License

- **Original model:** [Qwen Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Quantization software:** [oMLX](https://github.com/jundot/omlx)
- **Quantized and published by:** GBP-DE
- **License:** Qwen Community License 1.0.
  Original license terms apply.