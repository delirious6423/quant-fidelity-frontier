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
  - oq4e
  - imatrix
  - mtp
  - apple-silicon
  - wls-refit
---

# Qwen3.8-Flash-Next — oQ4e WLSRefit MTP

An enhanced, importance-aware quantization of the original
**Qwen3.8-Flash-Next**, created with **oMLX 0.7.1.dev1**
and its improved **Weighted Least-Squares (WLS) Refit**.

## Why this version?

- **Improved oQe quantization:** The new WLS Refit is designed
  to better preserve the original model's quality.
  The oMLX developer reports approximately 11–15% lower
  KL divergence in quantizer evaluations.
- **iMatrix-guided quantization** to preserve sensitive weights.
- **MTP preserved** for compatible accelerated decoding.
- **BF16 non-quantized weights.**
- **Designed for Apple Silicon** with compatible oMLX runtimes.

## Tested and reviewed with GPT-6

This model was evaluated using demanding reasoning and
analytical challenges, with its outputs reviewed by GPT-6.

Compared with an earlier oQ4e conversion of the same base
model, this new build showed a **noticeable improvement in
practical output quality** during local testing.

In the 60-question **Silbergrat Reasoning Challenge**,
it demonstrated:

- Strong logical coherence and evidence evaluation.
- Substantially fewer serious reasoning inconsistencies.
- Much better language stability than the earlier conversion.
- Careful handling of uncertainty and competing explanations.

A separate **XAGUSD quantitative trading research challenge**
also demonstrated strong analytical capabilities, producing
testable hypotheses without inventing profitability claims.

**Overall assessment:** Promising improvement in practical
reasoning quality compared with the previously tested oQ4e
build, particularly for complex, long-form analytical tasks.

These observations are qualitative, not standardized
benchmarks. The runtime version also changed, so improvements
cannot be attributed exclusively to WLS Refit.

## Compatibility

Requires a compatible [oMLX](https://github.com/jundot/omlx)
runtime. This is an oMLX/oQ checkpoint, not a GGUF model.

## Credits and License

- **Original model:** [Qwen Team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- **Quantization software:** [oMLX](https://github.com/jundot/omlx)
- **Quantized and published by:** GBP-DE
- **License:** Qwen Community License 1.0.
  Original license terms apply.