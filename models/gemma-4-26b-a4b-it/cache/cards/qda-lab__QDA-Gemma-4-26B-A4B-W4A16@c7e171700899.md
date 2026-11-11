---
license: apache-2.0
license_link: https://ai.google.dev/gemma/docs/gemma_4_license
base_model: google/gemma-4-26B-A4B-it
library_name: compressed-tensors
pipeline_tag: text-generation
tags:
  - qda
  - quantized
  - compressed-tensors
---

# QDA Gemma-4-26B-A4B W4A16

**Pre-release.** Evaluation is pending; the full release, with evaluation results, is planned for 2026-11-20.

Weight-only quantization of [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it): 4-bit integer weights, asymmetric, in groups of 64; 8-bit integer weights, symmetric, in groups of 64. Activations stay in 16-bit.
Stored in the compressed-tensors format.

## License

This checkpoint is derived from google/gemma-4-26B-A4B-it and is distributed under the same license (apache-2.0). The license text is in `LICENSE`, taken from the licence page the base model names; `LICENSE-SOURCE.txt` records where and when.
