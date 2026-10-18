---
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
quantized_by: coolbho3k
license: mit
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - glm5_next
  - vllm
  - compressed-tensors
  - nvfp4
  - fp8
  - quantized
---

# GLM-5.3-Flash NVFP4 Optimized

An experimental mixed-precision quantization of
[zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash),
derived from and compared against
[RedHatAI/GLM-5.3-Flash-NVFP4](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4).

## What changed

- NVFP4 routed-expert weights with repaired shared W1/W3 scale handling.
- Reconstruction-optimized NVFP4 scaling.
- Selected dense, shared-expert, and attention weights retained in FP8.
- Tensors unnecessarily converted to BF16 in the earlier export were restored
  to their native checkpoint representation.
- Tokenizer, processor, chat template, and generation configuration are
  included.

These changes improve weight reconstruction error relative to the starting
NVFP4 export.

## Quantization methodology

We started with the Red Hat NVFP4 export and used Z.ai's FP8 checkpoint as the
reference. For each routed-expert group, we tested nearby valid NVFP4 scales
and kept the one with the lowest reconstruction error. On a representative
sample of 72 matrices, mean relative weight RMSE improved from 9.1114% to
7.7185%.

We also restored passthrough tensors that had been expanded to BF16 back to
their source FP8 values. W1 and W3 keep shared scales because Marlin serves
them as one fused projection; the extra per-matrix scale tuning is applied to
W2 only.

## Serving

The tested two-node DGX Spark/SM121 recipe is available here:

https://github.com/coolbho3k/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark/tree/feature/exl3-ab-current

The recipe supports selectable NVFP4 or EXL3 weights, FP8 or native-FP4 KV
cache, DCP2, and optional DFlash2 speculative decoding.

## Limitations

- The published serving results are specific to the repository's patched vLLM
  stack on two DGX Sparks.
- Independent downstream benchmark evaluation is still recommended.
- This is a community derivative and is not an official Z.ai or Red Hat
  release.

## Acknowledgements

Thanks to Z.ai for GLM-5.3-Flash and Red Hat AI for the original NVFP4
checkpoint and quantization work.
