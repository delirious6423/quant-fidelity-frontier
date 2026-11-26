---
license: mit
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- tensorfold
- tfq3
- quantized
- apple-silicon
- metal
- mtp
---

# GLM-5.3-Flash-TFQ3

A **167.50 GB (156.00 GiB)** TensorFold checkpoint based on [Z.ai’s GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), with 4-bit TFQ3 routed expert weights and the MTP head.

<p align="center">
  <img src="assets/glm-logo.svg" alt="Z.ai GLM" width="88" />
</p>

## Before you download

This checkpoint requires a TFQ3-enabled `tensorfold-native` build with Metal 4 and NAX support. The tested device was an Apple M5 Ultra. Keep the TFQ3 files and all 63 Safetensors shards together.

If you downloaded an earlier revision, download this revision into a fresh directory so the runtime loads the complete replacement checkpoint.

## WikiText-2 quality and memory

| Metric | FP8 reference | TFQ3 |
| --- | ---: | ---: |
| Weight buffers in RAM | 318.60 GB | 166.36 GB |
| Chunk 1 NLL, 2,047 scored tokens | 0.661032 | 0.683424 |
| Chunk 2 NLL, 2,047 scored tokens | 0.508224 | 0.528005 |
| Mean NLL, 4,094 scored tokens | 0.584628 | 0.605715 |
| Perplexity, 4,094 scored tokens | 1.7943 | 1.8326 |

On this evaluation, TFQ3 used **47.8% less memory for weight buffers** with **2.1% higher perplexity** than the FP8 reference.

## Apple M5 Ultra serving

The reported shared-machine run measured **54.0–58.3 tok/s** with MTP. MTP depth 3 matched depth 0 token for token in the reported verification.

## Licence and credit

The original model and MTP head are by Z.ai. TensorFold publishes this quantized checkpoint under the upstream [MIT licence](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE).
