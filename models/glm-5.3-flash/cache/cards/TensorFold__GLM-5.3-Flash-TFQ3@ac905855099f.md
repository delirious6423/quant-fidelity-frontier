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

A 139.55 GB checkpoint of [Z.ai’s GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) for the TensorFold native runtime. It includes the model’s MTP speculative head.

<p align="center">
  <img src="assets/glm-logo.svg" alt="Z.ai GLM" width="88" />
</p>

## Before you download

This checkpoint requires a TFQ3-enabled `tensorfold-native` build with Metal 4 and NAX support. The measurements below are from an Apple M5 Ultra. The checkpoint uses TensorFold’s `.tffpw` format alongside Safetensors. A standard Transformers, vLLM, or MLX-LM loader cannot open the full checkpoint.

## Checkpoint and quality comparison

| Metric | Raw FP8 | MLX 4-bit | TFQ3 |
| --- | ---: | ---: | ---: |
| Weight files on disk | 319.76 GB | 180.58 GB | **139.55 GB** |
| NLL / perplexity, 146 tokens | 2.8367 / 17.06 | 2.8858 / 17.92 | **2.7884 / 16.26** |
| Weighted NLL / perplexity, 4 prompts, 235 tokens | 3.1846 / 24.16 | 3.2316 / 25.32 | **3.1184 / 22.61** |

**TFQ3 beat the FP8 reference on perplexity in both evaluations.** Perplexity was **4.7% lower** on the 146-token test (16.26 vs 17.06) and **6.4% lower** across four prompts totalling 235 tokens (22.61 vs 24.16). TFQ3 also used **56.4% less disk space** than the FP8 reference and **41.03 GB less** than the MLX 4-bit reference.

## Apple M5 Ultra measurements

| Metric | MLX 4-bit checkpoint | TFQ3 checkpoint |
| --- | ---: | ---: |
| Prefill, 2,048-token prompt | 1,400 tok/s | **1,567 tok/s** |
| Live decode, MTP with copy drafting | — | **87–104 tok/s** |

The TFQ3 weight buffers used 138.23 GB in the measured run. TFQ3 MTP depth 3 produced the same output as depth 0 in the reported verification run.

## Licence and credit

The original model and MTP head are by Z.ai. TensorFold publishes this quantized checkpoint under the upstream [MIT licence](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE).
