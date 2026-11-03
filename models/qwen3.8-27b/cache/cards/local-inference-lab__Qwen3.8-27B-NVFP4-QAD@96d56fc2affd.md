---
library_name: transformers
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
tags:
  - quantized
  - nvfp4
  - mxfp8
  - modelopt
  - distillation
  - quantization-aware-distillation
  - multimodal
  - conversational
---

*** WORK IN PROGRESS ***

# Qwen3.8-27B-NVFP4-QAD

A mixed NVFP4/MXFP8 quantization-aware distillation of
[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), exported at training
step 7,000.

The student learns from the original BF16 teacher with quantized MLP weights
in its forward pass. Distillation updates the MLP weights, text normalization
weights and BF16 student LM head. This is a trained distillation checkpoint,
not a post-training conversion of the original weights.

## Precision

| Component | Representation |
| --- | --- |
| MLP gate, up and down projections | NVFP4, 16-element blocks |
| Gated delta network projections | MXFP8, 32-element blocks; frozen |
| Full-attention query, key, value and output projections | Original BF16; frozen |
| Student LM head | Trained FP32 master rounded to BF16 |
| Text normalization weights | Trained FP32 masters |
| Token embeddings, GDN convolutions and dynamics | Original BF16; frozen |
| Vision encoder and remaining source tensors | Unchanged |

Packed NVFP4 and MXFP8 weights reconstruct to the BF16 weight values used
during training. The tokenizer, chat template, generation configuration and
multimodal processors are retained from the base model.

## Activation scales

The 192 MLP input scales are copied exactly from `Qwen3.8-27B-QAD-E1`,
whose step-4,779 weights were calibrated on 390,497,191 raw-text and chat
tokens. That calibration selected the pooled p99.999 token-row maximum
from exact BF16 histograms. These scales were not recalibrated at step 7,000.

Each dense layer has equal gate/up scales in separate tensors and an
independent down-projection scale. Serving uses calibrated NVFP4 MLP
activations and dynamic MXFP8 GDN activations, adding activation quantization
beyond the BF16 training forward.

## Format

Hugging Face safetensors with ModelOpt mixed-precision metadata. The runtime
must support the base architecture, NVFP4 dense linears, MXFP8 linears and
the BF16 full-attention/head exclusions. Packed reconstruction, serialized
tensors and copied input scales are checked. Serving quality and performance
have not been evaluated for this export.

## License

Apache 2.0, following the base model.
