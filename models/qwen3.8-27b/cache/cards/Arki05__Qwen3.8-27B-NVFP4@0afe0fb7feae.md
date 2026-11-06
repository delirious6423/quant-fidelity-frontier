---
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: vllm
tags:
- nvfp4
- compressed-tensors
- vllm
- fp4
---

# Qwen3.8-27B NVFP4

[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) with every linear layer of its 64 text blocks in NVFP4,
Hessian-quantized from Qwen's BF16 weights, as a [compressed-tensors](https://github.com/vllm-project/compressed-tensors)
checkpoint for [vLLM](https://github.com/vllm-project/vllm).

- **Weights only (W4A16):** FP4 (E2M1) values, an FP8 (E4M3) scale per 16 values and an FP32 scale per tensor;
  activations stay in 16 bits. Layers vLLM fuses (q/k/v, gate/up, the linear attention's input projections) share
  their per-tensor scale.
- **Quantized:** the 496 linear layers of the text blocks (attention, linear attention, MLP), 24.4 B parameters.
- **Unchanged (BF16, as in the original):** the embedding, `lm_head`, the vision encoder and the multi-token-prediction
  layer.
- **Size:** 20.6 GB, against about 54 GB in BF16.

## Use

```sh
vllm serve Arki05/Qwen3.8-27B-NVFP4
```

No quantization flag is needed: vLLM reads the format from `config.json` and runs it as weight-only FP4 (Marlin).
Tested with vLLM 0.33.0 on 2× RTX 5080 16 GB, text only:

```sh
vllm serve Arki05/Qwen3.8-27B-NVFP4 --tensor-parallel-size 2 --max-model-len 4096 \
  --max-num-batched-tokens 1024 --max-num-seqs 4 --limit-mm-per-prompt '{"image":0,"video":0}'
```

The weights take about 9 GB per GPU there; on a single 24 GB+ GPU the defaults apply.

## How it was made

Each layer's weights were chosen with second-order information about the layer's inputs and about how its outputs
affect the model's next-token distribution, measured on a mix of web text, encyclopedic and multilingual text, math,
code, science, instruction following, tool use and long-context documents, including the model's own responses.
Every layer was quantized from the original BF16 weights; nothing here was converted from another quantization.

## Quality

KL divergence against the BF16 model on held-out text follows; this card will be updated with it. A GGUF build of the
same model at many sizes, with measurements, is at [Arki05/Qwen3.8-27B-GGUF](https://huggingface.co/Arki05/Qwen3.8-27B-GGUF).

Apache-2.0, as the original model. Modified versions of the original weights.
