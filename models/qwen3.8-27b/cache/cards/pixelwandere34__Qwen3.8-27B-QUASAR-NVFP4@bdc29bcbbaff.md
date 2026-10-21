---
library_name: transformers
pipeline_tag: image-text-to-text
base_model: Qwen/Qwen3.8-27B
tags:
  - nvfp4
  - fp4
  - quantization
  - quantization-aware-training
  - quasar
  - compressed-tensors
  - vllm
---

# Qwen3.8-27B NVFP4, trained with QUASAR

`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` is a 4-bit NVFP4 version of
[`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B), produced with
[**QUASAR**](https://arxiv.org/abs/2608.13966), a quantization-aware training (QAT) method.

Use it exactly as you would a post-training-quantized (PTQ) build of Qwen3.8-27B: it is a
drop-in, lower-precision replacement for the original model, not a fine-tune. The quantized
weights are trained by distilling directly from the frozen BF16 original as the teacher, on
the teacher's own output distribution, so the model stays anchored to the original's
behavior instead of drifting away from it. The difference from PTQ is only that the weights
are *learned* under quantization rather than rounded after the fact, which recovers more of
the original quality.

📄 Paper: [QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware
Reconstruction](https://arxiv.org/abs/2608.13966)

This model uses the most aggressive quantization settings available for Qwen3.8-27B:
**every linear layer in every transformer layer is NVFP4 (W4A4)** — self-attention, gated
delta-net, and MLP alike, 496 of 496. Usually for NVFP4 quantization of this model,
attention and gated delta-net layers are kept in higher precision (FP8 or BF16), as the
model quality can collapse when they are quantized to NVFP4. However, QUASAR preserves
quality similar to the original model even when all linear layers (attention, gated delta
net, MLPs) are quantized to NVFP4, which makes this the **smallest available NVFP4
checkpoint** of Qwen3.8-27B and **one of the highest quality**.

## How to run

Compatible with vLLM, with no conversion step:

```bash
pip install "vllm>=0.27"

vllm serve QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.85 \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

On a 32 GB card such as an RTX 5090, lower the context to `--max-model-len 65536`.

Requires an NVIDIA GPU with FP4 support (Blackwell, compute capability 10.0+).

## Quality and size comparison

We compare against the two other public NVFP4 builds of this model, both of which reach
their size by leaving much of the network above 4 bits:
`unsloth/Qwen3.8-27B-NVFP4` (23.4 GB) quantizes only 168 of 496 linears to NVFP4, holding
all of self-attention and most of the delta-net at FP8, and
`Inferact/Qwen3.8-27B-NVFP4` (26.4 GB) quantizes 304, leaving 192 of the 240 delta-net
projections in BF16.

| Model | Size | GPQA-Diamond (2 runs, n=396) | AIME26 (3 repeats, n=90) |
|---|---|---|---|
| `Qwen/Qwen3.8-27B` (BF16 original) | 55.6 GB | **0.9141** | **1.0000** |
| **`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4`** (this model) | **19.7 GB** | 0.9091 | **1.0000** |
| `unsloth/Qwen3.8-27B-NVFP4` | 23.4 GB | 0.8939 | 0.9778 |
| `Inferact/Qwen3.8-27B-NVFP4` | 26.4 GB | 0.8763 | 0.9667 |

## Training

One epoch of loss-aware NVFP4 quantization-aware distillation against the frozen BF16
teacher: global batch size 32, learning rate 1e-6, 2446 steps.

## Citation

[arxiv.org/abs/2608.13966](https://arxiv.org/abs/2608.13966)

```bibtex
@article{counathe2026quasar,
  title={QUASAR: Lowering the Loss Floor of Quantization-Aware Training with Loss-Aware Reconstruction},
  author={Counathe, Vincent and Athiwaratkun, Ben and De Sa, Christopher and Zhang, Tianyi},
  journal={arXiv preprint arXiv:2608.13966},
  year={2026}
}
```
