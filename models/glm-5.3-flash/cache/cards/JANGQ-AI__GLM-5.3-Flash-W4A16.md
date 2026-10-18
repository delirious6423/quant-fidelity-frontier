---
license: other
license_name: glm-5.3
license_link: LICENSE
language:
  - en
  - zh
library_name: transformers
tags:
  - jang
  - w4a16
  - int4
  - compressed-tensors
  - vllm
  - marlin
  - glm
  - glm5-next
  - moe
  - multimodal
  - hopper
  - h200
  - mtp
pipeline_tag: image-text-to-text
base_model: zai-org/GLM-5.3-Flash
base_model_relation: quantized
---

<p align="center"><img src="jangq-logo.png" width="160"/></p>

# GLM-5.3-Flash-W4A16

**GLM-5.3-Flash — ~195 GB on disk** (down from the ~643 GB BF16 source) — **INT4
W4A16** quantization of the routed experts in **compressed-tensors** pack-quantized
format, for **Hopper (H100 / H200)** via vLLM **Marlin** kernels.

- **Source:** [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16)
  — the `glm5_next` architecture: a **multimodal hybrid** model (vision tower +
  mixed linear/full attention + MoE), 46 text layers, **288 routed experts** top-8,
  1 shared expert, MTP head at layer 45.
- **Quantization:** **INT4 symmetric, group-32**, RTN, on the **routed-expert
  weights only** (`model.language_model.layers.3..44.mlp.experts.*.{gate,up,down}_proj`),
  packed bit-exactly with the `compressed-tensors` library's `pack_to_int32`
  (`weight_packed` int32 / `weight_scale` bf16 / `weight_shape` int64).
- **A16 = activations, attention, and KV cache stay 16-bit.** Only expert weight
  *storage* is 4-bit.
- **Kept at BF16 (source precision):** the **vision tower** (`model.visual.*`),
  all attention including the **linear-attention** params (`A_log`, `dt_bias`) and
  the DSA-style indexer, the **shared expert**, dense MLP (layers 0–2), router
  gates, RMSNorms, embeddings, `lm_head`, and the **MTP head** (layer 45
  `eh_proj`/`enorm`/`hnorm`).
- **Bundle:** ~195 GB · **37,152** expert tensors quantized (verified == expected),
  MTP + vision preserved.

## What's in the bundle

| Module | Source dtype | Bundle dtype |
|---|---|---|
| Routed experts (288 × 3 × MoE layers 3–44) | BF16 | **INT4 sym g32** (`weight_packed`/`weight_scale`/`weight_shape`) |
| Vision tower (`model.visual.*`) | BF16 | **BF16** |
| Attention (full + linear `A_log`/`dt_bias`, indexer) | BF16 | **BF16** |
| Shared expert, dense MLP (layers 0–2) | BF16 | **BF16** |
| Router gates, norms, embed, `lm_head`, MTP (layer 45) | BF16 / F32 | **BF16 / F32** |

Dequant per group of 32: `w = int4 * weight_scale` (`scale = amax/7.5`, range `[-8,7]`).

## Usage

```bash
vllm serve JANGQ-AI/GLM-5.3-Flash-W4A16 --tensor-parallel-size 4 --trust-remote-code
```

> **Runtime note:** `glm5_next` is a very new hybrid multimodal architecture.
> Confirm your serving stack (vLLM / SGLang build) has `Glm5Next` support before
> deploying; the weights here are standard compressed-tensors W4A16 and load via
> the Marlin path once the architecture is recognized.

## Method

Streaming shard-by-shard RTN quantizer (one tensor in memory at a time — no GPU,
no large host RAM), architecture-agnostic: the routed-expert tensor set is taken
from the model's own index, packed via `compressed_tensors.pack_to_int32` for a
bit-exact on-disk layout. Data-free (no calibration). Every layer verified present
(0–45), MTP + vision preserved, expert count checked against the source index.

---
*Quantized by [JANGQ-AI](https://huggingface.co/JANGQ-AI).*
