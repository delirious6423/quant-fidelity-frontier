---
license: other
license_name: glm-5.3
license_link: https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE
base_model: zai-org/GLM-5.3
base_model_relation: quantized
quantized_by: LibertAIDAI
tags:
  - nvfp4
  - fp4
  - blackwell
  - vllm
  - sglang
  - glm
  - glm-5
  - glm_moe_dsa
  - moe
  - modelopt
language:
  - en
  - zh
pipeline_tag: text-generation
---

<div align="center">

# GLM-5.3 · NVFP4

### 753B total · 256 routed experts · DeepSeek-sparse attention · 1M context

**1403 GiB → 433 GiB** &nbsp;·&nbsp; **−69%** &nbsp;·&nbsp; round-trip cosine **0.9961**

[![Base](https://img.shields.io/badge/base-zai--org%2FGLM--5.3-1f6feb?style=for-the-badge)](https://huggingface.co/zai-org/GLM-5.3)
[![Format](https://img.shields.io/badge/format-NVFP4-6f42c1?style=for-the-badge)](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
[![Fits](https://img.shields.io/badge/fits-4%C3%97B200-2da44e?style=for-the-badge)](#-hardware)

*NVFP4 quantization of Z.ai's GLM-5.3, following NVIDIA's own GLM-5.2 recipe tensor-for-tensor.*

**Quantized by [LibertAI](https://libertai.io)** · not affiliated with Z.ai / Zhipu / NVIDIA

</div>

---

## ✨ What this is

An **NVFP4** checkpoint of GLM-5.3. The **routed-expert FFN** tensors — 57,600 of them, the overwhelming majority of the model's weight — are quantized to **NVFP4** (E2M1, FP8-E4M3 per-16-block scales, FP32 per-tensor global scale). Everything outlier-sensitive stays in **BF16**:

> all DeepSeek-sparse attention including the sparse indexer · all shared experts · the MoE routers · the three dense layers (0–2) · the **entire MTP layer 78**, its experts included · token embeddings · `lm_head` · all norms

Activations are not calibrated — see [the `input_scale` note](#-about-input_scale) below, which is the one genuinely interesting thing we found while making this.

---

## 🎯 The recipe is not ours — and that is the point

Z.ai state it plainly in the GLM-5.3 model card:

> *"GLM-5.3 uses the same base model as GLM-5.2 — every gain comes from post-training."*

The checkpoints bear that out exactly: GLM-5.3 and GLM-5.2 are both `glm_moe_dsa`, both **753,329,940,480** parameters, with identical `config.json` geometry. So rather than invent a partition, we **reproduce the one NVIDIA ships in [`nvidia/GLM-5.2-NVFP4`](https://huggingface.co/nvidia/GLM-5.2-NVFP4)** — a checkpoint that has been serving production traffic on our 4×B200 box since June 2026.

We verified the reproduction two ways **before quantizing a single weight**:

| check | result |
|---|---|
| emitted tensor-name set vs `nvidia/GLM-5.2-NVFP4` | **232,385 / 232,385 identical** |
| `quantization_config.ignore` vs NVIDIA's | **156 / 156 identical** |
| rest of `quantization_config` | identical except `producer.version` (we report the ModelOpt we actually ran) |

And once the weights were built, the finished checkpoint closed the loop: its summed `total_size` is **464,795,267,072 bytes — byte-identical to NVIDIA's**. Same names, same shapes, same dtypes, same byte count.

Concretely, what gets quantized is:

```
model.layers.{3..77}.mlp.experts.{0..255}.{gate,up,down}_proj     # 75 × 256 × 3 = 57,600
```

…and nothing else. Layers 0–2 are dense (`first_k_dense_replace: 3`) and layer 78 is the MTP head; NVIDIA keeps both entirely in BF16, so we do too.

---

## 🔬 About `input_scale`

Each quantized tensor here ships four entries — `.weight`, `.weight_scale`, `.weight_scale_2`, and **`.input_scale`**. That last one is **exactly `1.0`** for all 57,600 tensors, and that is deliberate.

We sampled 24 `.input_scale` values out of `nvidia/GLM-5.2-NVFP4` by HTTP range-read. **Every one is 1.0.** NVIDIA — who wrote ModelOpt — performed no activation calibration for that checkpoint; the tensors are unit placeholders. So in substance this is a **weight-only** NVFP4 quantization, and we did not fabricate a calibration we did not run.

But the tensors must still be **present**. vLLM's `ModelOptNvFp4FusedMoE` folds `input_scale` into the dequantization alpha:

```
alpha = weight_scale_2 * input_scale
```

When the tensor is *absent*, vLLM folds an **uninitialised** `PerTensorScaleParameter` — observed as `0.0` — which zeroes every expert and makes the model emit garbage. We hit exactly this on our own earlier weight-only GLM-5.3-Flash checkpoint and filed it as [vllm#54189](https://github.com/vllm-project/vllm/issues/54189). Emitting `1.0` makes the fold a no-op and is precisely what the production-proven NVIDIA checkpoint does.

**If you are quantizing an NVFP4 MoE yourself: emit `input_scale`, even if you did not calibrate.** That is the takeaway.

---

## 🖥 Hardware

| GPUs | VRAM | Verdict |
|---|---|---|
| **4× B200** (183 GB) | 732 GB | ✅ fits, with room for a large KV pool |
| 8× H200 / H100 | 640–1128 GB | ✅ fits |
| 8× RTX PRO 6000 (96 GB) | 765 GB | ✅ fits |
| 4× RTX PRO 6000 | 384 GB | ❌ weights alone are 433 GiB |

The FP8 release (~700 GiB) needs 8× B200; this NVFP4 build is what makes GLM-5.3 a **4-GPU** model.

---

## 🚀 Serving

Unlike GLM-5.3-Flash — a genuinely new `glm5_next` architecture that needed per-model engine images — **GLM-5.3 is `glm_moe_dsa`, which vLLM and SGLang already support.** No custom build, no patches.

```bash
vllm serve LibertAIDAI/GLM-5.3-NVFP4 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice --tool-call-parser glm47 \
  --reasoning-parser qwen3 \
  --served-model-name glm-5.3
```

Notes carried over from running its GLM-5.2 sibling in production:

- **`--tool-call-parser glm47`** and **`--reasoning-parser qwen3`** are the right pair for this family.
- 64 attention heads divide cleanly by TP 2 / 4 / 8.
- `reasoning_effort` accepts `low`, `high`, `max` — and **defaults to `max`**, so budget `max_tokens` generously.
- The chat template's `clear_thinking` defaults to `false`; pass `clear_thinking=true` for chat-style use.

---

## 🔧 How it was made

CPU-only shard-streaming pass with [NVIDIA ModelOpt](https://github.com/NVIDIA/TensorRT-Model-Optimizer) — each of the 282 source shards is fetched, quantized, written, and dropped, so the job never holds more than a few shards at once and needs nowhere near 1.37 TiB of scratch.

Source is [`zai-org/GLM-5.3-BF16`](https://huggingface.co/zai-org/GLM-5.3-BF16), not the FP8 release: the FP8 repo carries a `modules_to_not_convert` list, the signature of a converted checkpoint, so BF16 is the original and quantizing from it avoids stacking two lossy conversions.

Round-trip fidelity, measured by dequantizing back to BF16 and taking cosine similarity against the source weights: **mean 0.9961** across sampled expert tensors.

The quantizer, the config builder, and the validation harness are published in full: **[server-admin/servers/glm53-nvfp4-files](https://github.com/Libertai)**.

---

## 🙏 Credits

**[Z.ai](https://huggingface.co/zai-org)** for GLM-5.3 and for releasing a BF16 checkpoint alongside the FP8 · **[NVIDIA](https://github.com/NVIDIA/TensorRT-Model-Optimizer)** for ModelOpt and for the GLM-5.2 recipe this follows · quantized and published by **[LibertAI](https://libertai.io)**.

*Quantization only — no weights were retrained, distilled, or otherwise modified beyond the numeric format described above.*
