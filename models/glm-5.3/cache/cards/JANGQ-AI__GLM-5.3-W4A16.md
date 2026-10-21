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
  - glm-moe-dsa
  - moe
  - hopper
  - h200
  - mtp
pipeline_tag: text-generation
base_model: zai-org/GLM-5.3
base_model_relation: quantized
---

<p align="center"><img src="jangq-logo.png" width="160"/></p>

# GLM-5.3-W4A16

**GLM-5.3 — ~460 GB on disk** (down from the ~1.5 TB BF16 source) — **INT4
W4A16** quantization of the routed experts in **compressed-tensors** pack-quantized
format. Loads and runs on **Hopper (H100 / H200)** via vLLM's **Marlin** kernels.

- **Source:** [zai-org/GLM-5.3](https://huggingface.co/zai-org/GLM-5.3-BF16)
  (GLM-MoE-DSA architecture, 78 layers, 256 routed experts top-8, 1 shared
  expert, MLA + DeepSeek Sparse Attention, MTP head, 1M context)
- **Quantization:** **INT4 symmetric, group-32**, RTN, on the routed-expert
  weights only (incl. MTP) — packed with the `compressed-tensors` library's own
  `pack_to_int32` for a bit-exact on-disk layout
  (`weight_packed` int32, `weight_scale` bf16, `weight_shape` int64)
- **Why W4A16 (not NVFP4 / MXFP4):** NVFP4 and MXFP4 are Blackwell-native
  formats and run through slow dequant emulation on Hopper. INT4 W4A16 →
  Marlin is the mature, fast 4-bit path on H100 / H200. **A16 = activations,
  attention math, and KV cache all stay 16-bit** — only expert weight *storage*
  is 4-bit
- **Attention & router kept high-precision:** MLA / DSA attention, dense MLP,
  the shared expert, router gates, norms, embeddings, and `lm_head` all stay
  **BF16** — only the 256 routed experts per layer are INT4, and those are
  ~97% of the parameters
- **MTP included:** the MTP layer (`model.layers.78.*`) experts are quantized
  like every other layer
- **Bundle size:** **~460 GB**
- **Runs on:** H100 / H200 with vLLM tensor-parallel

## What's in the bundle

| Module | Source dtype | Bundle dtype |
|---|---|---|
| Routed experts (`experts.*.{gate,up,down}_proj`, incl. MTP) | BF16 | **INT4 sym g32** (`weight_packed` i32 / `weight_scale` bf16 / `weight_shape` i64) |
| Attention (MLA q/kv proj, o_proj, DSA indexer) | BF16 | **BF16** |
| Dense MLP (first 3 layers) + shared expert | BF16 | **BF16** |
| Router gates, `e_score_correction_bias`, RMSNorms | BF16 / F32 | **BF16 / F32** |
| `embed_tokens`, `lm_head`, MTP glue (`eh_proj`, norms) | BF16 | **BF16** |

Dequant per group of 32: `w = int4 * weight_scale` (symmetric, `scale = amax/7.5`,
range `[-8, 7]`).

## Usage (vLLM)

```bash
vllm serve JANGQ-AI/GLM-5.3-W4A16 \
  --tensor-parallel-size 8 \
  --trust-remote-code
```

vLLM auto-selects the Marlin MoE kernel for compressed-tensors pack-quantized
INT4 at group-32 on Hopper.

## Method

Converted from the BF16 release with a streaming shard-by-shard quantizer (one
tensor in memory at a time — no GPU, no large host RAM). Round-to-nearest INT4
is packed via `compressed_tensors.pack_to_int32` so the layout is bit-exact with
what vLLM expects; the format was ground-truthed against a known-working
DeepSeek-family W4A16 model. This is a **data-free** RTN quant (no calibration).
For maximum quality a calibrated GPTQ/AWQ pass on the routed experts can be run
later on-device.

---
*Quantized by [JANGQ-AI](https://huggingface.co/JANGQ-AI).*
