---
license: mit
base_model: ornith-ai/Ornith-1.5-35B-A3B
base_model_relation: quantized
pipeline_tag: image-text-to-text
library_name: transformers
tags:
  - ornith
  - qwen3_5_moe
  - w4a16
  - compressed-tensors
  - int4
  - 4-bit
  - 4bit
  - moe
  - vllm
  - symmetric
  - quantized
  - multimodal
  - mtp
---

# Ornith-1.5-35B-A3B-W4A16-SYM

**vLLM-compatible symmetric `W4A16` (group size 32) re-quantization** of
[`ornith-ai/Ornith-1.5-35B-A3B`](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B),
in `compressed-tensors` / `pack-quantized` format.

This is the same model as `ulkaa/Ornith-1.5-35B-A3B-AWQ-INT4`, re-quantized so it
**loads and runs on NVIDIA GPUs in vLLM**. The author's original release is
**asymmetric** W4A16 (int8 per-group zero-points), which vLLM's MoE WNA16 kernels
cannot load: they dequantize with a fixed zero-point of 8 and hard-assert
`symmetric` for MoE (`AssertionError: Only symmetric quantization is supported
for MoE`). That asym build is validated for **SGLang on Intel Arc** only.

| | |
|---|---|
| Format | `compressed-tensors` / `pack-quantized` |
| Scheme | **W4A16 symmetric**, group size 32 |
| Quantized | routed MoE experts only (40 layers × 256 experts × 3 proj = 30,720 modules) |
| Vision tower | preserved, BF16 |
| MTP head | preserved, BF16 (`model-mtp.safetensors`) |
| Served on | vLLM ≥ 0.19, `CompressedTensorsWNA16MarlinMoEMethod` (Marlin backend) |

## How it was produced

Starting from the author's asymmetric checkpoint, each expert projection was:

1. **Dequantized exactly** to float32: `W = (q − zp)·s` (original bf16 scale).
2. **Re-quantized symmetric** about zero with fresh per-group scales:
   `s_new = amax(W, per 32-group) / 7.5` (stored bf16),
   `q = clamp(round(W / s_new), −8, 7)`, repacked 8 codes per int32.
3. The **zero-points were dropped** and the config flipped to
   `symmetric: true, zp_dtype: null`.

vLLM reconstructs each weight as `(q − 8)·s_new`. The cost is **one extra
re-quantization step** (~`amax/15` per group) on top of the original AWQ
quantization. Everything non-quantized (BF16 vision tower, linear/full
attention, shared experts, norms, `lm_head`, embeddings, biases, `A_log`/
`dt_bias`/`conv1d`, and the full BF16 MTP head) is **byte-identical** to the
source. The conversion script is `ornith_asym_to_sym.py` (deterministic,
resumable, validates that no zero-points and no 3-D packed tensors remain).

## Serve with vLLM

```bash
vllm serve /path/to/Ornith-1.5-35B-A3B-W4A16-SYM \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 9001 \
  --limit-mm-per-prompt '{"image":1,"video":1}' \
  --mm-processor-kwargs '{"max_pixels":401408}'
```

On two RTX 3090 (Ampere) it loads with `Using Marlin backend for WNA16 MoE
(group_size=32, num_bits=4)` and serves at ~20 GB per GPU. Use
`--kv-cache-dtype auto` on Ampere (no fp8). The reasoning/tool parsers are
model-built-in for this first-serve config.

## Quality

Verified against the release coherence gate on vLLM: multi-step arithmetic
(correct), syllogistic reasoning (correct), memoized Python (correct), and a
basic color image (correct dominant color). As with any 4-bit build, greedy
output diverges from BF16; this is a 4-bit model.

## Provenance

- Base: `ornith-ai/Ornith-1.5-35B-A3B` (MIT), a fine-tune of the Qwen3.6/3.5 A3B
  MoE family.
- Source asym build: `ulkaa/Ornith-1.5-35B-A3B-AWQ-INT4` (MIT).
- Re-quantized by the Milner Team (MIRALABS) for local vLLM serving.

## License

MIT, inherited from `ornith-ai/Ornith-1.5-35B-A3B`.
