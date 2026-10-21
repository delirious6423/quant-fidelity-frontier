---
license: mit
library_name: sglang
base_model:
- deepseek-ai/DeepSeek-V4-Flash-0731
base_model_relation: quantized
tags:
- deepseek-v4
- deepseek-v4-flash
- fp8
- quantized
- safetensors
- sglang
- dspark
---

# DeepSeek-V4-Flash-0731-FP8

This is an **unofficial FP8 format conversion** of
[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731),
created from upstream revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`. It is not an official DeepSeek
release and does not claim the upstream model's published benchmark results.

The model weights and this repository are distributed under the upstream
[MIT License](LICENSE).

## Format

The upstream checkpoint combines MXFP4 routed experts with MXFP8 dense,
attention, shared-expert, and MTP tensors. This conversion presents a single
FP8 layout for SGLang:

| Item | Released format |
| --- | --- |
| Weight dtype | `float8_e4m3fn` |
| Scale dtype on disk | `float32` |
| Weight block size | `[128, 128]` |
| Quantization method | `fp8` |

The float32 scales are exact casts of the source E8M0 powers of two. The
checkpoint deliberately has no `scale_fmt` field: the tested SGLang FP8 loader
reads these scales as float32 and prepares its runtime representation itself.
See [`conversion/manifest.json`](conversion/manifest.json) for the complete
format and verification record.

## Conversion verification

- Converted routed-expert tensors: **35,328**
- Source scale tensors cast exactly to float32: **71,046**
- Routed-expert 128x128 blocks swept: **18,087,936**
- Blocks with scale spread above the lossless threshold: **0**
- Maximum observed dequantization error in the deterministic verification
  sample: **0.0**

The released `conversion/` scripts reproduce the conversion and verification.
They require PyTorch with float8 support and `safetensors`; run them on a
separate source and output directory.

```bash
python conversion/convert_fp4_to_fp8_f32_scales.py \
  --source /path/to/DeepSeek-V4-Flash-0731 \
  --output /path/to/DeepSeek-V4-Flash-0731-FP8

python conversion/verify_lossless_conversion.py \
  --source /path/to/DeepSeek-V4-Flash-0731 \
  --converted /path/to/DeepSeek-V4-Flash-0731-FP8 \
  --output error-metrics.json
```

## Validated serving path

This checkpoint was validated with SGLang on AMD MI325X using four-way tensor
parallelism and data parallelism. The following is the tested serving family;
adapt hardware-dependent parallelism and memory values to your deployment.

```bash
python -m sglang.launch_server \
  --model-path long10024070/DeepSeek-V4-Flash-0731-FP8 \
  --served-model-name deepseek-v4-flash-fp8 \
  --host 0.0.0.0 --port 30000 \
  --trust-remote-code --tp-size 4 --dp-size 4 --enable-dp-attention \
  --quantization fp8 --mem-fraction-static 0.85 \
  --attention-backend dsa --dsa-prefill-backend aiter \
  --dsa-decode-backend aiter --dsa-topk-backend sgl-kernel \
  --kv-cache-dtype fp8_e4m3 --enable-aiter-allreduce-fusion \
  --moe-runner-backend auto --cuda-graph-max-bs-decode 8
```

No speculative-decoding flags were enabled for the validation runs. This
repository does not claim compatibility with Transformers, vLLM, or other
runtimes that have not been tested against this exact layout.

## Chat formatting

Like the upstream release, this model does not include a Jinja chat template.
Use the included [`encoding/`](encoding) utilities to encode OpenAI-style
messages and parse completions. They are copied unchanged from the exact
upstream revision used for conversion.

## Integrity

`SHA256SUMS` contains SHA-256 hashes for every release payload file except the
checksum manifest itself. Verify a downloaded snapshot with:

```bash
sha256sum -c SHA256SUMS
```

## Citation

Please cite the original model and technical report when using this release:

```bibtex
@misc{deepseekai2026deepseekv4,
  title={DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence},
  author={DeepSeek-AI},
  year={2026},
}
```
