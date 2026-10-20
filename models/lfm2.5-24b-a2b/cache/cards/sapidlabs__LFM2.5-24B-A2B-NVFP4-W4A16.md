---
library_name: transformers
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: text-generation
base_model: LiquidAI/LFM2.5-24B-A2B
base_model_relation: quantized
tags:
- liquid
- lfm2.5
- nvfp4
- modelopt
- vllm
- dgx-spark
---

# LFM2.5-24B-A2B-NVFP4-W4A16

An NVFP4 weight-only quantization of [LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B), made by Sapid Labs. Liquid AI did not make or review this checkpoint.

It is 14.7 GB, against 48 GB for the BF16 release. On one DGX Spark it serves 64 concurrent agents at 859 tok/s in total.

## What is quantized

| Layers | Format |
|---|---|
| Routed experts (`w1`, `w2`, `w3` of 64 experts in 38 layers) | NVFP4 W4A16 |
| Dense feed-forward layers (layers 0 and 1) | NVFP4 W4A16 |
| Short convolution blocks, attention, router gates, embeddings (tied to the output head) | BF16 |

NVFP4 W4A16 stores weights as 4-bit E2M1 values with one FP8 E4M3 scale per block of 16 and one FP32 scale per tensor. Activations stay in BF16. The checkpoint uses ModelOpt's `MIXED_PRECISION` format, the layout of NVIDIA's own NVFP4 releases.

Weights were rounded to nearest with ModelOpt's `NVFP4QTensor.quantize`, with no calibration data. Each gate and up projection (`w1`, `w3`) shares one per-tensor scale, because vLLM fuses the pair into one matrix. The mean relative error of the dequantized weights against BF16 is 0.095. `scripts/convert_nvfp4.py --preset lfm2` is the script that made this checkpoint.

## Running it

Tested with vLLM nightly `0.31.1rc1.dev296+gfe4df3df0` (image `vllm/vllm-openai:nightly-fe4df3df0eb703862c686b256e9fe44fa94c8945`) on a DGX Spark. vLLM detects the format as `modelopt_mixed` without a `--quantization` flag and runs the FP4 layers on its Marlin W4A16 kernels.

```bash
vllm serve sapidlabs/LFM2.5-24B-A2B-NVFP4-W4A16 \
  --max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching --gpu-memory-utilization 0.80 \
  --enable-auto-tool-choice --tool-call-parser lfm2 --reasoning-parser qwen3
```

The model always reasons before it answers. `"chat_template_kwargs": {"enable_thinking": false}` does not stop the reasoning; it only makes vLLM return the reasoning inside `content` instead of separating it.

## Measured on one DGX Spark (GB10, 128 GB)

These come from our swarm benchmark: the serve command above (`scripts/serve_lab.sh`), then closed-loop agents that share a system prompt, grow their context every turn and never pause between turns, with 250-token replies. Qwen3.6-35B-A3B NVFP4 ran on the same Spark with the same settings, with thinking off.

| Agents | This model: total, per agent, first token | Qwen3.6-35B-A3B NVFP4 |
|---:|---|---|
| 1 | 59 tok/s, 62 tok/s, 0.14 s | 60 tok/s, 73 tok/s, 0.24 s |
| 4 | 153 tok/s, 41 tok/s, 0.20 s | 133 tok/s, 37 tok/s, 0.26 s |
| 16 | 328 tok/s, 22 tok/s, 0.24 s | 231 tok/s, 16 tok/s, 0.39 s |
| 32 | 549 tok/s, 18 tok/s, 0.32 s | 281 tok/s, 10 tok/s, 0.59 s |
| 64 | 859 tok/s, 14 tok/s, 0.39 s | 318 tok/s, 7 tok/s, 3.12 s |

Weights take 13.75 GiB. At `--gpu-memory-utilization 0.80` the KV cache holds 4,210,225 tokens.

## Quality

Temperature 0, thinking on, 8 requests at a time.

| Test | This model | BF16 original |
|---|---:|---:|
| GSM8K, first 200 test questions | 182/200 (91.0%) | 180/200 (90.0%) |
| Tool calls: the right one of six tools, or none, with the right arguments | 20/20 | 20/20 |
| JSON extraction without constrained decoding | 14/15 | 13/15 |

The BF16 runs had a smaller output limit (6,000 tokens for GSM8K, 1,500 for JSON) than the NVFP4 runs (12,000 and 8,000). No NVFP4 reply reached its limit.

## License

The base model is released by Liquid AI under the LFM Open License v1.0, and this checkpoint is distributed under the same license. `LICENSE` holds its text. The license limits commercial use to organizations with annual revenue under 10 million US dollars.

Sapid Labs changed these files: the `model-*.safetensors` weights (quantized as described above), `model.safetensors.index.json`, `config.json` (a `quantization_config` was added) and this `README.md`. `hf_quant_config.json` and `scripts/` are new. Liquid AI's original model card is kept unchanged as `ORIGINAL_MODEL_CARD.md`.
