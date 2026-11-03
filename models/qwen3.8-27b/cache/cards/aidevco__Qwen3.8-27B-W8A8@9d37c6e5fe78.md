---
library_name: transformers
license: apache-2.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - compressed-tensors
  - int8
  - w8a8
  - rocm
  - mi210
---

# Qwen3.8-27B — INT8 W8A8

A data-free INT8 quantization of [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) in the compressed-tensors
`int-quantized` format: int8 per-channel symmetric weights, dynamic per-token int8 activations. It was built to serve the
model on one AMD Instinct MI210 (gfx90a, 64 GB). That card has no FP8 matrix cores, so INT8 W8A8 is the only 8-bit format that
runs on it, on vLLM's `TritonInt8ScaledMMLinearKernel`.

Everything about the model itself (architecture, training, intended use, benchmarks, limitations) is in the
[original model card](https://huggingface.co/Qwen/Qwen3.8-27B). This card covers only what the quantization changes.

## What is quantized

| Part                                                                 | Precision                                        | Why                                                     |
| -------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------- |
| Decoder MLPs `gate_proj` / `up_proj` / `down_proj` (64 layers)       | int8 weights per channel, int8 dynamic per token | most of the bytes; fast Triton int8 scaled-mm on gfx90a |
| Full-attention `q_proj` / `k_proj` / `v_proj` / `o_proj` (16 layers) | as above                                         | as above                                                |
| Gated DeltaNet layers (`linear_attn`, 48 layers)                     | bf16                                             | recurrent state layers stay full precision              |
| Norms, embeddings, `lm_head`, MTP head, vision tower                 | bf16                                             | standard; MTP stays bf16 for speculative decoding       |

- 256 quantized modules; 36.8 GB in 18 shards (the bf16 source is about 55 GB).
- No calibration data: the weight scales come from a `memoryless-mse` observer, and the activation scales are computed at
  runtime per token.

## How it was made

- Tool: llm-compressor 0.14.0 `model_free_ptq` with compressed-tensors 0.19.0, data-free, on CPU only.
- Scheme: weights `num_bits=8`, `type=int`, `symmetric=true`, `strategy=channel`, observer `memoryless-mse`; input
  activations `num_bits=8`, `type=int`, `symmetric=true`, `strategy=token`, `dynamic=true`.
- Targets (both the checkpoint's and vLLM's module names):
  `re:^(?:model\.language_model|language_model\.model)\.layers\.\d+\.(?:mlp\.(?:gate_proj|up_proj|down_proj)|self_attn\.[qkvo]_proj)$`
- Source revision: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
- Built in 14 min with 2 worker processes. The output is deterministic: three builds from the same revision gave the same
  1,455 tensors and 36.8 GB.
- Verified after the build: every scheme field, the targets in both name forms, every shard present.

## Serving (vLLM, ROCm)

Tested with vLLM v0.31.0, v0.32.0 and v0.33.0 on one MI210. Quantization is detected from the config; do not pass
`--quantization`.

```bash
vllm serve <this repo> \
  --max-model-len=131072 --gpu-memory-utilization=0.90 --kv-cache-dtype=auto \
  --max-num-batched-tokens=4096 --language-model-only \
  --limit-mm-per-prompt='{"image":0,"video":0}' \
  --reasoning-parser=qwen3 --tool-call-parser=qwen3_coder \
  --speculative-config='{"method":"mtp","num_speculative_tokens":3}'
```

The startup log should name `TritonInt8ScaledMMLinearKernel for CompressedTensorsW8A8Int8`.

## Results

Measured on one MI210 with MTP speculative decoding (3 tokens), on vLLM v0.31.0 unless noted:

| Check                                                      | Result                                                                       |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Weights on the card                                        | 33.5 GiB                                                                     |
| KV cache capacity (`--gpu-memory-utilization=0.90`)        | 175,006 tokens (176,471 on v0.32.0); 1.34x concurrency at 131,072 tokens     |
| Output tokens per second, 1,536 prompt / 128 output tokens | 24.5 at concurrency 1, 43.7 at 2, 86.4 at 4                                  |
| Time per output token                                      | 41.6 / 47.0 / 45.3 ms at concurrency 1 / 2 / 4                               |
| Correctness checks (temperature 0, thinking on and off)    | 17 x 60 = 1,020; train trip of 155 min; a `get_weather` tool call all passed |
| lm-evaluation-harness, 50 samples per task, thinking on    | gsm8k 0.96, minerva algebra 0.74, humaneval (chat) 0.92, mbpp (chat) 0.88    |

The task scores are a 50-sample sanity check that the quantization kept the model's quality (± 0.03-0.06), not a full
benchmark. On the same card and day, a published community INT8 build of the same model ran at 23.5 / 41.9 / 84.2 output
tokens per second and scored within that noise.

## License and attribution

This is a derivative of Qwen3.8-27B by the Qwen team. It is distributed under the same license as the original, the
[Apache License 2.0](LICENSE), whose full text is in this repository's `LICENSE` file. The tokenizer, chat template and
configuration files are copied unchanged from the original repository; `config.json` adds only the quantization config.

The quantization changes the numerical precision of the weights listed above and nothing else. It is not an official Qwen
release, and the Qwen team has not reviewed or endorsed it.
