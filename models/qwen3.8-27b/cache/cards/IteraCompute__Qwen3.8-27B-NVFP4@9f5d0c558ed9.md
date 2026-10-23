---
license: apache-2.0
language:
  - en
  - zh
base_model: Qwen/Qwen3.8-27B
base_model_relation: quantized
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - qwen3.8
  - nvfp4
  - modelopt
  - blackwell
  - rtx-5090
  - vllm
  - quantization
  - conversational
---

# Qwen3.8-27B NVFP4 · IteraCompute

An open-weight release of the Qwen3.8-27B NVFP4 checkpoint deployed by [IteraCompute](https://iteracompute.com/), made available here for community use and reproducible deployment on NVIDIA Blackwell GPUs.

**Provenance:** the original model is from [Qwen](https://huggingface.co/Qwen/Qwen3.8-27B), and this NVFP4 quantization is from [Gittensor Model Hub](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090). IteraCompute hosts this verified checkpoint and contributes deployment documentation. The weights and serving files are unchanged from the pinned source; we do not claim authorship of the original model or quantization. See [release.json](release.json) for the exact source revision and [SOURCE_MODEL_CARD.md](SOURCE_MODEL_CARD.md) for the preserved source documentation.

中文：这是 IteraCompute 正在部署的 Qwen3.8-27B NVFP4 检查点，现公开托管，供社区下载和部署。原始模型来自 Qwen，NVFP4 量化来自 Gittensor Model Hub。我们保留原始权重、配置、tokenizer、许可证及上游声明，并提供来源校验和部署说明；不将上游量化工作署为 IteraCompute 原创。

## Checkpoint

| Item | Details |
| --- | --- |
| Model | Qwen3.8-27B |
| Quantization | NVIDIA Model Optimizer NVFP4; see `config.json` and `hf_quant_config.json` for layer coverage |
| Weight files | Two Safetensors shards, 17,916,112,584 bytes in total (about 17.92 GB) |
| Configuration architecture | `Qwen3_5ForConditionalGeneration` / `qwen3_5`, as supplied upstream |
| Configured maximum context | 262,144 tokens, input and output combined |
| IteraCompute deployment | vLLM 0.31.0, one RTX 5090 32 GB per instance, FP8 KV cache |
| License | Apache 2.0; retain [LICENSE](LICENSE) and upstream notices |

NVFP4 execution requires compatible Blackwell hardware and engine support. Available context and concurrency depend on engine settings and memory. The checkpoint's context configuration is separate from the limits of any hosted API plan. This release does not add a new quality evaluation or benchmark; figures in the preserved source card belong to its authors and their stated test conditions.

## Download and verify

```bash
hf download IteraCompute/Qwen3.8-27B-NVFP4 --local-dir ./Qwen3.8-27B-NVFP4
cd Qwen3.8-27B-NVFP4
sha256sum -c SHA256SUMS
```

For reproducible use, add `--revision <full-commit-sha>` with a commit from this repository's history. `SHA256SUMS` covers the checkpoint, serving files, and preserved upstream documentation. `release.json` records the source commit and confirms comparison with the deployed checkpoint.

## Serve with vLLM

The following settings reflect IteraCompute's vLLM 0.31.0 deployment. Install an appropriate CUDA/Blackwell build of vLLM in a separate environment, then run:

```bash
vllm serve ./Qwen3.8-27B-NVFP4 \
  --served-model-name qwen3.8-27b-nvfp4 \
  --host 127.0.0.1 --port 8000 \
  --quantization modelopt \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --gpu-memory-utilization 0.97 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --kv-cache-dtype fp8 \
  --attention-backend FLASHINFER \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b-nvfp4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}'
```

The example binds to localhost. Engine and driver compatibility must be checked for your environment. Use the included chat template for reasoning and tool calls; further upstream details are in [SOURCE_MODEL_CARD.md](SOURCE_MODEL_CARD.md).

## Qwen 4 NVFP4 is next

**IteraCompute plans to upload a Qwen 4 NVFP4 version as soon as possible after the official model becomes available and licensing, quantization, and inference validation are complete.** Follow our [Qwen4-27B-NVFP4 repository](https://huggingface.co/IteraCompute/Qwen4-27B-NVFP4) for the release. That repository is currently a preparation page with no Qwen 4 weights; its name is a planned target, subject to the official model lineup.

**中文：我们后续也会第一时间推进并上传 Qwen 4 的 NVFP4 版本。** 将在官方模型可用、确认许可证并完成量化及推理验证后尽快发布。预告仓库已准备好：[IteraCompute/Qwen4-27B-NVFP4](https://huggingface.co/IteraCompute/Qwen4-27B-NVFP4)。目前尚未提供 Qwen 4 权重，最终型号以官方发布为准。

## About IteraCompute

IteraCompute provides OpenAI-compatible model inference. Visit our [website](https://iteracompute.com/), [API documentation](https://iteracompute.com/docs.html), and [current free Qwen API offering](https://iteracompute.com/free/) for hosted access and current terms.

This is an independent community distribution, unaffiliated with Qwen or NVIDIA. Original model and quantization credit remains with the respective upstream authors.
